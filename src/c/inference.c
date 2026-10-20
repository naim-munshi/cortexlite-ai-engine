#include "cortexlite/inference.h"
#include "cortexlite/tensor.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CL_MAX_DIM 4096U

void cl_model_free(CLModel *model) {
    if (!model) return;
    free(model->w1);
    free(model->b1);
    free(model->w2);
    free(model->b2);
    memset(model, 0, sizeof(*model));
}

static int read_values(FILE *file, float *out, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        if (fscanf(file, "%f", &out[i]) != 1 || !isfinite(out[i])) return -1;
    }
    return 0;
}

int cl_model_load(const char *path, CLModel *model) {
    if (!path || !model) return -1;
    memset(model, 0, sizeof(*model));
    FILE *file = fopen(path, "r");
    if (!file) return -1;
    char magic[32];
    int result = -1;
    if (fscanf(file, "%31s", magic) != 1 || strcmp(magic, "CLMODEL1") != 0)
        goto done;
    if (fscanf(file, "%zu %zu %zu", &model->input_dim, &model->hidden_dim,
               &model->output_dim) != 3) goto done;
    if (model->input_dim == 0 || model->hidden_dim == 0 || model->output_dim == 0 ||
        model->input_dim > CL_MAX_DIM || model->hidden_dim > CL_MAX_DIM ||
        model->output_dim > CL_MAX_DIM) goto done;
    size_t w1_count = model->hidden_dim * model->input_dim;
    size_t w2_count = model->output_dim * model->hidden_dim;
    model->w1 = (float *)malloc(w1_count * sizeof(float));
    model->b1 = (float *)malloc(model->hidden_dim * sizeof(float));
    model->w2 = (float *)malloc(w2_count * sizeof(float));
    model->b2 = (float *)malloc(model->output_dim * sizeof(float));
    if (!model->w1 || !model->b1 || !model->w2 || !model->b2) goto done;
    if (read_values(file, model->w1, w1_count) ||
        read_values(file, model->b1, model->hidden_dim) ||
        read_values(file, model->w2, w2_count) ||
        read_values(file, model->b2, model->output_dim)) goto done;
    /* Reject unexpected trailing model content. */
    char trailing;
    if (fscanf(file, " %c", &trailing) == 1) goto done;
    result = 0;
done:
    fclose(file);
    if (result != 0) cl_model_free(model);
    return result;
}

int cl_model_predict_kernel(const CLModel *model, const float *input,
                            float *hidden, float *probabilities, CLKernel kernel) {
    if (!model || !input || !hidden || !probabilities || !model->w1 || !model->w2 ||
        !model->b1 || !model->b2) return -1;
    float *logits = (float *)malloc(model->output_dim * sizeof(float));
    if (!logits) return -1;
    if (cl_matvec_kernel(model->w1, input, hidden, model->hidden_dim, model->input_dim, kernel)) {
        free(logits);
        return -1;
    }
    for (size_t i = 0; i < model->hidden_dim; ++i) hidden[i] += model->b1[i];
    cl_relu(hidden, model->hidden_dim);
    if (cl_matvec_kernel(model->w2, hidden, logits, model->output_dim, model->hidden_dim, kernel)) {
        free(logits);
        return -1;
    }
    for (size_t i = 0; i < model->output_dim; ++i) logits[i] += model->b2[i];
    int result = cl_softmax(logits, probabilities, model->output_dim);
    free(logits);
    return result;
}

int cl_model_predict(const CLModel *model, const float *input,
                     float *hidden, float *probabilities) {
    return cl_model_predict_kernel(model, input, hidden, probabilities, CL_KERNEL_SCALAR);
}

void cl_quantized_model_free(CLQuantModel *model) {
    if (!model) return;
    free(model->w1); free(model->w2);
    free(model->s1); free(model->s2);
    free(model->b1); free(model->b2);
    memset(model, 0, sizeof(*model));
}

static int quantize_rows(const float *src, int8_t *dst, float *scales,
                         size_t rows, size_t cols) {
    for (size_t r = 0; r < rows; ++r) {
        float magnitude = 0.0f;
        for (size_t c = 0; c < cols; ++c) {
            const float v = src[r * cols + c];
            if (!isfinite(v)) return -1;
            if (fabsf(v) > magnitude) magnitude = fabsf(v);
        }
        scales[r] = magnitude > 0.0f ? magnitude / 127.0f : 1.0f;
        for (size_t c = 0; c < cols; ++c) {
            float scaled = src[r * cols + c] / scales[r];
            long rounded = lroundf(scaled);
            if (rounded > 127) rounded = 127;
            if (rounded < -127) rounded = -127;
            dst[r * cols + c] = (int8_t)rounded;
        }
    }
    return 0;
}

int cl_model_quantize(const CLModel *source, CLQuantModel *quant) {
    if (!source || !quant || !source->w1 || !source->w2 || !source->b1 || !source->b2 ||
        !source->input_dim || !source->hidden_dim || !source->output_dim ||
        source->input_dim > CL_MAX_DIM || source->hidden_dim > CL_MAX_DIM ||
        source->output_dim > CL_MAX_DIM) return -1;
    memset(quant, 0, sizeof(*quant));
    quant->input_dim = source->input_dim;
    quant->hidden_dim = source->hidden_dim;
    quant->output_dim = source->output_dim;
    size_t a = source->input_dim * source->hidden_dim;
    size_t b = source->hidden_dim * source->output_dim;
    quant->w1 = (int8_t *)malloc(a);
    quant->w2 = (int8_t *)malloc(b);
    quant->s1 = (float *)malloc(source->hidden_dim * sizeof(float));
    quant->s2 = (float *)malloc(source->output_dim * sizeof(float));
    quant->b1 = (float *)malloc(source->hidden_dim * sizeof(float));
    quant->b2 = (float *)malloc(source->output_dim * sizeof(float));
    if (!quant->w1 || !quant->w2 || !quant->s1 || !quant->s2 || !quant->b1 || !quant->b2)
        goto fail;
    for (size_t i = 0; i < source->hidden_dim; ++i)
        if (!isfinite(source->b1[i])) goto fail;
    for (size_t i = 0; i < source->output_dim; ++i)
        if (!isfinite(source->b2[i])) goto fail;
    memcpy(quant->b1, source->b1, source->hidden_dim * sizeof(float));
    memcpy(quant->b2, source->b2, source->output_dim * sizeof(float));
    if (quantize_rows(source->w1, quant->w1, quant->s1, source->hidden_dim, source->input_dim) ||
        quantize_rows(source->w2, quant->w2, quant->s2, source->output_dim, source->hidden_dim))
        goto fail;
    return 0;
fail:
    cl_quantized_model_free(quant);
    return -1;
}

/* Accumulate INT8 weights against FP32 activations, then apply row scales. */
static int quant_matvec(const int8_t *weights, const float *scales, const float *input,
                         float *out, size_t rows, size_t cols) {
    if (!weights || !scales || !input || !out || !rows || !cols) return -1;
    for (size_t r = 0; r < rows; ++r) {
        double sum = 0.0;
        for (size_t c = 0; c < cols; ++c)
            sum += (double)weights[r * cols + c] * input[c];
        out[r] = (float)(sum * scales[r]);
    }
    return 0;
}

int cl_quantized_model_predict(const CLQuantModel *model, const float *input,
                               float *hidden, float *probabilities) {
    if (!model || !input || !hidden || !probabilities || !model->w1 || !model->w2 ||
        !model->s1 || !model->s2 || !model->b1 || !model->b2) return -1;
    float *logits = (float *)malloc(model->output_dim * sizeof(float));
    if (!logits) return -1;
    int result = -1;
    if (quant_matvec(model->w1, model->s1, input, hidden,
                     model->hidden_dim, model->input_dim)) goto done;
    for (size_t i = 0; i < model->hidden_dim; ++i) hidden[i] += model->b1[i];
    cl_relu(hidden, model->hidden_dim);
    if (quant_matvec(model->w2, model->s2, hidden, logits,
                     model->output_dim, model->hidden_dim)) goto done;
    for (size_t i = 0; i < model->output_dim; ++i) logits[i] += model->b2[i];
    result = cl_softmax(logits, probabilities, model->output_dim);
done:
    free(logits);
    return result;
}

size_t cl_model_storage_bytes(const CLModel *model) {
    if (!model) return 0;
    return sizeof(float) * (model->input_dim * model->hidden_dim +
           model->hidden_dim * model->output_dim + model->hidden_dim + model->output_dim);
}
size_t cl_quantized_storage_bytes(const CLQuantModel *model) {
    if (!model) return 0;
    return model->input_dim * model->hidden_dim + model->hidden_dim * model->output_dim +
           sizeof(float) * (2 * model->hidden_dim + 2 * model->output_dim);
}
