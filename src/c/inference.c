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

int cl_model_predict(const CLModel *model, const float *input,
                     float *hidden, float *probabilities) {
    if (!model || !input || !hidden || !probabilities || !model->w1 || !model->w2 ||
        !model->b1 || !model->b2) return -1;
    float *logits = (float *)malloc(model->output_dim * sizeof(float));
    if (!logits) return -1;
    if (cl_matvec(model->w1, input, hidden, model->hidden_dim, model->input_dim)) {
        free(logits);
        return -1;
    }
    for (size_t i = 0; i < model->hidden_dim; ++i) hidden[i] += model->b1[i];
    cl_relu(hidden, model->hidden_dim);
    if (cl_matvec(model->w2, hidden, logits, model->output_dim, model->hidden_dim)) {
        free(logits);
        return -1;
    }
    for (size_t i = 0; i < model->output_dim; ++i) logits[i] += model->b2[i];
    int result = cl_softmax(logits, probabilities, model->output_dim);
    free(logits);
    return result;
}
