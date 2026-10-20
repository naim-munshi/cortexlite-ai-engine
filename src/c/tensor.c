#include "cortexlite/tensor.h"
#include <math.h>

int cl_matvec(const float *matrix, const float *vector, float *output,
              size_t rows, size_t cols) {
    if (!matrix || !vector || !output || !rows || !cols) return -1;
    for (size_t r = 0; r < rows; ++r) {
        double sum = 0.0;
        for (size_t c = 0; c < cols; ++c) {
            sum += (double)matrix[r * cols + c] * vector[c];
        }
        output[r] = (float)sum;
    }
    return 0;
}

void cl_relu(float *values, size_t size) {
    if (!values) return;
    for (size_t i = 0; i < size; ++i)
        if (values[i] < 0.0f) values[i] = 0.0f;
}

int cl_softmax(const float *logits, float *probabilities, size_t size) {
    if (!logits || !probabilities || !size) return -1;
    float max_value = logits[0];
    for (size_t i = 0; i < size; ++i) {
        if (!isfinite(logits[i])) return -1;
        if (logits[i] > max_value) max_value = logits[i];
    }
    double sum = 0.0;
    for (size_t i = 0; i < size; ++i) {
        probabilities[i] = expf(logits[i] - max_value);
        sum += probabilities[i];
    }
    if (!(sum > 0.0) || !isfinite(sum)) return -1;
    for (size_t i = 0; i < size; ++i) probabilities[i] = (float)(probabilities[i] / sum);
    return 0;
}

float cl_cosine_similarity(const float *a, const float *b, size_t size) {
    if (!a || !b || !size) return 0.0f;
    double dot = 0.0, aa = 0.0, bb = 0.0;
    for (size_t i = 0; i < size; ++i) {
        dot += (double)a[i] * b[i];
        aa += (double)a[i] * a[i];
        bb += (double)b[i] * b[i];
    }
    if (aa <= 0.0 || bb <= 0.0) return 0.0f;
    return (float)(dot / sqrt(aa * bb));
}
