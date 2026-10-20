#ifndef CORTEXLITE_TENSOR_H
#define CORTEXLITE_TENSOR_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Row-major matrix-vector product: output = matrix[rows x cols] * vector[cols]. */
int cl_matvec(const float *matrix, const float *vector, float *output,
              size_t rows, size_t cols);
void cl_relu(float *values, size_t size);
int cl_softmax(const float *logits, float *probabilities, size_t size);
/* Returns cosine similarity in [-1, 1]; zero vectors return 0. */
float cl_cosine_similarity(const float *a, const float *b, size_t size);

#ifdef __cplusplus
}
#endif
#endif
