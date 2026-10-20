#ifndef CORTEXLITE_TENSOR_H
#define CORTEXLITE_TENSOR_H

#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef enum { CL_KERNEL_SCALAR = 0, CL_KERNEL_SIMD = 1 } CLKernel;
/* Scalar implementation accumulates in double, serving as the reference. */
int cl_matvec(const float *matrix, const float *vector, float *output,
              size_t rows, size_t cols);
/* SIMD uses SSE2 on x86 or NEON on ARM; falls back to scalar only when unavailable. */
int cl_matvec_kernel(const float *matrix, const float *vector, float *output,
                     size_t rows, size_t cols, CLKernel kernel);
int cl_simd_available(void);
void cl_relu(float *values, size_t size);
int cl_softmax(const float *logits, float *probabilities, size_t size);
float cl_cosine_similarity(const float *a, const float *b, size_t size);
#ifdef __cplusplus
}
#endif
#endif
