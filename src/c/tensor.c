#include "cortexlite/tensor.h"
#include <math.h>

#if defined(__SSE2__)
#include <emmintrin.h>
#define CL_HAS_SIMD 1
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define CL_HAS_SIMD 1
#else
#define CL_HAS_SIMD 0
#endif

int cl_simd_available(void) { return CL_HAS_SIMD; }

int cl_matvec_kernel(const float *matrix, const float *vector, float *output,
                     size_t rows, size_t cols, CLKernel kernel) {
    if (!matrix || !vector || !output || !rows || !cols ||
        (kernel != CL_KERNEL_SCALAR && kernel != CL_KERNEL_SIMD)) return -1;
    if (rows > (size_t)-1 / cols) return -1;
    if (kernel == CL_KERNEL_SIMD && !CL_HAS_SIMD) return -1;
    for (size_t r = 0; r < rows; ++r) {
        const float *row = matrix + r * cols;
        if (kernel == CL_KERNEL_SIMD) {
#if defined(__SSE2__)
            __m128 acc = _mm_setzero_ps();
            size_t c = 0;
            for (; c + 4 <= cols; c += 4)
                acc = _mm_add_ps(acc, _mm_mul_ps(_mm_loadu_ps(row + c), _mm_loadu_ps(vector + c)));
            float partial[4];
            _mm_storeu_ps(partial, acc);
            double sum = (double)partial[0] + partial[1] + partial[2] + partial[3];
            for (; c < cols; ++c) sum += (double)row[c] * vector[c];
            output[r] = (float)sum;
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
            float32x4_t acc = vdupq_n_f32(0.0f);
            size_t c = 0;
            for (; c + 4 <= cols; c += 4)
                acc = vaddq_f32(acc, vmulq_f32(vld1q_f32(row + c), vld1q_f32(vector + c)));
            float partial[4];
            vst1q_f32(partial, acc);
            double sum = (double)partial[0] + partial[1] + partial[2] + partial[3];
            for (; c < cols; ++c) sum += (double)row[c] * vector[c];
            output[r] = (float)sum;
#else
            return -1;
#endif
        } else {
            double sum = 0.0;
            for (size_t c = 0; c < cols; ++c)
                sum += (double)row[c] * vector[c];
            output[r] = (float)sum;
        }
    }
    return 0;
}

int cl_matvec(const float *matrix, const float *vector, float *output,
              size_t rows, size_t cols) {
    return cl_matvec_kernel(matrix, vector, output, rows, cols, CL_KERNEL_SCALAR);
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
