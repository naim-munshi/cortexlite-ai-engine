#ifndef CORTEXLITE_INFERENCE_H
#define CORTEXLITE_INFERENCE_H
#include <stddef.h>
#include <stdint.h>
#include "cortexlite/tensor.h"
#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    size_t input_dim, hidden_dim, output_dim;
    float *w1, *b1, *w2, *b2;
} CLModel;
/* Symmetric per-output-row INT8 weights; biases remain FP32. Owns its allocations. */
typedef struct {
    size_t input_dim, hidden_dim, output_dim;
    int8_t *w1, *w2;
    float *s1, *s2, *b1, *b2;
} CLQuantModel;
int cl_model_load(const char *path, CLModel *model);
void cl_model_free(CLModel *model);
int cl_model_predict(const CLModel *model, const float *input,
                     float *hidden, float *probabilities);
int cl_model_predict_kernel(const CLModel *model, const float *input,
                            float *hidden, float *probabilities, CLKernel kernel);
int cl_model_quantize(const CLModel *source, CLQuantModel *quant);
int cl_quantized_model_predict(const CLQuantModel *model, const float *input,
                               float *hidden, float *probabilities);
void cl_quantized_model_free(CLQuantModel *model);
size_t cl_model_storage_bytes(const CLModel *model);
size_t cl_quantized_storage_bytes(const CLQuantModel *model);
#ifdef __cplusplus
}
#endif
#endif
