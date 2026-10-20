#ifndef CORTEXLITE_INFERENCE_H
#define CORTEXLITE_INFERENCE_H

#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

/* Portable two-layer MLP: ReLU(W1*x+b1), softmax(W2*hidden+b2). */
typedef struct {
    size_t input_dim;
    size_t hidden_dim;
    size_t output_dim;
    float *w1;
    float *b1;
    float *w2;
    float *b2;
} CLModel;

/* Load from CLMODEL1 text format; call cl_model_free after successful load. */
int cl_model_load(const char *path, CLModel *model);
void cl_model_free(CLModel *model);
/* Caller allocates hidden[hidden_dim] and probabilities[output_dim]. */
int cl_model_predict(const CLModel *model, const float *input,
                     float *hidden, float *probabilities);

#ifdef __cplusplus
}
#endif
#endif
