#include "cortexlite/tensor.h"
#include "cortexlite/inference.h"
#include "cortexlite/vector_index.hpp"
#ifdef NDEBUG
#undef NDEBUG  // Keep unit assertions active in Release builds.
#endif
#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <limits>

static bool nearly(float a, float b, float eps = 1e-4f) { return std::fabs(a - b) < eps; }
int main(int argc, char **argv) {
    float matrix[] = {1, 2, 3, 4, 5, 6}, vec[] = {1, 2, 3}, result[2];
    assert(cl_matvec(matrix, vec, result, 2, 3) == 0);
    assert(nearly(result[0], 14) && nearly(result[1], 32));
    float relu[] = {-2, 0, 3}; cl_relu(relu, 3);
    assert(nearly(relu[0], 0) && nearly(relu[1], 0) && nearly(relu[2], 3));
    float logits[] = {1000, 1001}; float probs[2];
    assert(cl_softmax(logits, probs, 2) == 0);
    assert(nearly(probs[0] + probs[1], 1));
    float aa[] = {1, 0}, bb[] = {0, 1}, cc[] = {2, 0};
    assert(nearly(cl_cosine_similarity(aa, bb, 2), 0));
    assert(nearly(cl_cosine_similarity(aa, cc, 2), 1));
    // Compare accelerated matrix multiplication against FP64-accumulating baseline,
    // including a non-multiple-of-four tail and different row magnitudes.
    std::vector<float> big_matrix(19 * 31), big_input(31), expected(19), accelerated(19);
    for (std::size_t i = 0; i < big_matrix.size(); ++i)
        big_matrix[i] = static_cast<float>(std::sin(i * 0.37) * 3.0);
    for (std::size_t i = 0; i < big_input.size(); ++i)
        big_input[i] = static_cast<float>(std::cos(i * 0.21) * 2.0);
    assert(cl_matvec(big_matrix.data(), big_input.data(), expected.data(), 19, 31) == 0);
    if (cl_simd_available()) {
        assert(cl_matvec_kernel(big_matrix.data(), big_input.data(), accelerated.data(),
                                19, 31, CL_KERNEL_SIMD) == 0);
        for (std::size_t i = 0; i < expected.size(); ++i)
            assert(std::fabs(expected[i] - accelerated[i]) < 3e-5f);
    } else {
        assert(cl_matvec_kernel(big_matrix.data(), big_input.data(), accelerated.data(),
                                19, 31, CL_KERNEL_SIMD) == -1);
    }
    assert(cl_matvec_kernel(nullptr, big_input.data(), accelerated.data(), 19, 31,
                            CL_KERNEL_SCALAR) == -1);
    cortexlite::IdMap table;
    for (std::size_t i = 0; i < 1000; ++i) assert(table.insert("sample" + std::to_string(i), i));
    std::size_t location = 0;
    for (std::size_t i = 0; i < 1000; ++i) {
        assert(table.find("sample" + std::to_string(i), location) && location == i);
    }
    assert(!table.insert("sample0", 1234));
    assert(!table.find("missing", location));
    cortexlite::VectorIndex index(2, 2);
    index.add("A", 0, {1, 0});
    index.add("B", 1, {0, 1});
    index.add("C", 2, {-1, 0});
    assert(index.contains("B") && !index.contains("Z") && index.size() == 3);
    auto hits = index.search_exact({1, 0}, 2);
    assert(hits.size() == 2 && hits.front().id == "A");
    auto graph = index.search_graph({1, 0}, 2, 10);
    assert(!graph.empty() && graph.front().id == "A");
    try { index.add("A", 0, {0, 0}); assert(false); }
    catch (const std::invalid_argument &) {}
    if (argc > 1) {
        CLModel model{};
        assert(cl_model_load(argv[1], &model) == 0);
        assert(model.input_dim == 2 && model.hidden_dim == 8 && model.output_dim == 3);
        std::vector<float> hidden(model.hidden_dim), distribution(model.output_dim);
        float xy[] = {1.8f, -2.0f};
        assert(cl_model_predict(&model, xy, hidden.data(), distribution.data()) == 0);
        assert(nearly(distribution[0] + distribution[1] + distribution[2], 1.0f));
        assert(distribution[1] > distribution[0] && distribution[1] > distribution[2]);
        CLQuantModel quant{};
        assert(cl_model_quantize(&model, &quant) == 0);
        assert(cl_quantized_storage_bytes(&quant) < cl_model_storage_bytes(&model));
        std::vector<float> qhidden(model.hidden_dim), qprob(model.output_dim);
        for (int i = 0; i < 25; ++i) {
            float coordinates[] = {static_cast<float>(i - 12) / 5.0f,
                                   static_cast<float>(i % 9 - 4) / 3.0f};
            assert(cl_model_predict(&model, coordinates, hidden.data(), distribution.data()) == 0);
            assert(cl_quantized_model_predict(&quant, coordinates, qhidden.data(), qprob.data()) == 0);
            for (std::size_t j = 0; j < model.output_dim; ++j)
                assert(std::fabs(distribution[j] - qprob[j]) < 0.03f);
            if (cl_simd_available()) {
                std::vector<float> shidden(model.hidden_dim), sprob(model.output_dim);
                assert(cl_model_predict_kernel(&model, coordinates, shidden.data(), sprob.data(),
                                               CL_KERNEL_SIMD) == 0);
                for (std::size_t j = 0; j < model.output_dim; ++j)
                    assert(std::fabs(distribution[j] - sprob[j]) < 0.0001f);
            }
        }
        // Quantized buffers survive releasing the original floating-point weights.
        cl_model_free(&model);
        assert(cl_quantized_model_predict(&quant, xy, qhidden.data(), qprob.data()) == 0);
        cl_quantized_model_free(&quant);
        assert(cl_quantized_storage_bytes(&quant) == 0);
    }
    // Regression: dividing the smallest finite weights by 127 used to
    // underflow to a zero quantization scale and cause division by zero.
    float tiny_w1[] = {std::numeric_limits<float>::denorm_min()};
    float tiny_b1[] = {0.0f};
    float tiny_w2[] = {1.0f, -1.0f};
    float tiny_b2[] = {0.0f, 0.0f};
    CLModel tiny_model{};
    tiny_model.input_dim = 1;
    tiny_model.hidden_dim = 1;
    tiny_model.output_dim = 2;
    tiny_model.w1 = tiny_w1;
    tiny_model.b1 = tiny_b1;
    tiny_model.w2 = tiny_w2;
    tiny_model.b2 = tiny_b2;
    CLQuantModel tiny_quant{};
    assert(cl_model_quantize(&tiny_model, &tiny_quant) == 0);
    assert(std::isfinite(tiny_quant.s1[0]) && tiny_quant.s1[0] > 0.0f);
    assert(tiny_quant.w1[0] > 0);
    cl_quantized_model_free(&tiny_quant);

    std::cout << "all cortexlite unit tests passed\n";
}
