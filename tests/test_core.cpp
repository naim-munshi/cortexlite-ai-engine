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
        cl_model_free(&model);
    }
    std::cout << "all cortexlite unit tests passed\n";
}
