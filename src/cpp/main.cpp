#include "cortexlite/inference.h"
#include "cortexlite/vector_index.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using cortexlite::VectorIndex;

namespace {
struct Row { std::string id; int label; float x; float y; };
struct Config {
    std::string command = "demo";
    std::string model = "data/demo_model.clm";
    std::string samples = "data/samples.csv";
    std::string eval = "data/eval.csv";
    std::string search = "exact";
    float x = 1.8f;
    float y = -2.0f;
    std::size_t k = 3;
    std::size_t ef = 32;
    std::size_t iterations = 1000;
};

void usage() {
    std::cout << "CortexLite — C inference + C++ DSA vector search\n"
              << "Usage: cortexlite [demo|evaluate|benchmark] [options]\n"
              << "Options: --model PATH --samples PATH --eval PATH --search exact|graph\n"
              << "         --x FLOAT --y FLOAT --k INT --ef INT --iterations INT\n";
}

Config parse(int argc, char **argv) {
    Config config;
    if (argc > 1 && argv[1][0] != '-') config.command = argv[1];
    int start = (argc > 1 && argv[1][0] != '-') ? 2 : 1;
    for (int i = start; i < argc; i += 2) {
        if (i + 1 >= argc) throw std::invalid_argument("option requires a value: " + std::string(argv[i]));
        std::string flag(argv[i]); std::string value(argv[i + 1]);
        if (flag == "--model") config.model = value;
        else if (flag == "--samples") config.samples = value;
        else if (flag == "--eval") config.eval = value;
        else if (flag == "--search") config.search = value;
        else if (flag == "--x") config.x = std::stof(value);
        else if (flag == "--y") config.y = std::stof(value);
        else if (flag == "--k") config.k = std::stoul(value);
        else if (flag == "--ef") config.ef = std::stoul(value);
        else if (flag == "--iterations") config.iterations = std::stoul(value);
        else throw std::invalid_argument("unknown option: " + flag);
    }
    if (config.command != "demo" && config.command != "evaluate" && config.command != "benchmark")
        throw std::invalid_argument("unknown command: " + config.command);
    if (config.search != "exact" && config.search != "graph") throw std::invalid_argument("unknown search mode");
    if (!config.k || !config.ef || !config.iterations || !std::isfinite(config.x) || !std::isfinite(config.y))
        throw std::invalid_argument("invalid numeric argument");
    return config;
}

std::vector<Row> load_rows(const std::string &path) {
    std::ifstream stream(path);
    if (!stream) throw std::runtime_error("cannot open CSV: " + path);
    std::vector<Row> rows;
    std::string line;
    if (!std::getline(stream, line) || line != "id,label,x,y")
        throw std::runtime_error("expected CSV header id,label,x,y: " + path);
    while (std::getline(stream, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string id, label, xs, ys, extra;
        if (!std::getline(ss, id, ',') || !std::getline(ss, label, ',') ||
            !std::getline(ss, xs, ',') || !std::getline(ss, ys, ',') ||
            std::getline(ss, extra, ',') || id.empty())
            throw std::runtime_error("bad CSV record: " + line);
        Row row{id, std::stoi(label), std::stof(xs), std::stof(ys)};
        if (!std::isfinite(row.x) || !std::isfinite(row.y) || row.label < 0)
            throw std::runtime_error("non-finite or invalid record: " + line);
        rows.push_back(row);
    }
    if (rows.empty()) throw std::runtime_error("empty CSV: " + path);
    return rows;
}

struct ModelHolder {
    CLModel value{};
    explicit ModelHolder(const std::string &path) {
        if (cl_model_load(path.c_str(), &value)) throw std::runtime_error("cannot load model: " + path);
        if (value.input_dim != 2) throw std::runtime_error("CSV demo requires a two-input model");
    }
    ~ModelHolder() { cl_model_free(&value); }
    ModelHolder(const ModelHolder &) = delete;
    ModelHolder &operator=(const ModelHolder &) = delete;
};

struct Prediction { std::vector<float> embedding; std::vector<float> probabilities; std::size_t label; };
Prediction predict(const CLModel &model, float x, float y) {
    Prediction p{{}, {}, 0};
    p.embedding.resize(model.hidden_dim);
    p.probabilities.resize(model.output_dim);
    const float input[] = {x, y};
    if (cl_model_predict(&model, input, p.embedding.data(), p.probabilities.data()))
        throw std::runtime_error("inference failed");
    p.label = static_cast<std::size_t>(std::distance(p.probabilities.begin(),
        std::max_element(p.probabilities.begin(), p.probabilities.end())));
    return p;
}

std::vector<cortexlite::SearchHit> search(const VectorIndex &index, const Config &config,
                                          const std::vector<float> &embedding) {
    if (config.search == "graph") return index.search_graph(embedding, config.k, config.ef);
    return index.search_exact(embedding, config.k);
}

void execute(const Config &config) {
    ModelHolder model(config.model);
    VectorIndex index(model.value.hidden_dim);
    auto references = load_rows(config.samples);
    for (const Row &row : references) {
        if (static_cast<std::size_t>(row.label) >= model.value.output_dim)
            throw std::runtime_error("sample label exceeds model classes");
        auto result = predict(model.value, row.x, row.y);
        index.add(row.id, row.label, std::move(result.embedding));
    }
    std::cout << std::fixed << std::setprecision(4);
    if (config.command == "demo") {
        auto prediction = predict(model.value, config.x, config.y);
        std::cout << "CortexLite demo\ninput: [" << config.x << ", " << config.y << "]\n"
                  << "predicted_class: " << prediction.label << "\nconfidence: "
                  << prediction.probabilities[prediction.label] << "\n"
                  << "index_size: " << index.size() << "\nsearch: " << config.search << "\n";
        for (const auto &hit : search(index, config, prediction.embedding))
            std::cout << "neighbor: " << hit.id << " class=" << hit.label
                      << " cosine=" << hit.similarity << "\n";
        return;
    }
    auto evaluations = load_rows(config.eval);
    std::size_t correct = 0, hits = 0;
    for (const Row &row : evaluations) {
        if (static_cast<std::size_t>(row.label) >= model.value.output_dim)
            throw std::runtime_error("evaluation label exceeds model classes");
        auto prediction = predict(model.value, row.x, row.y);
        if (prediction.label == static_cast<std::size_t>(row.label)) ++correct;
        const auto neighbors = search(index, config, prediction.embedding);
        if (!neighbors.empty() && neighbors.front().label == row.label) ++hits;
    }
    double total = static_cast<double>(evaluations.size());
    std::cout << "evaluation_samples: " << evaluations.size() << "\n"
              << "classification_accuracy: " << correct / total << "\n"
              << "retrieval_top1_label_accuracy: " << hits / total << "\n";
    if (config.command == "evaluate") return;

    std::vector<double> times_us;
    times_us.reserve(config.iterations);
    volatile std::size_t checksum = 0;
    for (std::size_t i = 0; i < config.iterations; ++i) {
        const Row &row = evaluations[i % evaluations.size()];
        const auto begin = std::chrono::steady_clock::now();
        auto prediction = predict(model.value, row.x, row.y);
        auto found = search(index, config, prediction.embedding);
        const auto end = std::chrono::steady_clock::now();
        checksum += prediction.label + found.size();
        times_us.push_back(std::chrono::duration<double, std::micro>(end - begin).count());
    }
    std::sort(times_us.begin(), times_us.end());
    const double elapsed_us = std::accumulate(times_us.begin(), times_us.end(), 0.0);
    const auto percentile = [&](double p) {
        return times_us[static_cast<std::size_t>((times_us.size() - 1) * p)];
    };
    std::cout << "benchmark_iterations: " << config.iterations << "\n"
              << "end_to_end_p50_us: " << percentile(0.50) << "\n"
              << "end_to_end_p95_us: " << percentile(0.95) << "\n"
              << "summed_operation_throughput_per_sec: " << (config.iterations * 1e6 / elapsed_us) << "\n"
              << "checksum: " << checksum << "\n";
}
}

int main(int argc, char **argv) {
    try {
        if (argc > 1 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
            usage(); return 0;
        }
        execute(parse(argc, argv));
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "error: " << e.what() << "\n";
        usage();
        return 1;
    }
}
