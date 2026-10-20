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
#include <thread>
#include <limits>
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
    std::string precision = "fp32";
    std::string kernel = "scalar";
    std::size_t threads = 1;
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
              << "         --x FLOAT --y FLOAT --k INT --ef INT --iterations INT\n"
              << "         --precision fp32|int8 --kernel scalar|simd --threads INT\n";
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
        else if (flag == "--precision") config.precision = value;
        else if (flag == "--kernel") config.kernel = value;
        else if (flag == "--threads") config.threads = std::stoul(value);
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
    if ((config.precision != "fp32" && config.precision != "int8") ||
        (config.kernel != "scalar" && config.kernel != "simd") ||
        (config.precision == "int8" && config.kernel == "simd") ||
        !config.threads || config.threads > 64 || !config.k || !config.ef ||
        !config.iterations || config.iterations > 10000000 ||
        !std::isfinite(config.x) || !std::isfinite(config.y))
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
    CLQuantModel quant{};
    std::size_t input_dim = 0, hidden_dim = 0, output_dim = 0;
    bool is_quantized = false;
    explicit ModelHolder(const std::string &path, const std::string &precision) {
        if (cl_model_load(path.c_str(), &value)) throw std::runtime_error("cannot load model: " + path);
        input_dim = value.input_dim;
        hidden_dim = value.hidden_dim;
        output_dim = value.output_dim;
        if (input_dim != 2) {
            cl_model_free(&value);
            throw std::runtime_error("CSV demo requires a two-input model");
        }
        if (precision == "int8") {
            if (cl_model_quantize(&value, &quant)) {
                cl_model_free(&value);
                throw std::runtime_error("weight quantization failed");
            }
            is_quantized = true;
            // Quantized execution does not retain an additional FP32 weight copy.
            cl_model_free(&value);
        }
    }
    ~ModelHolder() { cl_model_free(&value); cl_quantized_model_free(&quant); }
    ModelHolder(const ModelHolder &) = delete;
    ModelHolder &operator=(const ModelHolder &) = delete;
    std::size_t bytes() const {
        return is_quantized ? cl_quantized_storage_bytes(&quant) : cl_model_storage_bytes(&value);
    }
};

struct Prediction { std::vector<float> embedding; std::vector<float> probabilities; std::size_t label; };
Prediction predict(const ModelHolder &model, const Config &config, float x, float y) {
    Prediction p{{}, {}, 0};
    p.embedding.resize(model.hidden_dim);
    p.probabilities.resize(model.output_dim);
    const float input[] = {x, y};
    const int code = model.is_quantized
        ? cl_quantized_model_predict(&model.quant, input, p.embedding.data(), p.probabilities.data())
        : cl_model_predict_kernel(&model.value, input, p.embedding.data(), p.probabilities.data(),
                                  config.kernel == "simd" ? CL_KERNEL_SIMD : CL_KERNEL_SCALAR);
    if (code) throw std::runtime_error("inference failed");
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
    if (config.kernel == "simd" && !cl_simd_available())
        throw std::runtime_error("SIMD unavailable in this build; select scalar");
    ModelHolder model(config.model, config.precision);
    VectorIndex index(model.hidden_dim);
    auto references = load_rows(config.samples);
    for (const Row &row : references) {
        if (static_cast<std::size_t>(row.label) >= model.output_dim)
            throw std::runtime_error("sample label exceeds model classes");
        auto result = predict(model, config, row.x, row.y);
        index.add(row.id, row.label, std::move(result.embedding));
    }
    std::cout << std::fixed << std::setprecision(4);
    if (config.command == "demo") {
        auto prediction = predict(model, config, config.x, config.y);
        std::cout << "CortexLite demo\ninput: [" << config.x << ", " << config.y << "]\n"
                  << "predicted_class: " << prediction.label << "\nconfidence: "
                  << prediction.probabilities[prediction.label] << "\n"
                  << "precision: " << config.precision << "\nkernel: " << config.kernel
                  << "\nweight_storage_bytes: " << model.bytes() << "\nindex_size: " << index.size() << "\nsearch: " << config.search << "\n";
        for (const auto &hit : search(index, config, prediction.embedding))
            std::cout << "neighbor: " << hit.id << " class=" << hit.label
                      << " cosine=" << hit.similarity << "\n";
        return;
    }
    auto evaluations = load_rows(config.eval);
    std::size_t correct = 0, hits = 0;
    for (const Row &row : evaluations) {
        if (static_cast<std::size_t>(row.label) >= model.output_dim)
            throw std::runtime_error("evaluation label exceeds model classes");
        auto prediction = predict(model, config, row.x, row.y);
        if (prediction.label == static_cast<std::size_t>(row.label)) ++correct;
        const auto neighbors = search(index, config, prediction.embedding);
        if (!neighbors.empty() && neighbors.front().label == row.label) ++hits;
    }
    double total = static_cast<double>(evaluations.size());
    std::cout << "evaluation_samples: " << evaluations.size() << "\n"
              << "classification_accuracy: " << correct / total << "\n"
              << "retrieval_top1_label_accuracy: " << hits / total << "\n";
    if (config.command == "evaluate") return;

    // Independent, read-only C inference calls and read-only C++ search calls.
    // Wall-clock measurement includes scheduling and worker dispatch; per-operation
    // timing excludes index construction and CSV parsing.
    std::vector<double> times_us(config.iterations);
    std::vector<std::size_t> checksums(config.threads, 0);
    std::vector<std::thread> workers;
    workers.reserve(config.threads);
    const auto wall_begin = std::chrono::steady_clock::now();
    for (std::size_t t = 0; t < config.threads; ++t) {
        workers.emplace_back([&, t] {
            const std::size_t begin = config.iterations * t / config.threads;
            const std::size_t end = config.iterations * (t + 1) / config.threads;
            std::size_t local = 0;
            for (std::size_t i = begin; i < end; ++i) {
                const Row &row = evaluations[i % evaluations.size()];
                const auto tick = std::chrono::steady_clock::now();
                auto prediction = predict(model, config, row.x, row.y);
                auto found = search(index, config, prediction.embedding);
                const auto tock = std::chrono::steady_clock::now();
                local += prediction.label + found.size();
                times_us[i] = std::chrono::duration<double, std::micro>(tock - tick).count();
            }
            checksums[t] = local;
        });
    }
    for (auto &worker : workers) worker.join();
    const auto wall_end = std::chrono::steady_clock::now();
    const double wall_seconds = std::chrono::duration<double>(wall_end - wall_begin).count();
    const std::size_t checksum = std::accumulate(checksums.begin(), checksums.end(), std::size_t{0});
    std::sort(times_us.begin(), times_us.end());
    const auto percentile = [&](double p) {
        return times_us[static_cast<std::size_t>((times_us.size() - 1) * p)];
    };
    std::cout << "benchmark_iterations: " << config.iterations << "\n"
              << "threads: " << config.threads << "\n"
              << "precision: " << config.precision << "\n"
              << "kernel: " << config.kernel << "\n"
              << "weight_storage_bytes: " << model.bytes() << "\n"
              << "end_to_end_p50_us: " << percentile(0.50) << "\n"
              << "end_to_end_p95_us: " << percentile(0.95) << "\n"
              << "wall_throughput_ops_per_sec: " << (config.iterations / wall_seconds) << "\n"
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
