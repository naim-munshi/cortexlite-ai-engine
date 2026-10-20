#pragma once
#include <cstddef>
#include <string>
#include <vector>

namespace cortexlite {
struct SearchHit {
    std::string id;
    int label;
    float similarity;
};

// Custom open-addressing map: FNV-1a hash, linear probing, dynamic rehashing.
class IdMap {
public:
    IdMap();
    bool insert(const std::string &key, std::size_t value);
    bool find(const std::string &key, std::size_t &value) const;
private:
    struct Slot { std::string key; std::size_t value = 0; bool occupied = false; };
    std::vector<Slot> slots_;
    std::size_t count_ = 0;
    static std::size_t hash_key(const std::string &key);
    void grow();
};

class VectorIndex {
public:
    explicit VectorIndex(std::size_t dimensions, std::size_t max_neighbors = 4);
    void add(std::string id, int label, std::vector<float> embedding);
    bool contains(const std::string &id) const;
    std::size_t size() const;
    std::vector<SearchHit> search_exact(const std::vector<float> &query,
                                        std::size_t k) const;
    // Experimental best-first neighbor-graph search, not HNSW.
    std::vector<SearchHit> search_graph(const std::vector<float> &query,
                                        std::size_t k, std::size_t ef) const;
private:
    struct Entry { std::string id; int label; std::vector<float> vector; std::vector<std::size_t> neighbors; };
    std::size_t dimensions_;
    std::size_t max_neighbors_;
    std::vector<Entry> entries_;
    IdMap id_map_;
    void check_query(const std::vector<float> &query) const;
};
}
