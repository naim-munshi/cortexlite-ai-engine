#include "cortexlite/vector_index.hpp"
#include "cortexlite/tensor.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <queue>
#include <stdexcept>
#include <utility>

namespace cortexlite {
namespace {
struct Neighbor { std::size_t index; float score; };

/* Custom fixed-capacity min-heap retains the k highest-scoring candidates. */
class TopKHeap {
public:
    explicit TopKHeap(std::size_t k) : capacity_(k) {}
    void offer(std::size_t idx, float score) {
        if (!capacity_) return;
        Neighbor item{idx, score};
        if (heap_.size() < capacity_) {
            heap_.push_back(item);
            sift_up(heap_.size() - 1);
        } else if (worse(heap_[0], item)) {
            heap_[0] = item;
            sift_down(0);
        }
    }
    std::vector<Neighbor> sorted() const {
        auto v = heap_;
        std::sort(v.begin(), v.end(), [](const Neighbor &a, const Neighbor &b) {
            return a.score == b.score ? a.index < b.index : a.score > b.score;
        });
        return v;
    }
private:
    // A higher-index candidate loses a score tie to an earlier insertion.
    // The heap root must therefore be the worst of the retained candidates.
    static bool worse(const Neighbor &a, const Neighbor &b) {
        return a.score < b.score || (a.score == b.score && a.index > b.index);
    }
    std::size_t capacity_;
    std::vector<Neighbor> heap_;
    void sift_up(std::size_t i) {
        while (i) {
            std::size_t parent = (i - 1) / 2;
            if (!worse(heap_[i], heap_[parent])) break;
            std::swap(heap_[parent], heap_[i]);
            i = parent;
        }
    }
    void sift_down(std::size_t i) {
        while (true) {
            std::size_t smallest = i, left = i * 2 + 1, right = left + 1;
            if (left < heap_.size() && worse(heap_[left], heap_[smallest])) smallest = left;
            if (right < heap_.size() && worse(heap_[right], heap_[smallest])) smallest = right;
            if (smallest == i) break;
            std::swap(heap_[smallest], heap_[i]);
            i = smallest;
        }
    }
};
}

IdMap::IdMap() : slots_(16) {}
std::size_t IdMap::hash_key(const std::string &key) {
    std::uint64_t hash = UINT64_C(14695981039346656037);
    for (unsigned char c : key) { hash ^= c; hash *= UINT64_C(1099511628211); }
    return static_cast<std::size_t>(hash);
}
void IdMap::grow() {
    auto previous = std::move(slots_);
    slots_ = std::vector<Slot>(previous.size() * 2);
    count_ = 0;
    for (const auto &slot : previous) if (slot.occupied) insert(slot.key, slot.value);
}
bool IdMap::insert(const std::string &key, std::size_t value) {
    if ((count_ + 1) * 10 >= slots_.size() * 7) grow();
    std::size_t i = hash_key(key) % slots_.size();
    while (slots_[i].occupied) {
        if (slots_[i].key == key) return false;
        i = (i + 1) % slots_.size();
    }
    slots_[i] = Slot{key, value, true};
    ++count_;
    return true;
}
bool IdMap::find(const std::string &key, std::size_t &value) const {
    std::size_t i = hash_key(key) % slots_.size();
    while (slots_[i].occupied) {
        if (slots_[i].key == key) { value = slots_[i].value; return true; }
        i = (i + 1) % slots_.size();
    }
    return false;
}

VectorIndex::VectorIndex(std::size_t dimensions, std::size_t max_neighbors)
    : dimensions_(dimensions), max_neighbors_(max_neighbors) {
    if (!dimensions || !max_neighbors) throw std::invalid_argument("dimensions and max_neighbors must be positive");
}
void VectorIndex::check_query(const std::vector<float> &query) const {
    if (query.size() != dimensions_) throw std::invalid_argument("embedding dimension mismatch");
    for (float f : query) if (!std::isfinite(f)) throw std::invalid_argument("non-finite embedding value");
}
void VectorIndex::add(std::string id, int label, std::vector<float> embedding) {
    check_query(embedding);
    std::size_t unused;
    if (id.empty() || id_map_.find(id, unused)) throw std::invalid_argument("empty or duplicate vector ID");
    const std::size_t next = entries_.size();
    TopKHeap neighbors(max_neighbors_);
    for (std::size_t i = 0; i < next; ++i) {
        const auto &v = entries_[i].vector;
        float score = cl_cosine_similarity(embedding.data(), v.data(), dimensions_);
        neighbors.offer(i, score);
    }
    Entry entry{std::move(id), label, std::move(embedding), {}};
    for (const Neighbor &hit : neighbors.sorted()) entry.neighbors.push_back(hit.index);
    // Attach in both directions: a connected educational approximate-search graph.
    for (auto idx : entry.neighbors) entries_[idx].neighbors.push_back(next);
    id_map_.insert(entry.id, next);
    entries_.push_back(std::move(entry));
}
bool VectorIndex::contains(const std::string &id) const {
    std::size_t unused; return id_map_.find(id, unused);
}
std::size_t VectorIndex::size() const { return entries_.size(); }
std::vector<SearchHit> VectorIndex::search_exact(const std::vector<float> &query,
                                                  std::size_t k) const {
    check_query(query);
    TopKHeap top(k);
    for (std::size_t i = 0; i < entries_.size(); ++i) {
        float score = cl_cosine_similarity(query.data(), entries_[i].vector.data(), dimensions_);
        top.offer(i, score);
    }
    std::vector<SearchHit> hits;
    for (const Neighbor &n : top.sorted())
        hits.push_back({entries_[n.index].id, entries_[n.index].label, n.score});
    return hits;
}
std::vector<SearchHit> VectorIndex::search_graph(const std::vector<float> &query,
                                                  std::size_t k, std::size_t ef) const {
    check_query(query);
    if (entries_.empty() || !k || !ef) return {};
    std::vector<unsigned char> seen(entries_.size(), 0);
    std::priority_queue<std::pair<float, std::size_t>> frontier;
    frontier.push({cl_cosine_similarity(query.data(), entries_[0].vector.data(), dimensions_), 0});
    seen[0] = 1;
    TopKHeap top(k);
    std::size_t expanded = 0;
    while (!frontier.empty() && expanded < ef) {
        auto current = frontier.top(); frontier.pop();
        top.offer(current.second, current.first);
        ++expanded;
        for (std::size_t idx : entries_[current.second].neighbors) {
            if (seen[idx]) continue;
            seen[idx] = 1;
            float score = cl_cosine_similarity(query.data(), entries_[idx].vector.data(), dimensions_);
            frontier.push({score, idx});
        }
    }
    std::vector<SearchHit> hits;
    for (const Neighbor &n : top.sorted())
        hits.push_back({entries_[n.index].id, entries_[n.index].label, n.score});
    return hits;
}
}
