#include "cortexlite/vector_index.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace {
struct Config { std::size_t n=1000, d=32, queries=100, k=10, ef=64; unsigned seed=42; };
Config parse(int argc, char **argv) {
    Config c;
    for (int i=1; i<argc; i+=2) {
        if (i+1 >= argc) throw std::invalid_argument("missing argument value");
        const std::string flag=argv[i];
        const auto value=std::stoul(argv[i+1]);
        if (flag=="--vectors") c.n=value;
        else if (flag=="--dimensions") c.d=value;
        else if (flag=="--queries") c.queries=value;
        else if (flag=="--k") c.k=value;
        else if (flag=="--ef") c.ef=value;
        else if (flag=="--seed") c.seed=static_cast<unsigned>(value);
        else throw std::invalid_argument("unknown option: "+flag);
    }
    if (!c.n || !c.d || !c.queries || !c.k || !c.ef || c.n>20000 || c.d>512 ||
        c.queries>10000 || c.k>c.n || c.ef>c.n)
        throw std::invalid_argument("out of range argument");
    return c;
}
using Clock=std::chrono::steady_clock;
double us(Clock::time_point a,Clock::time_point b) {
    return std::chrono::duration<double,std::micro>(b-a).count();
}
double percentile(std::vector<double> a,double p) {
    std::sort(a.begin(),a.end());
    return a[static_cast<std::size_t>((a.size()-1)*p)];
}
void run(int argc,char **argv) {
    const Config c=parse(argc,argv);
    std::mt19937 gen(c.seed);
    std::normal_distribution<float> normal(0.0f,1.0f);
    std::vector<std::vector<float>> vectors(c.n,std::vector<float>(c.d));
    // Seeded Gaussian vectors, with deterministic perturbed-query samples.
    for (auto &vec: vectors)
        for (float &v: vec) v=normal(gen);
    std::vector<std::vector<float>> query_set;
    query_set.reserve(c.queries);
    for (std::size_t i=0;i<c.queries;++i) {
        auto q=vectors[(i*7919u+17u)%c.n];
        for (float &v:q) v+=0.12f*normal(gen);
        query_set.push_back(std::move(q));
    }
    cortexlite::VectorIndex index(c.d);
    const auto build_begin=Clock::now();
    for (std::size_t i=0;i<c.n;++i)
        index.add(std::to_string(i),static_cast<int>(i%3),std::move(vectors[i]));
    const auto build_end=Clock::now();
    std::vector<double> exact_us,graph_us;
    double recall_sum=0.0;
    std::size_t successful_queries=0;
    for (const auto &q:query_set) {
        auto start=Clock::now();
        const auto exact=index.search_exact(q,c.k);
        auto mid=Clock::now();
        const auto graph=index.search_graph(q,c.k,c.ef);
        auto end=Clock::now();
        exact_us.push_back(us(start,mid));
        graph_us.push_back(us(mid,end));
        std::unordered_set<std::string> truth;
        for (const auto &h:exact) truth.insert(h.id);
        std::size_t matches=0;
        for (const auto &h:graph) matches+=truth.count(h.id);
        recall_sum+=static_cast<double>(matches)/exact.size();
        successful_queries+=(matches==exact.size());
    }
    const double recall=recall_sum/c.queries;
    std::cout<<std::fixed<<std::setprecision(6)
             <<"seed: "<<c.seed<<"\n"
             <<"vectors: "<<c.n<<"\n"
             <<"dimensions: "<<c.d<<"\n"
             <<"queries: "<<c.queries<<"\n"
             <<"k: "<<c.k<<"\n"
             <<"ef: "<<c.ef<<"\n"
             <<"index_build_ms: "<<us(build_begin,build_end)/1000.0<<"\n"
             <<"exact_p50_us: "<<percentile(exact_us,0.5)<<"\n"
             <<"exact_p95_us: "<<percentile(exact_us,0.95)<<"\n"
             <<"graph_p50_us: "<<percentile(graph_us,0.5)<<"\n"
             <<"graph_p95_us: "<<percentile(graph_us,0.95)<<"\n"
             <<"graph_recall_at_k: "<<recall<<"\n"
             <<"exact_match_queries: "<<successful_queries<<"\n";
}
}
int main(int argc,char **argv) {
    try {run(argc,argv);return 0;}
    catch(const std::exception &e) {std::cerr<<"error: "<<e.what()<<"\n";return 1;}
}
