#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>
#include <type_traits>
#include <vector>
#include "runtime/init.h"

namespace {

using HostEngine = std::default_random_engine;
static_assert(std::is_same<HostEngine, std::minstd_rand0>::value,
              "This experiment requires the libstdc++ default CPU engine");

constexpr uint64_t host_modulus = HostEngine::modulus;

uint64_t host_power(uint64_t exponent) {
    uint64_t base = HostEngine::multiplier, result = 1;
    while (exponent) {
        if (exponent & 1) result = result * base % host_modulus;
        base = base * base % host_modulus;
        exponent >>= 1;
    }
    return result;
}

struct NormalBlock {
    uint64_t seed = 0;
    std::vector<float> values;

    void generate(size_t draws) {
        HostEngine engine(seed), tracker(seed);
        std::normal_distribution<float> normal(0.0f, 1.0f);
        values.clear();
        size_t used = 0;
        while (used < draws) {
            float first = normal(engine);
            float second = normal(engine);
            do {
                tracker();
                tracker();
                used += 2;
            } while (tracker != engine && used <= draws);
            if (used > draws) break;
            values.push_back(first);
            values.push_back(second);
        }
    }
};

struct JoinThreads {
    std::vector<std::thread> threads;
    ~JoinThreads() {
        for (auto& thread : threads) thread.join();
    }
};

void jt_parallel_normal(float* output, size_t count, HostEngine& global) {
    constexpr size_t draws = 262144;
    const char* configured = std::getenv("PARTLLM_CPU_NORMAL_THREADS");
    const size_t workers = configured
        ? static_cast<size_t>(std::max(1, std::min(12, std::atoi(configured)))) : 6;
    const uint64_t stride = host_power(draws);
    auto copy = global;
    // Invert the first raw draw to obtain the engine's current state.
    uint64_t seed = copy() * host_power(host_modulus - 2) % host_modulus;
    std::vector<NormalBlock> blocks(workers);
    for (auto& block : blocks) block.values.reserve(draws);

    while (count) {
        {
            JoinThreads work;
            work.threads.reserve(workers - 1);
            for (size_t i = 0; i < workers; ++i) {
                blocks[i].seed = seed;
                seed = seed * stride % host_modulus;
                if (i + 1 == workers) blocks[i].generate(draws);
                else work.threads.emplace_back(&NormalBlock::generate, &blocks[i], draws);
            }
        }
        for (auto& block : blocks) {
            size_t n = std::min(count, block.values.size());
            std::memcpy(output, block.values.data(), n * sizeof(float));
            output += n;
            count -= n;
            if (!count) {
                // Recover the precise engine state after the last consumed
                // pair, including when the caller requested an odd count.
                global.seed(block.seed);
                std::normal_distribution<float> normal(0.0f, 1.0f);
                for (size_t i = 0; i < n; ++i) normal(global);
                return;
            }
        }
    }
}

}  // namespace
