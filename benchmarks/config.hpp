#pragma once

// benchmark settings
constexpr auto kN = 10'000'000;
constexpr auto kMin = 0.0f;
constexpr auto kMax = 1024.0f;
constexpr auto kRange = kMax - kMin;

constexpr auto kRandomSeed = 114514;

// // Below are benchmark constants,
// // generated from using seed 114514 and n = 10'000'000
// // when loading

// constexpr auto kNumUnique = 9953841;
// constexpr auto kSortedMortonFile = "data/bm_sorted_mortons_u32_10m.bin";

// constexpr auto kNumBrtNodes = kNumUnique - 1;
// constexpr auto kBrtFile = "data/bm_brt_u32_10m.bin";

// // n unique
// constexpr auto kEdgeCountFile = "data/bm_edge_count_u32_10m.bin";

// // n unique + 1
// constexpr auto kPrefixSumFile = "data/bm_prefix_sum_u32_10m.bin";

// constexpr auto kNumOctNodes = 4669482;
// constexpr auto kOctFile = "data/bm_oct_u32_10m.bin";
