#pragma once

// constexpr auto kRadix = 256;
constexpr auto kRadixPasses = 4;

// General macros
constexpr auto LANE_COUNT = 32;  // Threads in a warp
constexpr auto LANE_MASK = 31;   // Mask of the lane count
constexpr auto LANE_LOG = 5;     // log2(LANE_COUNT)

constexpr auto RADIX = 256;             // Number of digit bins
constexpr auto RADIX_MASK = RADIX - 1;  // Mask of digit bins, to extract digits
constexpr auto RADIX_LOG = 8;           // log2(RADIX)

constexpr auto SEC_RADIX = 8;
constexpr auto THIRD_RADIX = 16;
constexpr auto FOURTH_RADIX = 24;

// Offset for retrieving value from global histogram buffer
constexpr auto SEC_RADIX_START = 256;
constexpr auto THIRD_RADIX_START = 512;
constexpr auto FOURTH_RADIX_START = 768;

// Warps/Threads per threadblock in k_GlobalHistogram
const auto G_HIST_WARPS = 8;
const auto G_HIST_THREADS = 256;
const auto G_TBLOCK_LOG = 11;  // log2(gridDim.x) , gridDim.x is 2048

#define LANE threadIdx.x                             // Lane of a thread
#define WARP_INDEX threadIdx.y                       // Warp of a thread
#define THREAD_ID (LANE + (WARP_INDEX << LANE_LOG))  // Threadid

#define G_HIST_PART_SIZE \
  (size >> G_TBLOCK_LOG)  // Partition tile size in k_GlobalHistogram

// For the digit binning
// #define BIN_PART_SIZE 7680  // Partition tile size in k_DigitBinning
// #define BIN_HISTS_SIZE \
//   4096  // Total size of warp histograms in shared memory in k_DigitBinning
// #define BIN_SUB_PART_SIZE \
//   480  // Subpartition tile size of a single warp in k_DigitBinning
// #define BIN_THREADS 512         // Threads per threadblock in k_DigitBinning
// #define BIN_WARPS 16            // Warps per threadblock in k_DigitBinning
// #define BIN_KEYS_PER_THREAD 15  // Keys per thread in k_DigitBinning

constexpr auto BIN_PART_SIZE = 7680;  // Partition tile size in k_DigitBinning
constexpr auto BIN_HISTS_SIZE =
    4096;  // Total size of warp histograms in shared memory in k_DigitBinning
constexpr auto BIN_SUB_PART_SIZE =
    480;  // Subpartition tile size of a single warp in k_DigitBinning

constexpr auto BIN_THREADS = 512;  // Threads per threadblock in k_DigitBinning
constexpr auto BIN_WARPS = 16;     // Warps per threadblock in k_DigitBinning
constexpr auto BIN_KEYS_PER_THREAD = 15;  // Keys per thread in k_DigitBinning
