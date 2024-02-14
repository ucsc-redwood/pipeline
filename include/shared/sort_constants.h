#pragma once

constexpr auto kRadixPasses = 4;

// ------------------------------
// General macros
// ------------------------------

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

// ------------------------------
// For the global histogram
// ------------------------------

// Warps/Threads per threadblock in k_GlobalHistogram
const auto G_HIST_WARPS = 8;
const auto G_HIST_THREADS = 256;
// log2(gridDim.x) , gridDim.x is 2048
const auto G_TBLOCK_LOG = 11;

#define LANE threadIdx.x                             // Lane of a thread
#define WARP_INDEX threadIdx.y                       // Warp of a thread
#define THREAD_ID (LANE + (WARP_INDEX << LANE_LOG))  // Threadid

// Partition tile size in k_GlobalHistogram
#define G_HIST_PART_SIZE (size >> G_TBLOCK_LOG)

// ------------------------------
// For the digit binning
// ------------------------------

// Partition tile size in k_DigitBinning
constexpr auto BIN_PART_SIZE = 7680;
// Total size of warp histograms in shared memory in k_DigitBinning
constexpr auto BIN_HISTS_SIZE = 4096;
// Subpartition tile size of a single warp in k_DigitBinning
constexpr auto BIN_SUB_PART_SIZE = 480;

// Threads per threadblock in k_DigitBinning
constexpr auto BIN_THREADS = 512;
// Warps per threadblock in k_DigitBinning
constexpr auto BIN_WARPS = 16;
// Keys per thread in k_DigitBinning
constexpr auto BIN_KEYS_PER_THREAD = 15;

// for the chained scan with decoupled lookback
// Flag value inidicating neither inclusive sum, nor reduction of a
// partition tile is ready
constexpr auto FLAG_NOT_READY = 0;
// Flag value indicating reduction of a partition tile is ready
constexpr auto FLAG_REDUCTION = 1;
// Flag value indicating inclusive sum of a partition tile is ready
constexpr auto FLAG_INCLUSIVE = 2;
// Mask used to retrieve flag values
constexpr auto FLAG_MASK = 3;

// Starting offset of a subpartition tile
#define BIN_SUB_PART_START (WARP_INDEX * BIN_SUB_PART_SIZE)
// Starting offset of a partition tile
#define BIN_PART_START (partitionIndex * BIN_PART_SIZE)
