#include "cuda/kernels/octree.cuh"
#include "cuda/morton.cuh"

namespace gpu {

constexpr int morton_bits = 30;

// processing for index 'i'
__host__ __device__ __forceinline__ void ProcessOctNode(
    int i,
    gpu::OctNode* oct_nodes,
    const int* node_offsets,    // prefix sum
    const int* rt_node_counts,  // edge count
    const unsigned int* codes,
    const uint8_t* rt_prefixN,
    const int* rt_parents,
    float min_coord,
    float range,
    int N  // number of brt nodes
) {
  auto oct_idx = node_offsets[i];
  const auto n_new_nodes = rt_node_counts[i];

  const auto root_level = rt_prefixN[0] / 3;

  // for each nodes, make n_new_nodes
  for (auto j = 0; j < n_new_nodes - 1; ++j) {
    const auto level = rt_prefixN[i] / 3 - j;

    const auto node_prefix = codes[i] >> (morton_bits - (3 * level));
    const auto which_child = node_prefix & 0b111;
    const auto parent = oct_idx + 1;

    oct_nodes[parent].SetChild(which_child, oct_idx);

    gpu::morton32_to_xyz(&oct_nodes[oct_idx].corner,
                         node_prefix << (morton_bits - (3 * level)),
                         min_coord,
                         range);

    // each cell is half the size of the level above it
    oct_nodes[oct_idx].cell_size =
        range / static_cast<float>(1 << (level - root_level));
    oct_idx = parent;
  }

  if (n_new_nodes > 0) {
    auto rt_parent = rt_parents[i];

    auto counter = 0;
    while (rt_node_counts[rt_parent] == 0) {
      rt_parent = rt_parents[rt_parent];

      ++counter;
      if (counter > 30) {  // 64 / 3
        // tb_trace_w("counter > 22");
        break;
      }
    }

    const auto oct_parent = node_offsets[rt_parent];
    const auto top_level = rt_prefixN[i] / 3 - n_new_nodes + 1;
    const auto top_node_prefix = codes[i] >> (morton_bits - (3 * top_level));

    const auto which_child = top_node_prefix & 0b111;

    oct_nodes[oct_parent].SetChild(which_child, oct_idx);

    gpu::morton32_to_xyz(&oct_nodes[oct_idx].corner,
                         top_node_prefix << (morton_bits - (3 * top_level)),
                         min_coord,
                         range);

    oct_nodes[oct_idx].cell_size =
        range / static_cast<float>(1 << (top_level - root_level));
  }
}

__global__ void k_MakeOctNodes(gpu::OctNode* oct_nodes,
                               const int* node_offsets,    // prefix sum
                               const int* rt_node_counts,  // edge count
                               const unsigned int* codes,
                               const uint8_t* rt_prefixN,
                               const int* rt_parents,
                               float min_coord,
                               float range,
                               int N  // number of brt nodes
) {
  const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
  const auto stride = blockDim.x * gridDim.x;

  for (auto i = idx; i < N; i += stride) {
    ProcessOctNode(i,
                   oct_nodes,
                   node_offsets,
                   rt_node_counts,
                   codes,
                   rt_prefixN,
                   rt_parents,
                   min_coord,
                   range,
                   N);
  }
}

}  // namespace gpu