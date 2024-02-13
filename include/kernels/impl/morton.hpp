#include "kernels/01_morton.hpp"

namespace cpu {

unsigned int single_point_to_code_v2(
    float x, float y, float z, float min_coord, float range);

void morton32_to_xyz(glm::vec4* ret,
                     unsigned int code,
                     float min_coord,
                     float range);

[[maybe_unused]] int compare_uint32_t(const void* a, const void* b);

}  // namespace cpu
