#pragma once

#if defined(__CUDACC__)
#define H_D_I __host__ __device__ __forceinline__
#else
#define H_D_I inline
#endif
