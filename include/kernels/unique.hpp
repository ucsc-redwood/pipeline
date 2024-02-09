#pragma once

int k_CountUnique(unsigned int *keys, int n);

namespace gpu {

int k_CountUnique(const unsigned int *keys, int n);

}
