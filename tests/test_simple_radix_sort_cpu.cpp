
#include <algorithm>
#include <iostream>

int main(const int argc, const char** argv) {
  int n = argc - 1;
  auto* data = new unsigned int[n];
  for (int i = 0; i < n; i++) {
    data[i] = atoi(argv[i + 1]);
  }

  std::sort(data, data + n);

  for (int i = 0; i < n; i++) std::cout << data[i] << " ";

  delete[] data;
  return 0;
}
