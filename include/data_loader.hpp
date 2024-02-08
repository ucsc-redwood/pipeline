#pragma once

#include <fstream>
#include <iostream>

template <typename T>
[[nodiscard]] T* loadFromBinaryFile(const std::string& filename, size_t& size) {
  std::ifstream inputFile(filename, std::ios::binary);

  if (!inputFile.is_open()) {
    std::cerr << "Failed to open file: " << filename << std::endl;
    return nullptr;
  }

  inputFile.seekg(0, std::ios::end);
  size = inputFile.tellg();
  inputFile.seekg(0, std::ios::beg);

  size_t numElements = size / sizeof(T);
  T* data = new T[numElements];

  inputFile.read(reinterpret_cast<char*>(data), size);
  inputFile.close();
  return data;
}
