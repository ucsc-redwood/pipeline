#pragma once

struct AppParams {
  int n = 640 * 480;
  float min = 0.0f;
  float max = 1024.0f;
  float range = max - min;
  int seed = 114514;
  int num_threads = 4;
  int my_num_blocks = 64;
};
