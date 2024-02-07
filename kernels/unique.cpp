
int k_RemoveConsecutiveDuplicates(unsigned int *keys, int n) {
  if (n == 0) {
    return 0;
  }

  int j = 0;
  for (int i = 1; i < n; ++i) {
    if (keys[i] != keys[j]) {
      ++j;
      keys[j] = keys[i];
    }
  }

  return j + 1;
}