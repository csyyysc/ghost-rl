#include "kernel.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  printf("Usage: %s <input_image> <output_image> <binsize>\n", argv[0]);
  printf("--------------------------------\n");

  int choice = choose_filter();
  const char *filename = argc > 2 ? argv[2] : "output";

  switch (choice) {
  case 1:
    kernel_callback(argv, filename, histogram);
    break;
  case 2:
    kernel_callback(argv, filename, histogram_equalization);
    break;
  default:
    printf("Invalid selection\n");
    return 1;
  }

  return 0;
}
