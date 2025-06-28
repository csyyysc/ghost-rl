#include "kernel.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {

  printf("Usage: %s <input_image> <output_image> <threshold>\n", argv[0]);
  printf("--------------------------------\n");

  if (argc < 4) {
    printf("Insufficient arguments.\n");
    return 1;
  }

  int choice = choose_filter();
  const char *filename = argv[2];

  switch (choice) {
  case 1:
    kernel_filter_callback(argv, filename, binarization_fixed_threshold);
    break;
  case 2:
    kernel_filter_callback(argv, filename, binarization_error_diffusion);
    break;
  default:
    printf("Invalid selection\n");
    return 1;
  }

  return 0;
}