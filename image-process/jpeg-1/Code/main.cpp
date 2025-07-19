#include "kernel.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  printf("Usage: %s <input_image> <output_image>\n", argv[0]);
  printf("--------------------------------\n");

  int choice = choose_filter();
  printf("choice: %d\n", choice);

  const char *filename = argv[2] ? argv[2] : "output";

  switch (choice) {
  case 1:
    kernel_callback(argv, filename, dct_idct);
    break;
  case 2:
    kernel_callback(argv, filename, dct_visualize);
    break;
  case 3:
    kernel_callback(argv, filename, dct_energy);
    break;
  case 4:
    kernel_callback(argv, filename, dct_random_analysis);
    break;
  default:
    printf("Invalid selection\n");
    return 1;
  }

  return 0;
}
