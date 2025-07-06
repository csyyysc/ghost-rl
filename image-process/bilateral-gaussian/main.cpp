#include "kernel.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  printf("Usage: %s <input_image> <output_image> <sigma_s> <sigma_r>\n",
         argv[0]);
  printf("--------------------------------\n");

  int choice = choose_filter();
  const char *filename = argc > 2 ? argv[2] : "output";

  switch (choice) {
  case 1:
    kernel_callback(argv, filename, gaussian_filter);
    break;
  case 2:
    kernel_callback(argv, filename, bilateral_filter);
    break;
  case 3:
    kernel_callback(argv, filename, guided_filter);
    break;
  default:
    printf("Invalid selection\n");
    return 1;
  }

  return 0;
}
