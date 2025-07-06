#include "kernel.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  printf("Usage: %s <input_image> <output_image> <x> <y>\n", argv[0]);
  printf("--------------------------------\n");

  int choice = choose_filter();
  const char *filename = argc > 2 ? argv[2] : "output";

  switch (choice) {
  case 1:
    kernel_callback(argv, filename, translation_transform);
    break;
  case 2:
    kernel_callback(argv, filename, scaling_transform);
    break;
  case 3:
    kernel_callback(argv, filename, rotation_transform);
    break;
  case 4:
    kernel_callback_transform(argv, filename, transform);
    break;
  case 5:
    kernel_callback_projective_transform(argv, filename, projective_transform);
    break;
  default:
    printf("Invalid selection\n");
    return 1;
  }

  return 0;
}
