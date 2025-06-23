#include "kernel.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {

  int choice = choose_filter();
  const char *filename = argc > 2 ? argv[2] : "output";

  switch (choice) {
  case 1:
    kernel_callback(argv, filename, luminance_inversion);
    break;
  case 2:
    kernel_callback(argv, filename, channel_swapping);
    break;
  default:
    printf("Invalid selection\n");
    return 1;
  }

  return 0;
}