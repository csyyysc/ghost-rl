#include "kernel.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {

  int choice = choose_filter();

  int winsize = 1;
  if (choice == 1) {
    winsize = choose_winsize();
  }

  const char *filename = argc > 2 ? argv[2] : "output";

  switch (choice) {
  case 1:
    kernel_filter_callback(argv, filename, winsize, mean_filter);
    break;
  case 2:
    kernel_filter_callback(argv, filename, winsize, horizontal_edge_filter);
    break;
  case 3:
    kernel_filter_callback(argv, filename, winsize, vertical_edge_filter);
    break;
  case 4:
    kernel_filter_callback(argv, filename, winsize, laplacian_filter);
    break;
  case 5:
    kernel_filter_callback(argv, filename, winsize, enhance_mean_filter);
    break;
  case 6:
    kernel_filter_callback(argv, filename, winsize, enhance_edge_filter);
    break;
  default:
    printf("Invalid selection\n");
    return 1;
  }

  return 0;
}