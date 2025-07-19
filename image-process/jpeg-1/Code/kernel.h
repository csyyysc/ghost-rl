#include <stdio.h>
#include <string>

#ifndef KERNEL_H
#define KERNEL_H

#define MAXSIZE 1024
#define MAXAMP 8.0e06

class Image;

int choose_filter();

void kernel_callback(char **argv, const char *filename,
                     void (*func)(Image *, Image *));

void DCT8x8(double input[8][8], double output[8][8]);
void IDCT8x8(double input[8][8], double output[8][8]);

// JPEG DCT-IDCT Filters
void verify_dct_idct(Image *in, Image *out);
void dct_idct(Image *in, Image *out);
void dct_visualize(Image *in, Image *out);
void dct_energy(Image *in, Image *out);
void dct_random_analysis(Image *in, Image *out);

#endif