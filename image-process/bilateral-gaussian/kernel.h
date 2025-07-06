#include <stdio.h>

#ifndef KERNEL_H
#define KERNEL_H

class Image;

int choose_filter();

void kernel_callback(char **argv, const char *name,
                     void (*func)(Image *, Image *, float, float));

void gaussian_filter(Image *in, Image *out, float sigma_s, float sigma_r);
void bilateral_filter(Image *in, Image *out, float sigma_s, float sigma_r);
void guided_filter(Image *in, Image *out, float eps, float r);

#endif