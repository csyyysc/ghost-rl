#include <stdio.h>

#ifndef KERNEL_H
#define KERNEL_H

class Image;

int choose_filter();

void kernel_callback(char **argv, const char *name,
                     void (*func)(Image *, Image *, int, FILE *));

void histogram(Image *in, Image *out, int binsize, FILE *fp);
void histogram_equalization(Image *in, Image *out, int binsize, FILE *fp);

#endif