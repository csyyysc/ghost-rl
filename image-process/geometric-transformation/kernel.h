#include <stdio.h>

#ifndef KERNEL_H
#define KERNEL_H

class Image;

int choose_filter();

// Overloaded callback functions
void kernel_callback(char **argv, const char *filename,
                     void (*func)(Image *, Image *, double));
void kernel_callback(char **argv, const char *filename,
                     void (*func)(Image *, Image *, double, double));
void kernel_callback_transform(char **argv, const char *filename,
                               void (*func)(Image *, Image *, double[3][3]));
void kernel_callback_projective_transform(char **argv, const char *filename,
                                          void (*func)(Image *, Image *,
                                                       double mat[3][3]));

int bilinear_interpolation(Image *in, double xx, double yy);
void translation_transform(Image *in, Image *out, double delta_x,
                           double delta_y);
void scaling_transform(Image *in, Image *out, double scale_x, double scale_y);
void rotation_transform(Image *in, Image *out, double angle);
void transform(Image *in, Image *out, double mat[3][3]);
void projective_transform(Image *in, Image *out, double mat[3][3]);

#endif