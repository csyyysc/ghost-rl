#ifndef KERNEL_H
#define KERNEL_H

class Image;

int choose_filter();

void kernel_callback(char **argv, const char *name,
                     void (*filter)(Image *, Image *));

void luminance_inversion(Image *in, Image *out);
void channel_swapping(Image *in, Image *out);

#endif