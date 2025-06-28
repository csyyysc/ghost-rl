#ifndef KERNEL_H
#define KERNEL_H

class Image;

const int ERROR_DIFFUTION_DIRS[4][2] = {{1, 0}, {0, 1}, {-1, 1}, {1, 1}};

const int ERROR_DIFFUTION_WEIGHTS[4] = {7, 5, 3, 1};

const float ERROR_DIFFUTION_SUM = 16.0;

// 課題3: フィルター選択関数
int choose_filter();

// 課題3: カーネル処理の共通関数
void kernel_filter_callback(char **argv, const char *name,
                            void (*filter)(Image *, Image *, double));

// 課題3: フィルターの定義
void binarization_fixed_threshold(Image *in, Image *out, double threshold);
void binarization_error_diffusion(Image *in, Image *out, double threshold);

#endif