#ifndef KERNEL_H
#define KERNEL_H

class Image;

const int MEAN_K[3][3] = {{1, 1, 1}, {1, 1, 1}, {1, 1, 1}};

const int SOBEL_HRZ_K[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};

const int SOBEL_VRT_K[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

const int LAPLACIAN_K[3][3] = {{0, 1, 0}, {1, -4, 1}, {0, 1, 0}};

const int LAPLACIAN_K_DIAG[3][3] = {{1, 1, 1}, {1, -8, 1}, {1, 1, 1}};

// 課題2: フィルター選択関数
int choose_filter();

// 課題2: ウィンドウサイズの選択
int choose_winsize();

// 課題2: カーネル処理の共通関数
void kernel_filter_callback(char **argv, const char *name, const int winsize,
                            void (*filter)(Image *, Image *, int));

// 課題2: フィルターの定義
void mean_filter(Image *in, Image *out, const int winsize);
void horizontal_edge_filter(Image *in, Image *out, const int winsize);
void vertical_edge_filter(Image *in, Image *out, const int winsize);
void laplacian_filter(Image *in, Image *out, const int winsize);
void enhance_mean_filter(Image *in, Image *out, const int winsize);
void enhance_edge_filter(Image *in, Image *out, const int winsize);

#endif