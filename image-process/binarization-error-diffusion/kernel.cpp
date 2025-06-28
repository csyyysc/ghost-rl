#include "kernel.h"
#include "image.h"
#include <iostream>

void kernel_filter_callback(char **argv, const char *filename,
                            void (*filter)(Image *, Image *, double)) {
  Image *in = new Image();
  in->read(argv[1]);

  Image *out = new Image();
  out->init(in->getWidth(), in->getHeight(), in->getCH());

  double threshold = atof(argv[3]);

  filter(in, out, threshold);
  out->save(filename);

  delete in;
  delete out;
}

int choose_filter() {
  int choice;

  printf("Select filter to apply:\n");
  printf("1. Binarization Fixed Threshold\n");
  printf("2. Binarization Error Diffusion\n");
  scanf("%d", &choice);

  printf("--------------------------------\n");
  return choice;
}

void binarization_fixed_threshold(Image *in, Image *out, double threshold) {
  int W = in->getWidth();
  int H = in->getHeight();
  int C = in->getCH();

  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      if (C == 1) {
        double val_in = in->get(x, y);
        double val_out = 0.0;

        if (val_in > threshold)
          val_out = 255.0;

        out->set(x, y, val_out);
      } else {
        double r_value = in->get(x, y, 0);
        double g_value = in->get(x, y, 1);
        double b_value = in->get(x, y, 2);

        if (r_value > threshold)
          r_value = 255.0;
        else
          r_value = 0.0;

        if (g_value > threshold)
          g_value = 255.0;
        else
          g_value = 0.0;

        if (b_value > threshold)
          b_value = 255.0;
        else
          b_value = 0.0;

        out->set(x, y, 0, r_value);
        out->set(x, y, 1, g_value);
        out->set(x, y, 2, b_value);
      }
    }
  }
}

void binarization_error_diffusion(Image *in, Image *out, double threshold) {
  int W = in->getWidth();
  int H = in->getHeight();
  int C = in->getCH();

  std::cout << "threshold: " << threshold << std::endl;

  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      if (C == 1) {
        double val_in = in->get(x, y);
        double val_out = val_in > threshold ? 255.0 : 0.0;
        double error = val_in - val_out;
        out->set(x, y, val_out);

        for (int p = 0; p < 4; ++p) {
          int dx = ERROR_DIFFUTION_DIRS[p][0];
          int dy = ERROR_DIFFUTION_DIRS[p][1];
          int xp = x + dx;
          int yp = y + dy;

          if (xp < 0 || xp >= W || yp < 0 || yp >= H)
            continue;

          double neighbor_in_value = in->get(xp, yp);
          neighbor_in_value +=
              error * ERROR_DIFFUTION_WEIGHTS[p] / ERROR_DIFFUTION_SUM;
          in->set(xp, yp, neighbor_in_value);
        }
      } else {
        double val_r_in = in->get(x, y, 0);
        double val_g_in = in->get(x, y, 1);
        double val_b_in = in->get(x, y, 2);

        double val_r_out = val_r_in > threshold ? 255.0 : 0.0;
        double val_g_out = val_g_in > threshold ? 255.0 : 0.0;
        double val_b_out = val_b_in > threshold ? 255.0 : 0.0;

        out->set(x, y, 0, val_r_out); // Set binarized R
        out->set(x, y, 1, val_g_out); // Set binarized G
        out->set(x, y, 2, val_b_out); // Set binarized B

        double error_r = val_r_in - val_r_out;
        double error_g = val_g_in - val_g_out;
        double error_b = val_b_in - val_b_out;

        for (int p = 0; p < 4; ++p) {
          int dx = ERROR_DIFFUTION_DIRS[p][0];
          int dy = ERROR_DIFFUTION_DIRS[p][1];
          int xp = x + dx;
          int yp = y + dy;

          if (xp < 0 || xp >= W || yp < 0 || yp >= H)
            continue;

          double val_r_in = in->get(xp, yp, 0);
          double val_g_in = in->get(xp, yp, 1);
          double val_b_in = in->get(xp, yp, 2);
          val_r_in +=
              error_r * ERROR_DIFFUTION_WEIGHTS[p] / ERROR_DIFFUTION_SUM;
          val_g_in +=
              error_g * ERROR_DIFFUTION_WEIGHTS[p] / ERROR_DIFFUTION_SUM;
          val_b_in +=
              error_b * ERROR_DIFFUTION_WEIGHTS[p] / ERROR_DIFFUTION_SUM;

          in->set(xp, yp, 0, val_r_in);
          in->set(xp, yp, 1, val_g_in);
          in->set(xp, yp, 2, val_b_in);
        }
      }
    }
  }
}
