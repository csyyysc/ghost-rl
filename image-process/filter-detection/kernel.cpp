#include "kernel.h"
#include "image.h"

void kernel_filter_callback(char **argv, const char *filename,
                            const int winsize,
                            void (*filter)(Image *, Image *, int)) {
  Image *in = new Image();
  in->read(argv[1]);

  Image *out = new Image();
  out->init(in->getWidth(), in->getHeight(), in->getCH());

  filter(in, out, winsize);
  out->save(filename);

  delete in;
  delete out;
}

int choose_filter() {
  int choice;

  printf("Select filter to apply:\n");
  printf("1. Mean Filter\n");
  printf("2. Horizontal Edge Filter\n");
  printf("3. Vertical Edge Filter\n");
  printf("4. Laplacian Filter\n");
  printf("5. Enhance Mean Filter\n");
  printf("6. Enhance Edge Filter\n");
  printf("Enter selection (1-6): ");
  scanf("%d", &choice);

  printf("--------------------------------\n");
  return choice;
}

int choose_winsize() {
  int winsize;

  printf("Select window size:\n");
  printf("1. 3x3\n");
  printf("2. 5x5\n");
  printf("3. 7x7\n");
  printf("4. 9x9\n");
  printf("5. 11x11\n");
  printf("Enter selection (1-3): ");
  scanf("%d", &winsize);

  printf("--------------------------------\n");
  return winsize;
}

void mean_filter(Image *in, Image *out, const int winsize) {
  int W = in->getWidth();
  int H = in->getHeight();
  int C = in->getCH();

  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      int cnt = 0;
      double mean_out = 0;

      for (int j = -winsize; j <= winsize; ++j) {
        int yp = y + j;

        if (yp < 0)
          yp = yp + H;
        if (yp > H - 1)
          yp = yp - H;

        for (int i = -winsize; i <= winsize; ++i) {
          int xp = x + i;
          if (xp < 0)
            xp = xp + W;
          if (xp > W - 1)
            xp = xp - W;

          double val_in = in->get(xp, yp);
          mean_out = mean_out + val_in;
          cnt = cnt + 1;
        }
      }

      mean_out = (double)mean_out / (double)cnt;
      out->set(x, y, mean_out);
    }
  }
}

void horizontal_edge_filter(Image *in, Image *out, const int winsize) {
  int W = in->getWidth();
  int H = in->getHeight();
  int C = in->getCH();

  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      double sobel_hrz_val = 0.0;

      for (int j = -winsize; j <= winsize; ++j) {
        int yp = y + j;
        if (yp < 0)
          yp = yp + H;
        if (yp > H - 1)
          yp = yp - H;

        for (int i = -winsize; i <= winsize; ++i) {
          int xp = x + i;
          if (xp < 0)
            xp = xp + W;
          if (xp > W - 1)
            xp = xp - W;

          double val_in = in->get(xp, yp);
          sobel_hrz_val += val_in * SOBEL_HRZ_K[j + 1][i + 1];
        }
      }

      sobel_hrz_val = std::max(0.0, std::min(255.0, sobel_hrz_val + 128.0));
      out->set(x, y, sobel_hrz_val);
    }
  }
}

void vertical_edge_filter(Image *in, Image *out, const int winsize) {
  int W = in->getWidth();
  int H = in->getHeight();
  int C = in->getCH();

  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      double sobel_vrt_val = 0.0;

      for (int j = -winsize; j <= winsize; ++j) {
        int yp = y + j;
        if (yp < 0)
          yp = 0;
        if (yp > H - 1)
          yp = H - 1;

        for (int i = -winsize; i <= winsize; ++i) {
          int xp = x + i;
          if (xp < 0)
            xp = xp + W;
          if (xp > W - 1)
            xp = xp - W;

          double val_in = in->get(xp, yp);
          sobel_vrt_val += val_in * SOBEL_VRT_K[j + 1][i + 1];
        }
      }

      sobel_vrt_val = std::max(0.0, std::min(255.0, sobel_vrt_val + 128.0));
      out->set(x, y, sobel_vrt_val);
    }
  }
}

void laplacian_filter(Image *in, Image *out, const int winsize) {
  int W = in->getWidth();
  int H = in->getHeight();
  int C = in->getCH();

  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      double lap_val = 0.0;

      for (int j = -winsize; j <= winsize; ++j) {
        int yp = y + j;
        if (yp < 0)
          yp = 0;
        if (yp > H - 1)
          yp = H - 1;

        for (int i = -winsize; i <= winsize; ++i) {
          int xp = x + i;
          if (xp < 0)
            xp = 0;
          if (xp > W - 1)
            xp = W - 1;

          double val_in = in->get(xp, yp);
          lap_val += val_in * LAPLACIAN_K_DIAG[j + 1][i + 1];
        }
      }

      lap_val = std::max(0.0, std::min(255.0, lap_val));
      out->set(x, y, lap_val);
    }
  }
}

void enhance_mean_filter(Image *in, Image *out, const int winsize) {
  int W = in->getWidth();
  int H = in->getHeight();
  int C = in->getCH();

  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      int count = 0;
      double mean_out = 0.0;

      for (int j = -winsize; j <= winsize; ++j) {
        int yp = y + j;
        if (yp < 0)
          yp = 0;
        if (yp > H - 1)
          yp = H - 1;

        for (int i = -winsize; i <= winsize; ++i) {
          int xp = x + i;
          if (xp < 0)
            xp = 0;
          if (xp > W - 1)
            xp = W - 1;

          double val_in = in->get(xp, yp);
          mean_out += val_in;
          count++;
        }
      }

      mean_out = mean_out / (double)count;
      double origin_val = in->get(x, y);
      double val_out = origin_val + (origin_val - mean_out);

      val_out = std::max(0.0, std::min(255.0, val_out));
      out->set(x, y, val_out);
    }
  }
}

void enhance_edge_filter(Image *in, Image *out, const int winsize) {
  int W = in->getWidth();
  int H = in->getHeight();
  int C = in->getCH();

  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      double val_out = 0.0;
      double edge_out = 0.0;

      for (int j = -winsize; j <= winsize; ++j) {
        int yp = y + j;
        if (yp < 0)
          yp = 0;
        if (yp > H - 1)
          yp = H - 1;

        for (int i = -winsize; i <= winsize; ++i) {
          int xp = x + i;
          if (xp < 0)
            xp = xp + W;
          if (xp > W - 1)
            xp = xp - W;

          double val_in = in->get(xp, yp);
          edge_out += val_in * LAPLACIAN_K[j + 1][i + 1];
        }
      }

      double origin_val = in->get(x, y);
      val_out += origin_val + edge_out;
      val_out = std::max(0.0, std::min(255.0, val_out + 128.0));

      out->set(x, y, val_out);
    }
  }
}