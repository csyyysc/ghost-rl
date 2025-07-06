#include "kernel.h"
#include "image.h"

const int GAUSSIAN_WINSIZE = 5;
const int BILATERAL_WINSIZE = 5;
const int GUIDED_WINSIZE = 7;

void kernel_callback(char **argv, const char *filename,
                     void (*func)(Image *, Image *, float, float)) {
  Image *in = new Image();
  in->read(argv[1]);

  Image *out = new Image();
  out->init(in->getWidth(), in->getHeight(), in->getCH());

  if (argv[3] == NULL) {
    argv[3] = (char *)"1.0";
  }
  if (argv[4] == NULL) {
    argv[4] = (char *)"1.0";
  }

  float sigma_s = atof(argv[3]);
  float sigma_r = atof(argv[4]);

  func(in, out, sigma_s, sigma_r);

  out->save(filename);

  delete in;
  delete out;
}

int choose_filter() {
  int choice;

  printf("Select filter to apply:\n");
  printf("1. Gaussian Filter\n");
  printf("2. Bilateral Filter\n");
  printf("3. Guided Filter\n");
  printf("Enter selection (1-3): ");
  scanf("%d", &choice);

  return choice;
}

void gaussian_filter(Image *in, Image *out, float sigma_s, float sigma_r) {
  int W = in->getWidth();
  int H = in->getHeight();

  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      double sum = 0.0;
      double w_sum = 0.0;
      double center_value = in->get(x, y);

      for (int j = -GAUSSIAN_WINSIZE; j <= GAUSSIAN_WINSIZE; ++j) {
        for (int i = -GAUSSIAN_WINSIZE; i <= GAUSSIAN_WINSIZE; ++i) {
          int xp = x + i;
          int yp = y + j;

          if (xp < 0)
            xp = 0;
          if (xp >= W)
            xp = W - 1;
          if (yp < 0)
            yp = 0;
          if (yp >= H)
            yp = H - 1;

          double nb_val = in->get(xp, yp);
          double dist = sqrt(i * i + j * j);
          double w = exp(-(dist * dist) / (2.0 * sigma_s * sigma_s));

          w_sum += w;
          sum += w * nb_val;
        }
      }
      double val = sum / w_sum;
      out->set(x, y, val);
    }
  }
}

void bilateral_filter(Image *in, Image *out, float sigma_s, float sigma_r) {
  int W = in->getWidth();
  int H = in->getHeight();

  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      double sum = 0.0;
      double w_sum = 0.0;
      double center_value = in->get(x, y);

      for (int j = -BILATERAL_WINSIZE; j <= BILATERAL_WINSIZE; ++j) {
        for (int i = -BILATERAL_WINSIZE; i <= BILATERAL_WINSIZE; ++i) {
          int xp = x + i;
          int yp = y + j;

          if (xp < 0)
            xp = 0;
          if (xp >= W)
            xp = W - 1;
          if (yp < 0)
            yp = 0;
          if (yp >= H)
            yp = H - 1;

          double nb_val = in->get(xp, yp);

          double dist = sqrt(i * i + j * j);
          double s_w = exp(-(dist * dist) / (2.0 * sigma_s * sigma_s));

          double inten_diff = fabs(center_value - nb_val);
          double r_w =
              exp(-(inten_diff * inten_diff) / (2.0 * sigma_r * sigma_r));

          double w = s_w * r_w;

          sum += w * nb_val;
          w_sum += w;
        }
      }

      double val = sum / w_sum;
      out->set(x, y, val);
    }
  }
}

void guided_filter(Image *in, Image *out, float eps, float r) {
  int W = in->getWidth();
  int H = in->getHeight();
  int C = in->getCH();

  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      int window_count = 0;
      double window_sum = 0.0;
      double window_sum_sq = 0.0;

      for (int j = -GUIDED_WINSIZE / 2; j <= GUIDED_WINSIZE / 2; ++j) {
        for (int i = -GUIDED_WINSIZE / 2; i <= GUIDED_WINSIZE / 2; ++i) {
          int xp = x + i;
          int yp = y + j;

          if (xp < 0)
            xp = 0;
          if (xp >= W)
            xp = W - 1;
          if (yp < 0)
            yp = 0;
          if (yp >= H)
            yp = H - 1;

          double nb_val = in->get(xp, yp);
          window_sum += nb_val;
          window_sum_sq += nb_val * nb_val;
          ++window_count;
        }
      }

      double window_avg = window_sum / window_count;
      double window_var =
          (window_sum_sq / window_count) - (window_avg * window_avg);

      double tilt = window_var / (window_var + eps);
      double intercept = window_avg - tilt * window_avg;

      double center_value = in->get(x, y);
      double filtered_value = tilt * center_value + intercept;

      filtered_value = std::max(0.0, std::min(255.0, filtered_value));
      out->set(x, y, filtered_value);
    }
  }
}