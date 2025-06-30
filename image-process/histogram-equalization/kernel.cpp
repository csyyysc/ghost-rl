#include "kernel.h"
#include "image.h"

void kernel_callback(char **argv, const char *filename,
                     void (*func)(Image *, Image *, int, FILE *)) {
  Image *in = new Image();
  in->read(argv[1]);

  Image *out = new Image();
  out->init(in->getWidth(), in->getHeight(), in->getCH());

  int binsize = atoi(argv[3]);
  FILE *fp = fopen(filename, "w");

  func(in, out, binsize, fp);

  out->save(filename);

  fclose(fp);
  delete in;
  delete out;
}

int choose_filter() {
  int choice;

  printf("Select filter to apply:\n");
  printf("1. Histogram\n");
  printf("2. Histogram Equalization\n");
  printf("Enter selection (1-2): ");
  scanf("%d", &choice);

  return choice;
}

void histogram(Image *in, Image *out, int binsize, FILE *fp) {
  int W = in->getWidth();
  int H = in->getHeight();
  int C = in->getCH();

  int nBins = 256 / binsize;

  if (C == 1) {
    int *histdata = new int[nBins];

    for (int idx = 0; idx < nBins; idx++) {
      histdata[idx] = 0;
    }

    for (int y = 0; y < H; y++) {
      for (int x = 0; x < W; x++) {
        int value = in->get(x, y);
        int bin = value / binsize;
        histdata[bin]++;
      }
    }

    fprintf(fp, "binsize: %d, nBins: %d\n", binsize, nBins);
    for (int idx = 0; idx < nBins; idx++) {
      int mid = idx * binsize + binsize / 2;

      fprintf(fp, "[%.03d-%.03d]\t", idx * binsize, (idx + 1) * binsize - 1);

      int num = histdata[idx];
      int denom = 0.1 * W * H / nBins;
      for (int cnt = 0; cnt < num / denom; cnt++) {
        fprintf(fp, "*");
      }
      fprintf(fp, "\n");
    }
  } else {
    int *r_histdata = new int[nBins];
    int *g_histdata = new int[nBins];
    int *b_histdata = new int[nBins];

    for (int idx = 0; idx < nBins; idx++) {
      r_histdata[idx] = 0;
      g_histdata[idx] = 0;
      b_histdata[idx] = 0;
    }

    for (int y = 0; y < H; y++) {
      for (int x = 0; x < W; x++) {
        int rvalue = in->get(x, y, 0);
        int gvalue = in->get(x, y, 1);
        int bvalue = in->get(x, y, 2);
        r_histdata[rvalue]++;
        g_histdata[gvalue]++;
        b_histdata[bvalue]++;
      }
    }

    for (int idx = 0; idx < nBins; idx++) {
      int mid = idx * binsize + binsize / 2;
      fprintf(fp, "%d\t%d\n", mid, r_histdata[idx]);
      fprintf(fp, "%d\t%d\n", mid, g_histdata[idx]);
      fprintf(fp, "%d\t%d\n", mid, b_histdata[idx]);
    }
  }
}

void histogram_equalization(Image *in, Image *out, int binsize, FILE *fp) {
  // binsize if not needed here, but for convenience it is kept

  int W = in->getWidth();
  int H = in->getHeight();
  int C = in->getCH();

  if (C == 1) {
    int histdata[256] = {0};
    for (int y = 0; y < H; y++) {
      for (int x = 0; x < W; x++) {
        int value = in->get(x, y);
        histdata[value]++;
      }
    }

    int cdf[256] = {0};
    cdf[0] = histdata[0];
    for (int i = 1; i < 256; i++) {
      cdf[i] = cdf[i - 1] + histdata[i];
    }

    int cdf_min = 0;
    for (int i = 0; i < 256; i++) {
      if (cdf[i] != 0) {
        cdf_min = cdf[i];
        break;
      }
    }

    int table[256];
    for (int i = 0; i < 256; i++) {
      if (cdf[i] == 0) {
        table[i] = 0;
        fprintf(fp, "%d\t0\n", i);
      } else {
        double equalized =
            round(((cdf[i] - cdf_min) * 255.0) / (W * H - cdf_min));
        table[i] = (int)equalized;
        fprintf(fp, "%d\t%d\n", i, table[i]);
      }
    }

    for (int y = 0; y < H; y++) {
      for (int x = 0; x < W; x++) {
        int original_value = in->get(x, y);
        int equalized_value = table[original_value];
        out->set(x, y, equalized_value);
      }
    }
  } else {
    // Convert from RGB to YUV
    for (int y = 0; y < H; y++) {
      for (int x = 0; x < W; x++) {
        double r_value = in->get(x, y, 0);
        double g_value = in->get(x, y, 1);
        double b_value = in->get(x, y, 2);

        double y_value = 0.299 * r_value + 0.587 * g_value + 0.114 * b_value;
        double u_value =
            -0.1687 * r_value - 0.3313 * g_value + 0.5 * b_value + 128;
        double v_value =
            0.5 * r_value - 0.4187 * g_value - 0.0813 * b_value + 128;

        out->set(x, y, 0, y_value);
        out->set(x, y, 1, u_value);
        out->set(x, y, 2, v_value);
      }
    }

    int histdata[256] = {0};
    for (int y = 0; y < H; y++) {
      for (int x = 0; x < W; x++) {
        int value = out->get(x, y, 0);
        histdata[value]++;
      }
    }

    int cdf[256] = {0};
    cdf[0] = histdata[0];
    for (int i = 1; i < 256; i++) {
      cdf[i] = cdf[i - 1] + histdata[i];
    }

    int cdf_min = 0;
    for (int i = 0; i < 256; i++) {
      if (cdf[i] != 0) {
        cdf_min = cdf[i];
        break;
      }
    }

    int table[256];
    for (int i = 0; i < 256; i++) {
      if (cdf[i] == 0) {
        table[i] = 0;
        fprintf(fp, "%d\t0\n", i);
      } else {
        double equalized =
            round(((cdf[i] - cdf_min) * 255.0) / (W * H - cdf_min));
        table[i] = (int)equalized;
        fprintf(fp, "%d\t%d\n", i, table[i]);
      }
    }

    // Apply equalization to Y channel in output image
    for (int y = 0; y < H; y++) {
      for (int x = 0; x < W; x++) {
        int original_value = out->get(x, y, 0);
        int equalized_value = table[original_value];
        out->set(x, y, 0, equalized_value);
      }
    }

    // Convert from YUV to RGB using equalized Y channel
    for (int y = 0; y < H; y++) {
      for (int x = 0; x < W; x++) {
        double y_value = out->get(x, y, 0);
        double u_value = out->get(x, y, 1);
        double v_value = out->get(x, y, 2);

        double r_value = y_value + 1.402 * (v_value - 128);
        double g_value =
            y_value - 0.34414 * (u_value - 128) - 0.71414 * (v_value - 128);
        double b_value = y_value + 1.772 * (u_value - 128);

        r_value = std::max(0.0, std::min(255.0, r_value));
        g_value = std::max(0.0, std::min(255.0, g_value));
        b_value = std::max(0.0, std::min(255.0, b_value));

        out->set(x, y, 0, r_value);
        out->set(x, y, 1, g_value);
        out->set(x, y, 2, b_value);
      }
    }
  }
}