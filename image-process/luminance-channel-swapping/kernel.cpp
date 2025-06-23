#include "kernel.h"
#include "image.h"

void kernel_callback(char **argv, const char *filename,
                     void (*filter)(Image *, Image *)) {
  Image *in = new Image();
  in->read(argv[1]);

  Image *out = new Image();
  out->init(in->getWidth(), in->getHeight(), in->getCH());

  filter(in, out);
  out->save(filename);

  delete in;
  delete out;
}

int choose_filter() {
  int choice;

  printf("Select filter to apply:\n");
  printf("1. Luminance Inversion\n");
  printf("2. Channel Swapping\n");
  printf("Enter selection (1-2): ");
  scanf("%d", &choice);

  return choice;
}

void luminance_inversion(Image *in, Image *out) {
  int W = in->getWidth();
  int H = in->getHeight();

  in->getInfo();

  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      double value = in->get(x, y);
      out->set(x, y, 255 - value);
    }
  }
}

void channel_swapping(Image *in, Image *out) {
  int W = in->getWidth();
  int H = in->getHeight();
  int CH = in->getCH();

  in->getInfo();

  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      double r_value = in->get(x, y, 0);
      double g_value = in->get(x, y, 1);
      double b_value = in->get(x, y, 2);

      out->set(x, y, 0, b_value);
      out->set(x, y, 1, r_value);
      out->set(x, y, 2, g_value);
    }
  }
}