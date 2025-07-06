#include "kernel.h"
#include "image.h"

void kernel_callback(char **argv, const char *filename,
                     void (*func)(Image *, Image *, double)) {
  Image *in = new Image();
  in->read(argv[1]);

  Image *out = new Image();
  out->init(in->getWidth(), in->getHeight(), in->getCH());

  double degree = atof(argv[3]);

  func(in, out, degree);

  out->save(filename);

  delete in;
  delete out;
}

void kernel_callback(char **argv, const char *filename,
                     void (*func)(Image *, Image *, double, double)) {
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

  double sigma_s = atof(argv[3]);
  double sigma_r = atof(argv[4]);

  func(in, out, sigma_s, sigma_r);

  out->save(filename);

  delete in;
  delete out;
}

void kernel_callback_transform(char **argv, const char *filename,
                               void (*func)(Image *, Image *, double[3][3])) {
  double mat_trans[3][3] = {{1, 0, 150}, {0, 1, 150}, {0, 0, 2}};
  double mat_scale[3][3] = {{2, 0, 0}, {0, 2, 0}, {0, 0, 1}};
  double mat_rot[3][3] = {{cos(M_PI / 7), -sin(M_PI / 7), 0},
                          {sin(M_PI / 7), cos(M_PI / 7), 0},
                          {0, 0, 1}};
  Image *img1 = new Image();
  img1->read(argv[1]);
  int W = img1->getWidth();
  int H = img1->getHeight();
  int C = img1->getCH();

  Image *img_trans = new Image();
  Image *img_scale = new Image();
  Image *img_rot = new Image();

  img_trans->init(W, H, C);
  img_scale->init(W, H, C);
  img_rot->init(W, H, C);

  func(img1, img_trans, mat_trans);      // Translation
  func(img_trans, img_scale, mat_scale); // Scaling
  func(img_scale, img_rot, mat_rot);     // Rotation

  img_rot->save(filename);

  delete img1;
  delete img_trans;
  delete img_scale;
  delete img_rot;
}

void kernel_callback_projective_transform(char **argv, const char *filename,
                                          void (*func)(Image *, Image *,
                                                       double mat[3][3])) {
  // Card 6
  double mat6[3][3] = {{2.20e-1, 1.11e-1, 5.50e1},
                       {-8.22e-2, 1.10e-1, 1.47e2},
                       {9.10e-5, -1.37e-4, 8.20e-1}};

  // Card 7
  double mat7[3][3] = {{2.21e-1, 9.12e-2, 2.20e2},
                       {-6.98e-2, 1.14e-1, 1.79e2},
                       {8.99e-5, -1.46e-4, 7.70e-1}};

  Image *img1 = new Image();
  img1->read(argv[1]);

  int W = img1->getWidth();
  int H = img1->getHeight();
  int C = img1->getCH();

  Image *img2 = new Image();
  img2->init(400, 600, 1);

  func(img1, img2, mat6);

  img2->save(filename);

  delete img1;
  delete img2;
}

int choose_filter() {
  int choice;

  printf("Select filter to apply:\n");
  printf("1. Translation\n");
  printf("2. Scaling\n");
  printf("3. Rotation\n");
  printf("4. Transform\n");
  printf("5. Projective Transform\n");
  printf("Enter selection (1-5): ");
  scanf("%d", &choice);

  return choice;
}

int bilinear_interpolation(Image *in, double val_x, double val_y) {
  int W = in->getWidth();
  int H = in->getHeight();

  int x, y;
  x = int(val_x);
  y = int(val_y);

  int v1, v2, v3, v4;
  v1 = v2 = v3 = v4 = 0;

  if (x >= 0 && x < W - 1 && y >= 0 && y < H - 1) {
    v1 = in->get(x, y);
    v2 = in->get(x + 1, y);
    v3 = in->get(x, y + 1);
    v4 = in->get(x + 1, y + 1);
  }

  int value;
  double dx, dy;

  dx = val_x - x;
  dy = val_y - y;

  // Computes the weighted average of the 4 surrounding pixel values
  //  based on how close the target point is to each of them.

  value = (unsigned char)(v1 * (1 - dx) * (1 - dy) + v2 * dx * (1 - dy) +
                          v3 * (1 - dx) * dy + v4 * dx * dy + 0.5);

  // Casting a float directly to unsigned char truncates it (floors it),
  //  add 0.5 to get nearest integer (standard rounding).

  return value;
}

void translation_transform(Image *in, Image *out, double delta_x,
                           double delta_y) {
  int W = in->getWidth();
  int H = in->getHeight();

  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      double dx = x - delta_x;
      double dy = y - delta_y;

      int value = bilinear_interpolation(in, dx, dy);
      out->set(x, y, value);
    }
  }
}

void scaling_transform(Image *in, Image *out, double scale_x, double scale_y) {
  int W = in->getWidth();
  int H = in->getHeight();

  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      double ratio_x, ratio_y;
      ratio_x = x / scale_x;
      ratio_y = y / scale_y;

      int value = bilinear_interpolation(in, ratio_x, ratio_y);
      out->set(x, y, value);
    }
  }
}

void rotation_transform(Image *in, Image *out, double angle) {
  int W = in->getWidth();
  int H = in->getHeight();

  double rad = angle * M_PI / 180.0;

  double cos_angle = cos(rad);
  double sin_angle = sin(rad);
  double center_x = (W - 1) / 2.0;
  double center_y = (H - 1) / 2.0;

  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      double rotate_x =
          (x - center_x) * cos_angle - (y - center_y) * sin_angle + center_x;
      double rotate_y =
          (x - center_x) * sin_angle + (y - center_y) * cos_angle + center_y;

      int value = bilinear_interpolation(in, rotate_x, rotate_y);
      out->set(x, y, value);
    }
  }
}

void transform(Image *in, Image *out, double mat[3][3]) {
  int W = in->getWidth();
  int H = in->getHeight();

  double center_x = (W - 1) / 2.0;
  double center_y = (H - 1) / 2.0;

  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      double centered_x = x - center_x;
      double centered_y = y - center_y;

      double xx = mat[0][0] * centered_x + mat[0][1] * centered_y + mat[0][2];
      double yy = mat[1][0] * centered_x + mat[1][1] * centered_y + mat[1][2];
      double ss = mat[2][0] * centered_x + mat[2][1] * centered_y + mat[2][2];

      // Handle homogeneous coordinates
      if (ss != 0) {
        xx /= ss;
        yy /= ss;
      }

      xx += center_x;
      yy += center_y;

      int value = bilinear_interpolation(in, xx, yy);
      out->set(x, y, value);
    }
  }
}

void projective_transform(Image *in, Image *out, double mat[3][3]) {
  int out_W = out->getWidth();  // Use OUTPUT image dimensions
  int out_H = out->getHeight(); // Use OUTPUT image dimensions

  for (int y = 0; y < out_H; ++y) {   // Process ALL output rows
    for (int x = 0; x < out_W; ++x) { // Process ALL output columns
      double xx = mat[0][0] * x + mat[0][1] * y + mat[0][2];
      double yy = mat[1][0] * x + mat[1][1] * y + mat[1][2];
      double ss = mat[2][0] * x + mat[2][1] * y + mat[2][2];

      if (ss != 0) {
        xx /= ss;
        yy /= ss;
      }

      int value = bilinear_interpolation(in, xx, yy);
      out->set(x, y, value);
    }
  }
}