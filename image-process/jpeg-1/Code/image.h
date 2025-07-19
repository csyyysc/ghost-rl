#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

class Image {
private:
  int mW;
  int mH;
  int mCH;
  double *mData;

  void SkipComments(FILE *fp) {
    int ch;
    char line[100];

    while ((ch = fgetc(fp)) != EOF && isspace(ch))
      ;

    if (ch == '#') {
      fgets(line, sizeof(line), fp);
      SkipComments(fp);
    } else {
      fseek(fp, -1, SEEK_CUR);
    }
  }

public:
  Image(void) { this->mData = NULL; };

  ~Image() {
    if (this->mData != NULL) {
      delete[] this->mData;
    }
  }

  void init(int W, int H, int CH) {
    this->mW = W;
    this->mH = H;
    this->mCH = CH;
    if (this->mData != NULL)
      delete[] this->mData;
    this->mData = new double[(this->mW) * (this->mH) * (this->mCH)];
  }

  int getWidth(void) { return this->mW; }

  int getHeight(void) { return this->mH; }

  int getCH(void) { return this->mCH; }

  double *getDataPtr(void) { return this->mData; }

  double get(int x, int y) const { return this->mData[y * (this->mW) + x]; }

  double get(int x, int y, int CH) const {
    return this->mData[this->mCH * (y * (this->mW) + x) + CH];
  }

  void set(int x, int y, double value) {
    this->mData[y * (this->mW) + x] = value;
  }

  void set(int x, int y, int CH, double value) {
    this->mData[this->mCH * (y * (this->mW) + x) + CH] = value;
  }

  void getInfo(void);
  void read(const char *filename);
  void save(const char *filename);
  void divide(int size);

  // Block processing methods
  int getBlockCountX(int blockSize);
  int getBlockCountY(int blockSize);
  void getBlock(int blockX, int blockY, int blockSize, Image &block);
  void setBlock(int blockX, int blockY, int blockSize, const Image &block);
};

void Image::getInfo() {
  printf("Image Resolution: (%d x %d)\n", this->getWidth(), this->getHeight());
}

void Image::read(const char *filename) {

  FILE *file = fopen(filename, "rb");
  if (file == NULL) {
    printf("Cannot open %s\n", filename);
    exit(1);
  }
  printf("Read an image from: %s\n", filename);

  char buf[256];
  char filetype[256];
  int W, H, Range, CH;

  fgets(buf, sizeof(buf), file);
  sscanf(buf, "%s", filetype);

  SkipComments(file);
  fgets(buf, sizeof(buf), file);
  sscanf(buf, "%d%d", &W, &H);

  SkipComments(file);
  fgets(buf, sizeof(buf), file);
  sscanf(buf, "%d", &Range);
  // printf("Header: %s, %d, %d, %d\n", filetype, W, H, Range);

  SkipComments(file);

  if (strcmp(filetype, "P5") == 0) {
    CH = 1;
  } else if (strcmp(filetype, "P6") == 0) {
    CH = 3;
  } else {
    printf("Unknown image type\n");
    exit(1); // abnormal termination
  }

  if (Range != 255) {
    printf("Invalid data\n");
    exit(1); // abnormal termination
  }

  // create Image class

  init(W, H, CH);

  // read ppm data

  int datalength = this->mW * this->mH * this->mCH;
  unsigned char *temp = new unsigned char[datalength];
  fread(temp, sizeof(unsigned char), datalength, file);

  double *ptr1 = this->mData;
  unsigned char *ptr2 = temp;

  for (int i = 0; i < datalength; i++) {
    *ptr1 = (double)*ptr2;
    ptr1++;
    ptr2++;
  }

  delete[] temp;

  fclose(file);
}

void Image::save(const char *filename) {

  char filenamefull[256];
  if (this->mCH == 1) {
    sprintf(filenamefull, "%s.pgm", filename);
  } else {
    sprintf(filenamefull, "%s.ppm", filename);
  }

  FILE *file = fopen(filenamefull, "wb");
  printf("Write an image to: %s \n", filenamefull);

  if (this->mCH == 1) {
    fprintf(file, "P5\n");
  } else {
    fprintf(file, "P6\n");
  }

  fprintf(file, "%d %d\n", this->mW, this->mH);
  fprintf(file, "255\n");

  int datalength = this->mW * this->mH * this->mCH;
  unsigned char *temp = new unsigned char[datalength];

  double *ptr1 = this->mData;
  unsigned char *ptr2 = temp;

  for (int i = 0; i < datalength; i++) {
    double value = *ptr1;
    value = round(value);
    if (value > 255)
      value = 255;
    if (value < 0)
      value = 0;
    *ptr2 = (unsigned char)value;
    ptr1++;
    ptr2++;
  }

  fwrite(temp, sizeof(unsigned char), datalength, file);
  delete[] temp;

  fclose(file);
}

void Image::divide(int size) {
  printf("Image divided into %dx%d blocks of size %dx%d\n",
         getBlockCountX(size), getBlockCountY(size), size, size);
}

int Image::getBlockCountX(int blockSize) {
  return (this->mW + blockSize - 1) / blockSize;
}

int Image::getBlockCountY(int blockSize) {
  return (this->mH + blockSize - 1) / blockSize;
}

void Image::getBlock(int blockX, int blockY, int blockSize, Image &block) {
  int xStart = blockX * blockSize;
  int yStart = blockY * blockSize;
  int xEnd = xStart + blockSize;
  int yEnd = yStart + blockSize;

  if (xEnd > this->mW)
    xEnd = this->mW;
  if (yEnd > this->mH)
    yEnd = this->mH;

  block.init(xEnd - xStart, yEnd - yStart, this->mCH);

  for (int y = yStart; y < yEnd; y++) {
    for (int x = xStart; x < xEnd; x++) {
      if (this->mCH == 1) {
        block.set(x - xStart, y - yStart, this->get(x, y));
      } else {
        for (int CH = 0; CH < this->mCH; CH++) {
          block.set(x - xStart, y - yStart, CH, this->get(x, y, CH));
        }
      }
    }
  }
}

void Image::setBlock(int blockX, int blockY, int blockSize,
                     const Image &block) {
  int xStart = blockX * blockSize;
  int yStart = blockY * blockSize;
  int xEnd = xStart + blockSize;
  int yEnd = yStart + blockSize;

  if (xEnd > this->mW)
    xEnd = this->mW;
  if (yEnd > this->mH)
    yEnd = this->mH;

  for (int y = yStart; y < yEnd; y++) {
    for (int x = xStart; x < xEnd; x++) {
      if (this->mCH == 1) {
        this->set(x, y, block.get(x - xStart, y - yStart));
      } else {
        for (int CH = 0; CH < this->mCH; CH++) {
          this->set(x, y, CH, block.get(x - xStart, y - yStart, CH));
        }
      }
    }
  }
}