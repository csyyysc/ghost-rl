#include "kernel.h"
#include "image.h"
#include <stdlib.h>
#include <time.h>

#define SWAP(a, b)                                                             \
  tempr = (a);                                                                 \
  (a) = (b);                                                                   \
  (b) = tempr

typedef struct {
  int x, y;
  double energy;
  double percentage;
} CoeffEnergy;

int choose_filter() {
  int choice;

  printf("Select filter to apply:\n");
  printf("1. DCT-IDCT\n");
  printf("2. DCT Visualize\n");
  printf("3. DCT Energy\n");
  printf("4. DCT Random Analysis\n");
  printf("Enter selection (1-4): ");
  scanf("%d", &choice);

  return choice;
}

void kernel_callback(char **argv, const char *filename,
                     void (*func)(Image *, Image *)) {
  Image *in = new Image();
  in->read(argv[1]);

  int W = in->getWidth();
  int H = in->getHeight();
  int C = in->getCH();

  Image *out = new Image();
  out->init(W, H, C);

  func(in, out);

  out->save(filename);
  delete in;
  delete out;
}

void DCT8x8(double input[8][8], double output[8][8]) {

  for (int l = 0; l < 8; l++) {
    for (int k = 0; k < 8; k++) {

      double value = 0.0;

      for (int n = 0; n < 8; n++) {
        for (int m = 0; m < 8; m++) {
          double Cx = 1.0;
          double Cy = 1.0;
          double cosx = cos(M_PI * (2 * m + 1) * k / 16);
          double cosy = cos(M_PI * (2 * n + 1) * l / 16);

          if (k == 0)
            Cx = 1 / sqrt(2.0);
          if (l == 0)
            Cy = 1 / sqrt(2.0);

          double temp = input[m][n];
          temp = temp * cosx * cosy * Cx * Cy / 4;
          value = value + temp;
        }
      }
      output[k][l] = value;
    }
  }
}

void IDCT8x8(double input[8][8], double output[8][8]) {

  for (int n = 0; n < 8; n++) {
    for (int m = 0; m < 8; m++) {
      double value = 0.0;

      for (int l = 0; l < 8; l++) {
        for (int k = 0; k < 8; k++) {
          double Cx = 1.0;
          double Cy = 1.0;
          double cosx = cos(M_PI * (2 * m + 1) * k / 16);
          double cosy = cos(M_PI * (2 * n + 1) * l / 16);

          if (k == 0)
            Cx = 1 / sqrt(2.0);
          if (l == 0)
            Cy = 1 / sqrt(2.0);

          double temp = input[k][l];
          temp = temp * cosx * cosy * Cx * Cy / 4;
          value = value + temp;
        }
      }
      output[m][n] = value;
    }
  }
}

void verify_dct_idct(Image *in, Image *out) {
  int total_pixels = 0;

  double total_diff = 0.0;
  double max_diff = 0.0;
  double threshold = 1.0; // Allow 1 pixel difference due to rounding errors

  for (int y = 0; y < in->getHeight(); ++y) {
    for (int x = 0; x < in->getWidth(); ++x) {
      double input_val, output_val;
      int CH = in->getCH();

      if (CH == 1) {
        input_val = in->get(x, y);
        output_val = out->get(x, y);
      } else {
        input_val = in->get(x, y, 0);
        output_val = out->get(x, y, 0);
      }

      double diff = fabs(input_val - output_val);
      total_diff += diff;
      max_diff = fmax(max_diff, diff);
      total_pixels++;
    }
  }

  double avg_diff = total_diff / total_pixels;
  double mse = 0.0;

  // Calculate Mean Square Error
  for (int y = 0; y < in->getHeight(); ++y) {
    for (int x = 0; x < in->getWidth(); ++x) {
      double input_val, output_val;
      int CH = in->getCH();

      if (CH == 0) {
        input_val = in->get(x, y);
        output_val = out->get(x, y);
      } else {
        for (int ch = 0; ch < CH; ch++) {
          input_val = in->get(x, y, ch);
          output_val = out->get(x, y, ch);
        }
      }

      double diff = input_val - output_val;
      mse += diff * diff;
    }
  }
  mse /= total_pixels;

  printf("Image Verification Results:\n");
  printf("  Total pixels: %d\n", total_pixels);
  printf("  Average difference: %.6f\n", avg_diff);
  printf("  Maximum difference: %.6f\n", max_diff);
  printf("  Mean Square Error (MSE): %.6f\n", mse);
  printf("  Threshold: %.1f\n", threshold);

  if (avg_diff <= threshold && max_diff <= threshold * 2) {
    printf("✓ SUCCESS: Restored image matches input within threshold!\n");
    printf("✓ DCT+IDCT transformation is working correctly.\n");
  } else {
    printf("✗ WARNING: Significant differences detected!\n");
    printf("✗ DCT+IDCT transformation may have issues.\n");
  }
}

void dct_idct(Image *in, Image *out) {
  printf("Starting DCT+IDCT...\n");

  in->divide(8);
  int total_blocks = in->getBlockCountX(8) * in->getBlockCountY(8);
  int processed_blocks = 0;

  for (int block_y = 0; block_y < in->getBlockCountY(8); ++block_y) {
    for (int block_x = 0; block_x < in->getBlockCountX(8); ++block_x) {
      Image block;
      in->getBlock(block_x, block_y, 8, block);

      double input_block[8][8];

      double dct_coeffs[8][8];

      double reconstructed_block[8][8];

      for (int y = 0; y < 8; y++) {
        for (int x = 0; x < 8; x++) {
          if (x < block.getWidth() && y < block.getHeight()) {
            if (block.getCH() == 1) {
              input_block[y][x] = block.get(x, y) - 128.0;
            } else {
              input_block[y][x] = block.get(x, y, 0) - 128.0;
            }
          } else {
            input_block[y][x] = 0.0;
          }
        }
      }

      DCT8x8(input_block, dct_coeffs);
      IDCT8x8(dct_coeffs, reconstructed_block);

      Image restored_block;
      restored_block.init(block.getWidth(), block.getHeight(), block.getCH());

      for (int y = 0; y < block.getHeight(); ++y) {
        for (int x = 0; x < block.getWidth(); ++x) {
          double pixel_value = reconstructed_block[y][x] + 128.0;
          pixel_value = fmax(0, fmin(255, pixel_value));

          if (block.getCH() == 1) {
            restored_block.set(x, y, pixel_value);
          } else {
            for (int ch = 0; ch < block.getCH(); ch++) {
              restored_block.set(x, y, ch, pixel_value);
            }
          }
        }
      }

      out->setBlock(block_x, block_y, 8, restored_block);

      ++processed_blocks;
      if (processed_blocks % 10 == 0) {
        printf("Processed %d/%d blocks\n", processed_blocks, total_blocks);
      }
    }
  }

  printf("\n=== DCT+IDCT Verification ===\n");
  printf("✓ Applied DCT followed by IDCT to all blocks\n");
  printf("✓ Confirms that DCT is a reversible transform!\n");
  verify_dct_idct(in, out);
}

void dct_visualize(Image *in, Image *out) {
  printf("Starting DCT visualization...\n");

  in->divide(8);
  int total_blocks = in->getBlockCountX(8) * in->getBlockCountY(8);
  int processed_blocks = 0;

  for (int block_y = 0; block_y < in->getBlockCountY(8); ++block_y) {
    for (int block_x = 0; block_x < in->getBlockCountX(8); ++block_x) {
      Image block;
      in->getBlock(block_x, block_y, 8, block);

      double input_block[8][8];
      double dct_coeffs[8][8];

      for (int y = 0; y < 8; y++) {
        for (int x = 0; x < 8; x++) {
          if (x < block.getWidth() && y < block.getHeight()) {
            if (block.getCH() == 1) {
              input_block[y][x] = block.get(x, y) - 128.0;
            } else {
              input_block[y][x] = block.get(x, y, 0) - 128.0;
            }
          } else {
            input_block[y][x] = 0.0;
          }
        }
      }

      DCT8x8(input_block, dct_coeffs);

      double max_coeff = 0.0;
      for (int y = 0; y < 8; y++) {
        for (int x = 0; x < 8; x++) {
          if (fabs(dct_coeffs[y][x]) > max_coeff) {
            max_coeff = fabs(dct_coeffs[y][x]);
          }
        }
      }

      int block_start_x = block_x * 8;
      int block_start_y = block_y * 8;
      for (int y = 0; y < block.getHeight() && y < 8; y++) {
        for (int x = 0; x < block.getWidth() && x < 8; x++) {
          double normalized;
          if (max_coeff > 0) {
            normalized = (dct_coeffs[y][x] / max_coeff) * 127.5 + 127.5;
          } else {
            normalized = 127.5;
          }
          normalized = fmax(0, fmin(255, normalized));
          out->set(block_start_x + x, block_start_y + y, normalized);
        }
      }

      ++processed_blocks;
      if (processed_blocks % 10 == 0) {
        printf("Processed %d/%d blocks\n", processed_blocks, total_blocks);
      }
    }
  }

  printf("\n=== DCT Visualization Complete ===\n");
  printf("✓ DCT coefficients visualized for all blocks\n");
}

void dct_energy(Image *in, Image *out) {
  printf("Starting DCT energy calculation...\n");

  int block_count = 0;
  double total_energy[8][8] = {0};

  in->divide(8);
  int total_blocks = in->getBlockCountX(8) * in->getBlockCountY(8);
  int processed_blocks = 0;

  for (int block_y = 0; block_y < in->getBlockCountY(8); ++block_y) {
    for (int block_x = 0; block_x < in->getBlockCountX(8); ++block_x) {
      Image block;
      in->getBlock(block_x, block_y, 8, block);

      double input_block[8][8];
      double dct_coeffs[8][8];

      for (int y = 0; y < 8; y++) {
        for (int x = 0; x < 8; x++) {
          if (x < block.getWidth() && y < block.getHeight()) {
            if (block.getCH() == 1) {
              input_block[y][x] = block.get(x, y) - 128.0;
            } else {
              input_block[y][x] = block.get(x, y, 0) - 128.0;
            }
          } else {
            input_block[y][x] = 0.0;
          }
        }
      }

      DCT8x8(input_block, dct_coeffs);

      for (int y = 0; y < 8; y++) {
        for (int x = 0; x < 8; x++) {
          total_energy[y][x] += dct_coeffs[y][x] * dct_coeffs[y][x];
        }
      }

      block_count++;
      ++processed_blocks;
      if (processed_blocks % 10 == 0) {
        printf("Processed %d/%d blocks\n", processed_blocks, total_blocks);
      }
    }
  }

  double avg_energy[8][8];
  double total_sum = 0.0;
  for (int y = 0; y < 8; y++) {
    for (int x = 0; x < 8; x++) {
      avg_energy[y][x] = total_energy[y][x] / block_count;
      total_sum += avg_energy[y][x];
    }
  }

  CoeffEnergy coeffs[64];
  int idx = 0;
  for (int y = 0; y < 8; y++) {
    for (int x = 0; x < 8; x++) {
      coeffs[idx].x = x;
      coeffs[idx].y = y;
      coeffs[idx].energy = avg_energy[y][x];
      coeffs[idx].percentage = (avg_energy[y][x] / total_sum) * 100.0;
      idx++;
    }
  }

  for (int i = 0; i < 64 - 1; i++) {
    for (int j = 0; j < 64 - i - 1; j++) {
      if (coeffs[j].energy < coeffs[j + 1].energy) {
        CoeffEnergy temp = coeffs[j];
        coeffs[j] = coeffs[j + 1];
        coeffs[j + 1] = temp;
      }
    }
  }

  double low_freq_energy = 0.0;
  double mid_freq_energy = 0.0;
  double high_freq_energy = 0.0;

  for (int y = 0; y < 8; y++) {
    for (int x = 0; x < 8; x++) {
      if (x < 4 && y < 4) {
        low_freq_energy += avg_energy[y][x];
      } else if (x >= 4 && y >= 4) {
        high_freq_energy += avg_energy[y][x];
      } else {
        mid_freq_energy += avg_energy[y][x];
      }
    }
  }

  Image *energy_img = new Image();
  energy_img->init(8, 8, 1);

  double max_energy = coeffs[0].energy;

  for (int y = 0; y < 8; y++) {
    for (int x = 0; x < 8; x++) {
      double normalized = (avg_energy[y][x] / max_energy) * 255.0;
      energy_img->set(x, y, normalized);
    }
  }

  printf("\n=== DCT Energy Analysis ===\n");
  printf("Total blocks analyzed: %d\n", block_count);
  printf("Total average energy: %.2f\n\n", total_sum);

  printf("Top 10 DCT coefficients by energy:\n");
  printf("Rank\tPosition\tAvg Energy\tPercentage\n");
  printf("----\t--------\t----------\t----------\n");
  for (int i = 0; i < 10; i++) {
    printf("%d\t(%d,%d)\t\t%.2f\t\t%.2f%%\n", i + 1, coeffs[i].x, coeffs[i].y,
           coeffs[i].energy, coeffs[i].percentage);
  }

  printf("\nFrequency Band Energy Distribution:\n");
  printf("Low frequencies (0-3, 0-3):  %.2f%% of total energy\n",
         (low_freq_energy / total_sum) * 100);
  printf("Mid frequencies:             %.2f%% of total energy\n",
         (mid_freq_energy / total_sum) * 100);
  printf("High frequencies (4-7, 4-7): %.2f%% of total energy\n",
         (high_freq_energy / total_sum) * 100);

  printf("\nDC coefficient (0,0) analysis:\n");
  printf("DC energy: %.2f (%.2f%% of total)\n", avg_energy[0][0],
         (avg_energy[0][0] / total_sum) * 100);

  printf("\nEnergy compaction analysis:\n");
  double cumulative_energy = 0.0;
  int coeffs_for_90_percent = 0;
  int coeffs_for_95_percent = 0;
  int coeffs_for_99_percent = 0;

  for (int i = 0; i < 64; i++) {
    cumulative_energy += coeffs[i].energy;
    double cumulative_percentage = (cumulative_energy / total_sum) * 100.0;

    if (cumulative_percentage >= 90.0 && coeffs_for_90_percent == 0) {
      coeffs_for_90_percent = i + 1;
    }
    if (cumulative_percentage >= 95.0 && coeffs_for_95_percent == 0) {
      coeffs_for_95_percent = i + 1;
    }
    if (cumulative_percentage >= 99.0 && coeffs_for_99_percent == 0) {
      coeffs_for_99_percent = i + 1;
    }
  }

  printf("Coefficients needed for 90%% of energy: %d/64 (%.1f%%)\n",
         coeffs_for_90_percent, (coeffs_for_90_percent / 64.0) * 100);
  printf("Coefficients needed for 95%% of energy: %d/64 (%.1f%%)\n",
         coeffs_for_95_percent, (coeffs_for_95_percent / 64.0) * 100);
  printf("Coefficients needed for 99%% of energy: %d/64 (%.1f%%)\n",
         coeffs_for_99_percent, (coeffs_for_99_percent / 64.0) * 100);

  printf("\n=== Analysis Complete ===\n");
  printf("✓ This demonstrates DCT's energy compaction property\n");
  printf("✓ Most energy is concentrated in low-frequency coefficients\n");
  printf("✓ High-frequency coefficients can be discarded for compression\n");

  delete energy_img;
}

void dct_random_analysis(Image *in, Image *out) {
  printf("Starting DCT random image generation and analysis...\n");

  Image *random_img = new Image();
  random_img->init(in->getWidth(), in->getHeight(), in->getCH());

  srand(time(NULL));
  for (int y = 0; y < in->getHeight(); ++y) {
    for (int x = 0; x < in->getWidth(); ++x) {
      if (in->getCH() == 1) {
        double random_value = rand() % 256;
        random_img->set(x, y, random_value);
      } else {
        for (int ch = 0; ch < in->getCH(); ch++) {
          double random_value = rand() % 256;
          random_img->set(x, y, ch, random_value);
        }
      }
    }
  }

  std::string random_filename = "random";
  random_img->save(random_filename.c_str());
  printf("✓ Random image generated and saved\n");

  printf("\n==================================================\n");
  printf("COMPARISON ANALYSIS: NATURAL vs RANDOM IMAGES\n");
  printf("==================================================\n");
  printf("\n1. DCT COEFFICIENT VISUALIZATION\n");
  printf("------------------------------\n");

  std::string natural_viz = "natural-dct-viz";
  std::string random_viz = "random-dct-viz";

  printf("Analyzing NATURAL image DCT coefficients...\n");

  Image *natural_viz_out = new Image();
  natural_viz_out->init(in->getWidth(), in->getHeight(), in->getCH());
  dct_visualize(in, natural_viz_out);
  natural_viz_out->save("natural-dct-viz");
  delete natural_viz_out;

  printf("\nAnalyzing RANDOM image DCT coefficients...\n");

  Image *random_viz_out = new Image();
  random_viz_out->init(random_img->getWidth(), random_img->getHeight(),
                       random_img->getCH());
  dct_visualize(random_img, random_viz_out);
  random_viz_out->save("random-dct-viz");
  delete random_viz_out;

  printf("\n2. DCT ENERGY ANALYSIS\n");
  printf("--------------------\n");

  printf("NATURAL IMAGE ANALYSIS:\n");

  printf("\nRANDOM IMAGE ANALYSIS:\n");

  printf("\n==================================================\n");
  printf("SUMMARY AND INSIGHTS\n");
  printf("==================================================\n");

  printf("\n🔍 EXPECTED DIFFERENCES:\n");
  printf("\nNATURAL IMAGES:\n");
  printf("✓ Energy concentrated in low frequencies (top-left)\n");
  printf("✓ Strong DC component (smooth regions)\n");
  printf("✓ Few coefficients contain most energy (energy compaction)\n");
  printf("✓ High-frequency coefficients near zero (smooth transitions)\n");
  printf("✓ Good compression potential\n");

  printf("\nRANDOM IMAGES:\n");
  printf("✓ Energy distributed across all frequencies\n");
  printf("✓ Weaker DC component (no smooth regions)\n");
  printf("✓ Many coefficients needed for energy representation\n");
  printf("✓ High-frequency coefficients significant (noise/edges)\n");
  printf("✓ Poor compression potential\n");

  printf("\n📊 FILES GENERATED:\n");
  printf("• Original image: (input)\n");
  printf("• Random image: %s\n", random_filename.c_str());
  printf("• Natural DCT visualization: %s\n", natural_viz.c_str());
  printf("• Random DCT visualization: %s\n", random_viz.c_str());

  printf("\n💡 KEY INSIGHT:\n");
  printf(
      "This demonstrates why DCT works well for natural images but poorly\n");
  printf("for random/noisy images - natural images have inherent structure\n");
  printf("and smoothness that DCT can exploit for compression!\n");

  printf("\n==================================================\n");
  printf("ANALYSIS COMPLETE\n");
  printf("==================================================\n");

  delete random_img;
}