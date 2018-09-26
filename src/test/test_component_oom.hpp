#ifndef TEST_CONVOLUTION_OOM_HPP
#define TEST_CONVOLUTION_OOM_HPP

#include "HLS/hls.h"
#include <stdio.h>
#include <math.h>
#include "../tensor3.hpp"
#include "../tensor4.hpp"
#include "../common.hpp"
#include "../component_convolver.hpp"

int test_component_oom_args(uint input_rows,
                            uint input_cols,
                            uint input_depth,
                            uint input_tile_rows,
                            uint input_tile_cols,
                            uint input_tile_depth,
                            uint input_padding_rows,
                            uint input_padding_cols,
                            uint output_padding_rows,
                            uint output_padding_cols) {

  Numeric weights[3][3] = {
    { 0.0f, 1.0f, 2.0f },
    { 0.0f, 1.0f, 2.0f },
    { 0.0f, 1.0f, 2.0f }
  };

  uint kernel_len = 3;
  uint input_rows_p = input_rows + (2 * input_padding_cols);
  uint input_cols_p = input_cols + (2 * input_padding_cols);
  uint output_rows = input_rows_p - kernel_len + 1;
  uint output_cols = input_cols_p - kernel_len + 1;

  tiled_tensor3 input;
  tiled_tensor3_init_padding(&input, input_rows, input_cols, input_depth, input_tile_rows, input_tile_cols, input_tile_depth, ROW_MAJ, ROW_MAJ, input_padding_rows, input_padding_cols);
  tiled_tensor3_set_data_sequential_row_padding(&input, input_padding_rows, input_padding_cols);

  tiled_tensor3 output;
  tiled_tensor3_init_padding(&output, output_rows, output_cols, 1, input_tile_rows, input_tile_cols, input_tile_depth, ROW_MAJ, ROW_MAJ, output_padding_rows, output_padding_cols);
  tiled_tensor3_fill_zero(&output);

  mm_src mm_src_weights(weights, POW2(kernel_len) * sizeof(Numeric));
  mm_src mm_src_input(input.data, input.rows * input.cols * sizeof(Numeric));
  mm_src mm_src_output(output.data, output.rows * output.cols * sizeof(Numeric));

  return 0;
}

int test_component_oom_tiles_3_2__5_5() {
  int rows_t = 3;
  int cols_t = 2;
  int tile_rows = 5;
  int tile_cols = 5;

  int rows = tile_rows * rows_t;
  int cols = tile_cols * cols_t;
  int depth = 1;

  return test_component_oom_args(rows, cols, depth, tile_rows, tile_cols, depth, 0, 0, 0, 0);
}

int test_component_oom_5_5() {
  Numeric arr_weights[3][3] = {
    { 1.0f, 1.0f, 1.0f },
    { 2.0f, 2.0f, 2.0f },
    { 3.0f, 3.0f, 3.0f }
  };

  Numeric arr_input[5][5] = {
    { 0.0f, 0.0f, 0.0f, 1.0f, 1.0f },
    { 0.0f, 0.0f, 0.0f, 1.0f, 1.0f },
    { 0.0f, 0.0f, 0.0f, 1.0f, 1.0f },
    { 0.0f, 0.0f, 0.0f, 1.0f, 1.0f },
    { 0.0f, 0.0f, 0.0f, 1.0f, 1.0f }
  };

  Numeric arr_output[3][3] = {
    { 0.0f, 0.0f, 0.0f },
    { 0.0f, 0.0f, 0.0f },
    { 0.0f, 0.0f, 0.0f }
  };
  Numeric exp_output[3][3] = {
    { 0.0f, 6.0f, 12.0f },
    { 0.0f, 6.0f, 12.0f },
    { 0.0f, 6.0f, 12.0f }
  };

  mm_src mm_src_weights(arr_weights, 9 * sizeof(Numeric));
  mm_src mm_src_input(arr_input, 25 * sizeof(Numeric));
  mm_src mm_src_output(arr_output, 9 * sizeof(Numeric));

  tiled_tensor3 in, out;
  tiled_tensor3_init_dims(
    &in, 
    5, 5, 1,
    5, 5, 1,
    ROW_MAJ, ROW_MAJ);

  tiled_tensor3_init_dims(
    &out, 
    3, 3, 1,
    3, 3, 1,
    ROW_MAJ, ROW_MAJ);

  convolution9(mm_src_input, mm_src_output, mm_src_weights, in, out, 0, 0, 0);

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      // printf("%f %f\n", NUMERIC_VAL(exp_output[i][j]), NUMERIC_VAL(arr_output[i][j]));
      if (!fcompare(exp_output[i][j], arr_output[i][j])) {
        return 1;
      }
    }
  }
  return 0;
}



#endif