#ifndef TEST_CONVOLUTION2_HPP
#define TEST_CONVOLUTION2_HPP

#include "HLS/hls.h"
#include <stdio.h>
#include <math.h>
#include "../tensor3.hpp"
#include "../tensor4.hpp"
#include "../component_convolver.hpp"

int test_component_convolver_5_5() {
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

  Numeric bram_fifo_in0[BUFFER_SIZE];
  Numeric bram_fifo_in1[BUFFER_SIZE];
  Numeric bram_fifo_in2[BUFFER_SIZE];
  Numeric bram_fifo_out0[BUFFER_SIZE];

  convolution8(mm_src_input, mm_src_output, mm_src_weights, bram_fifo_in0, bram_fifo_in1, bram_fifo_in2, bram_fifo_out0, 0, 5, 5, 0, 0);

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

int test_component_convolver_5_5_padding_1_1() {
  Numeric arr_weights[3][3] = {
    { 1.0f, 1.0f, 1.0f },
    { 2.0f, 2.0f, 2.0f },
    { 3.0f, 3.0f, 3.0f }
  };

  Numeric arr_input[7][7] = {
    { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
    { 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f},
    { 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f},
    { 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f},
    { 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f},
    { 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f},
    { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}
  };

  Numeric arr_output[7][9] = {
    { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },
    { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },
    { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },
    { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },
    { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },
    { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },
    { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f }
  };
  Numeric exp_output[7][9] = {
    { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },
    { 0.0f, 0.0f, 0.0f, 0.0f, 5.0f, 10.0f, 10.0f, 0.0f, 0.0f },
    { 0.0f, 0.0f, 0.0f, 0.0f, 6.0f, 12.0f, 12.0f, 0.0f, 0.0f },
    { 0.0f, 0.0f, 0.0f, 0.0f, 6.0f, 12.0f, 12.0f, 0.0f, 0.0f },
    { 0.0f, 0.0f, 0.0f, 0.0f, 6.0f, 12.0f, 12.0f, 0.0f, 0.0f },
    { 0.0f, 0.0f, 0.0f, 0.0f, 3.0f, 6.0f, 6.0f, 0.0f, 0.0f },
    { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f }
  };

  mm_src mm_src_weights(arr_weights, 9 * sizeof(Numeric));
  mm_src mm_src_input(arr_input, 7 * 7 * sizeof(Numeric));
  mm_src mm_src_output(arr_output, 7 * 9 * sizeof(Numeric));

  Numeric bram_fifo_in0[BUFFER_SIZE];
  Numeric bram_fifo_in1[BUFFER_SIZE];
  Numeric bram_fifo_in2[BUFFER_SIZE];
  Numeric bram_fifo_out0[BUFFER_SIZE];

  convolution8(mm_src_input, mm_src_output, mm_src_weights, bram_fifo_in0, bram_fifo_in1, bram_fifo_in2, bram_fifo_out0, 0, 7, 7, 1, 2);

  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 5; ++j) {
      // printf("%f %f\n", NUMERIC_VAL(exp_output[i][j]), NUMERIC_VAL(arr_output[i][j]));
      if (!fcompare(exp_output[i][j], arr_output[i][j])) {
        return 1;
      }
    }
  }
  return 0;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////

int test_component_3_3_convolver_args(uint input_rows,
                                      uint input_cols,
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

  tensor3 input;
  tensor3_init_padding(&input, input_rows, input_cols, 3, ROW_MAJ, input_padding_rows, input_padding_cols);
  tensor3_set_data_sequential_row_padding(&input, input_padding_rows, input_padding_cols);

  tensor3 output;
  tensor3_init_padding(&output, output_rows, output_cols, 1, ROW_MAJ, output_padding_rows, output_padding_cols);
  tensor3_fill_zero(&output);

  mm_src mm_src_weights(weights, POW2(kernel_len) * sizeof(Numeric));
  mm_src mm_src_input(input.data, input.rows * input.cols * sizeof(Numeric));
  mm_src mm_src_output(output.data, output.rows * output.cols * sizeof(Numeric));

  Numeric bram_fifo_in0[BUFFER_SIZE];
  Numeric bram_fifo_in1[BUFFER_SIZE];
  Numeric bram_fifo_in2[BUFFER_SIZE];
  Numeric bram_fifo_out0[BUFFER_SIZE];

  convolution8(
    mm_src_input,    // input
    mm_src_output,   // output
    mm_src_weights,  // weights
    bram_fifo_in0, bram_fifo_in1, bram_fifo_in2,  // input buffers
    bram_fifo_out0,  // output buffer
    0,               // weight offset
    input.rows, input.cols,       // input size
    output_padding_rows, output_padding_cols);  // padding

  // tensor3_print(&output);

  // Checks convolution values
  for (uint i = 0; i < output_rows; i++) {
    for (uint j = 0; j < output_cols; j++) {
      uint kernel_rows_non_padding = kernel_len;

      Numeric value = NUMERIC_VAL(tensor3_val(&output, i + output_padding_rows, j + output_padding_cols, 0));
      Numeric expected_value = 0;

      // Kernel has row(s) within padding
      if (i < input_padding_rows) {
        // top portion of padding
        kernel_rows_non_padding = kernel_len - (input_padding_rows - i);
      } else if (i >= output_rows - input_padding_rows) {
        // bottom portion of padding
        kernel_rows_non_padding = (input_rows - (i - input_padding_rows));
      }

      for (uint k = 0; k < kernel_len; ++k) {
        uint input_col = j + k;
        // Kernel column is NOT within padding
        if (!(input_col < input_padding_cols || input_col >= input_cols + input_padding_cols)) {
                            // col of original input  // multiply by kernel value // multiply by number of kernel values not in padding
          expected_value += (input_col - input_padding_cols) * k * kernel_rows_non_padding;
        }
      }

      // printf("%lf ", expected_value);
      // printf("%lf ", NUMERIC_VAL(value));
      if (!fcompare(value, expected_value)) {
        return 1;
      }
      
    }
    // printf("\n");
  }

  // CHecks padding
  for (uint i = 0; i < output.rows; i++) {
    for (uint j = 0; j < output.cols; j++) {

      Numeric value = NUMERIC_VAL(tensor3_val(&output, i, j, 0));
      Numeric expected_value = 0;

      if (i < output_padding_rows || i >= output.rows - output_padding_rows) {
        if (!fcompare(value, expected_value)) {
          return 1;
        }
      }

      if (j < output_padding_cols || j >= output.cols - output_padding_cols) {
        if (!fcompare(value, expected_value)) {
          return 1;
        }
      }
    }
  }

  return 0;
}

int test_component_3_3_convolver_variable() {
  return test_component_3_3_convolver_args(CONVOLVER_INPUT_SIZE, CONVOLVER_INPUT_SIZE, 0, 0, CONVOLVER_PADDING_SIZE, CONVOLVER_PADDING_SIZE);
}

int test_component_convolver_3_3_on_224_224_padding_1_1() {
  return test_component_3_3_convolver_args(224, 224, 1, 1, 1, 1);
}


#endif