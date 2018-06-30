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

  Numeric bram_fifo_in0[BUFFER_SIZE * 3];
  Numeric bram_fifo_out0[BUFFER_SIZE];

  convolution7(mm_src_input, mm_src_output, mm_src_weights, bram_fifo_in0, bram_fifo_out0, 0, 5, 5, 0, 0);

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

  Numeric arr_input[5][5] = {
    /* 0.0f      0.0f, 0.0f, 0.0f, 0.0f, 0.0f,      0.0f */
    /* 0.0f */ { 0.0f, 0.0f, 0.0f, 1.0f, 1.0f }, /* 0.0f */
    /* 0.0f */ { 0.0f, 0.0f, 0.0f, 1.0f, 1.0f }, /* 0.0f */
    /* 0.0f */ { 0.0f, 0.0f, 0.0f, 1.0f, 1.0f }, /* 0.0f */
    /* 0.0f */ { 0.0f, 0.0f, 0.0f, 1.0f, 1.0f }, /* 0.0f */
    /* 0.0f */ { 0.0f, 0.0f, 0.0f, 1.0f, 1.0f }  /* 0.0f */
    /* 0.0f      0.0f, 0.0f, 0.0f, 0.0f, 0.0f,      0.0f */
  };

  Numeric arr_output[5][5] = {
    { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },
    { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },
    { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },
    { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },
    { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f }
  };
  Numeric exp_output[5][5] = {
    { 0.0f, 0.0f, 5.0f, 10.0f, 10.0f },
    { 0.0f, 0.0f, 6.0f, 12.0f, 12.0f },
    { 0.0f, 0.0f, 6.0f, 12.0f, 12.0f },
    { 0.0f, 0.0f, 6.0f, 12.0f, 12.0f },
    { 0.0f, 0.0f, 3.0f, 6.0f, 6.0f }
  };

  mm_src mm_src_weights(arr_weights, 9 * sizeof(Numeric));
  mm_src mm_src_input(arr_input, 25 * sizeof(Numeric));
  mm_src mm_src_output(arr_output, 25 * sizeof(Numeric));

  Numeric bram_fifo_in0[BUFFER_SIZE * 3];
  Numeric bram_fifo_out0[BUFFER_SIZE];

  convolution7(mm_src_input, mm_src_output, mm_src_weights, bram_fifo_in0, bram_fifo_out0, 0, 5, 5, 1, 1);

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

#define CONVOLVER_KERNEL_LEN (3)
#define CONVOLVER_INPUT_LEN (CONVOLVER_INPUT_SIZE + (CONVOLVER_PADDING_SIZE * 2))
#define CONVOLVER_OUTPUT_LEN (CONVOLVER_INPUT_LEN - CONVOLVER_KERNEL_LEN + 1)

int test_component_3_3_convolver_variable() {

  Numeric arr_weights[3][3] = {
    { 0.0f, 1.0f, 2.0f },
    { 0.0f, 1.0f, 2.0f },
    { 0.0f, 1.0f, 2.0f }
  };

  tensor3 input;
  tensor3_init(&input, CONVOLVER_INPUT_SIZE, CONVOLVER_INPUT_SIZE, 3, ROW_MAJ);
  tensor3_set_data_sequential_row(&input);

  Numeric output[CONVOLVER_OUTPUT_LEN][CONVOLVER_OUTPUT_LEN];
  for (uint i = 0; i < CONVOLVER_OUTPUT_LEN; ++i) {
  	for (uint j = 0; j < CONVOLVER_OUTPUT_LEN; ++j) {
  		output[i][j] = 0.0f;
  	}
  }

  mm_src mm_src_weights(arr_weights, POW2(CONVOLVER_KERNEL_LEN) * sizeof(Numeric));
  mm_src mm_src_input(input.data, POW2(CONVOLVER_INPUT_SIZE) * sizeof(Numeric));
  mm_src mm_src_output((Numeric *)output, POW2(CONVOLVER_OUTPUT_LEN) * sizeof(Numeric));

  Numeric bram_fifo_in0[BUFFER_SIZE * 3];
  Numeric bram_fifo_out0[BUFFER_SIZE];

  convolution7(
    mm_src_input,    // input
    mm_src_output,   // output
    mm_src_weights,  // weights
    bram_fifo_in0,   // input buffer
    bram_fifo_out0,  // output buffer
    0,               // weight offset
    CONVOLVER_INPUT_SIZE, CONVOLVER_INPUT_SIZE,       // input size
    CONVOLVER_PADDING_SIZE, CONVOLVER_PADDING_SIZE);  // padding

  for (uint i = 0; i < CONVOLVER_OUTPUT_LEN; i++) {
  	for (uint j = 0; j < CONVOLVER_OUTPUT_LEN; j++) {
      uint kernel_rows_non_padding = CONVOLVER_KERNEL_LEN;
      Numeric expected_value = 0;

      // Kernel has row(s) within padding
      if (i < CONVOLVER_PADDING_SIZE) {
        // top portion of padding
        kernel_rows_non_padding = CONVOLVER_KERNEL_LEN - (CONVOLVER_PADDING_SIZE - i);
      } else if (i >= CONVOLVER_OUTPUT_LEN - CONVOLVER_PADDING_SIZE) {
        // bottom portion of padding
        kernel_rows_non_padding = (CONVOLVER_OUTPUT_LEN - i);
      }

      for (uint k = 0; k < CONVOLVER_KERNEL_LEN; ++k) {
        uint input_col = j + k;

        // Kernel column is NOT within padding
        if (!(input_col < CONVOLVER_PADDING_SIZE || input_col >= CONVOLVER_INPUT_SIZE + CONVOLVER_PADDING_SIZE)) {
                            // col of original input  // multiply by kernel value // multiply by number of kernel values not in padding
          expected_value += (input_col - CONVOLVER_PADDING_SIZE) * k * kernel_rows_non_padding;
        }
      }

      // printf("%lf ", expected_value);
      printf("%lf ", NUMERIC_VAL(output[i][j]));
  		// if (!fcompare(NUMERIC_VAL(output[i][j]), expected_value)) {
    //     return 1;
    //   }
  		
  	}
  	printf("\n");
  }

  return 0;

}



#endif