#ifndef TEST_COMPONENT_POOLING_HPP
#define TEST_COMPONENT_POOLING_HPP

#include "HLS/hls.h"
#include <stdio.h>
#include <math.h>
#include "../tensor3.hpp"
#include "../tensor4.hpp"
#include "../component_convolver.hpp"

int test_component_max_pooling_6_6() {

  Numeric arr_input[6][6] = {
    { 0.0f, 1.0f, 4.0f, 5.0f, 8.0f, 9.0f },
    { 2.0f, 3.0f, 6.0f, 7.0f, 10.0f, 11.0f },
    { 0.0f, 1.0f, 4.0f, 5.0f, 8.0f, 9.0f },
    { 2.0f, 3.0f, 6.0f, 7.0f, 10.0f, 11.0f },
    { 0.0f, 1.0f, 4.0f, 5.0f, 8.0f, 9.0f },
    { 2.0f, 3.0f, 6.0f, 7.0f, 10.0f, 11.0f },
  };

  Numeric arr_output[3][3] = {
    { 0.0f, 0.0f, 0.0f },
    { 0.0f, 0.0f, 0.0f },
    { 0.0f, 0.0f, 0.0f }
  };

  Numeric max_exp_output[3][3] = {
    { 3.0f, 7.0f, 11.0f },
    { 3.0f, 7.0f, 11.0f },
    { 3.0f, 7.0f, 11.0f }
  };

  Numeric input_bram_buffer[224];
  mm_src mm_src_output(arr_output, 9 * sizeof(Numeric));

  // Transfer arr_input into bram buffer of length 224
  int buffer_idx = 0;
  for (int i = 0; i < 6; ++i) {
    for (int j = 0; j < 6; ++j) {
      input_bram_buffer[buffer_idx++] = arr_input[i][j];
    }
  }

  pooling_max7(input_bram_buffer, mm_src_output, 6, 6);

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      if (!fcompare(max_exp_output[i][j], arr_output[i][j])) {
        return 1;
      }
    }
  }
  return 0;
}

int test_component_avg_pooling_6_6() {

  Numeric arr_input[6][6] = {
    { 0.0f, 1.0f, 4.0f, 5.0f, 8.0f, 9.0f },
    { 2.0f, 3.0f, 6.0f, 7.0f, 10.0f, 11.0f },
    { 0.0f, 1.0f, 4.0f, 5.0f, 8.0f, 9.0f },
    { 2.0f, 3.0f, 6.0f, 7.0f, 10.0f, 11.0f },
    { 0.0f, 1.0f, 4.0f, 5.0f, 8.0f, 9.0f },
    { 2.0f, 3.0f, 6.0f, 7.0f, 10.0f, 11.0f },
  };

  Numeric arr_output[3][3] = {
    { 0.0f, 0.0f, 0.0f },
    { 0.0f, 0.0f, 0.0f },
    { 0.0f, 0.0f, 0.0f }
  };

  Numeric avg_exp_output[3][3] = {
    { 1.5f, 5.5f, 9.5f },
    { 1.5f, 5.5f, 9.5f },
    { 1.5f, 5.5f, 9.5f }
  };

  Numeric input_bram_buffer[224];
  mm_src mm_src_output(arr_output, 9 * sizeof(Numeric));

  // Transfer arr_input into bram buffer of length 224
  int buffer_idx = 0;
  for (int i = 0; i < 6; ++i) {
    for (int j = 0; j < 6; ++j) {
      input_bram_buffer[buffer_idx++] = arr_input[i][j];
    }
  }

  pooling_avg7(input_bram_buffer, mm_src_output, 6, 6);

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      if (!fcompare(avg_exp_output[i][j], arr_output[i][j])) {
        return 1;
      }
    }
  }
  return 0;
}

// int test_component_3_3_convolver_variable() {
//   Numeric arr_weights[3][3] = {
//     { 0.0f, 1.0f, 2.0f },
//     { 0.0f, 1.0f, 2.0f },
//     { 0.0f, 1.0f, 2.0f }
//   };

//   tensor3 input;
//   tensor3_init(&input, CONVOLVER_TEST_INPUT_SIZE, CONVOLVER_TEST_INPUT_SIZE, 3, ROW_MAJ);
//   tensor3_set_data_sequential_row(&input);

//   Numeric output[CONVOLVER_TEST_INPUT_SIZE - 2][CONVOLVER_TEST_INPUT_SIZE - 2];
//   for (uint i = 0; i < (CONVOLVER_TEST_INPUT_SIZE - 2); ++i) {
//   	for (uint j = 0; j < (CONVOLVER_TEST_INPUT_SIZE - 2); ++j) {
//   		output[i][j] = 0.0f;
//   	}
//   }

//   mm_src mm_src_weights(arr_weights, 3 * 3 * sizeof(Numeric));
//   mm_src mm_src_input(input.data, CONVOLVER_TEST_INPUT_SIZE * CONVOLVER_TEST_INPUT_SIZE * 3 * sizeof(Numeric));
//   mm_src mm_src_output((Numeric *)output, (CONVOLVER_TEST_INPUT_SIZE - 2) * (CONVOLVER_TEST_INPUT_SIZE - 2) * sizeof(Numeric));

//   Numeric bram_fifo_in0[BUFFER_SIZE * 3];
//   Numeric bram_fifo_out0[BUFFER_SIZE];

//   convolution7(mm_src_input, mm_src_output, mm_src_weights, bram_fifo_in0, bram_fifo_out0, 0, CONVOLVER_TEST_INPUT_SIZE, CONVOLVER_TEST_INPUT_SIZE);

//   for (uint i = 0; i < (CONVOLVER_TEST_INPUT_SIZE - 2); i++) {
//   	for (uint j = 0; j < (CONVOLVER_TEST_INPUT_SIZE - 2); j++) {
//   		Numeric expected_value = ((j + 1) * 1 * 3) + ((j + 2) * 2 * 3);
//   		// printf("(%d, %d) %f %f\n", i, j, NUMERIC_VAL(output[i][j]), expected_value);
//   		if (!fcompare(NUMERIC_VAL(output[i][j]), expected_value)) {
//         	return 1;
//       	}
  		
//   	}
//   	// printf("\n");
//   }

//   return 0;

// }



#endif