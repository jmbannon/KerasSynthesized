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

  Numeric arr_output[3][3] = { 0.0f };

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

  convolution7(mm_src_input, mm_src_output, mm_src_weights, bram_fifo_in0, bram_fifo_out0, 0, 5, 5);

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      // printf("%f %f\n", NUMERIC_VAL(exp_output[i][j]), NUMERIC_VAL(arr_output[i][j]));
      if (!fcompare(exp_output[i][j], arr_output[i][j])) {
        return 1;
      }
      //printf("%f ", output_stream.read());
    }
    //printf("\n");
  }
  return 0;
}

int test_component_convolver_256_256() {
  Numeric arr_weights[3][3] = {
    { 0.0f, 1.0f, 2.0f },
    { 0.0f, 1.0f, 2.0f },
    { 0.0f, 1.0f, 2.0f }
  };

  tensor3 input;
  tensor3_init(&input, 64, 64, 3, ROW_MAJ);
  tensor3_set_data_sequential_row(&input);

  Numeric output[62][62] = { 0. };

  mm_src mm_src_weights(arr_weights, 9 * sizeof(Numeric));
  mm_src mm_src_input(input.data, 64 * 64 * 3 * sizeof(Numeric));
  mm_src mm_src_output((Numeric *)output, 62 * 62 * sizeof(Numeric));

  Numeric bram_fifo_in0[BUFFER_SIZE * 3];
  Numeric bram_fifo_out0[BUFFER_SIZE];

  convolution7(mm_src_input, mm_src_output, mm_src_weights, bram_fifo_in0, bram_fifo_out0, 0, 64, 64);

  for (uint i = 0; i < 62; i++) {
  	for (uint j = 0; j < 62; j++) {
  		Numeric expected_value = ((j + 1) * 1 * 3) + ((j + 2) * 2 * 3);
  		// printf("(%d, %d) %f %f\n", i, j, NUMERIC_VAL(output[i][j]), expected_value);
  		if (!fcompare(NUMERIC_VAL(output[i][j]), expected_value)) {
        	return 1;
      	}
  		
  	}
  	// printf("\n");
  }

  return 0;

}



#endif