#ifndef TEST_CONVOLUTION2_HPP
#define TEST_CONVOLUTION2_HPP

#include "HLS/hls.h"
#include <stdio.h>
#include <math.h>
#include "../tensor3.hpp"
#include "../tensor4.hpp"
#include "../component_convolver.hpp"


int test_convolution() {
  Numeric arr_weights[3][3] = {
    { 0.0f, 1.0f, 2.0f },
    { 0.0f, 1.0f, 2.0f },
    { 0.0f, 1.0f, 2.0f }
  };

  tensor3 input;
  tensor3_init(&input, 256, 256, 3, ROW_MAJ);
  tensor3_set_data_sequential_row(&input);

  Numeric output[256][256] = { 0. };

  mm_src mm_src_weights(arr_weights, 9 * sizeof(Numeric));
  mm_src mm_src_input(input.data, 256 * 256 * sizeof(Numeric));
  mm_src mm_src_output((Numeric *)output, 256 * 256 * sizeof(Numeric));

  Numeric bram_fifo_in0[BUFFER_SIZE * 3];
  Numeric bram_fifo_out0[BUFFER_SIZE];

  convolution7(mm_src_input, mm_src_output, mm_src_weights, bram_fifo_in0, bram_fifo_out0, 0, 256, 256);

  bool pass = true;
  for (uint i = 0; i < 254; i++) {
  	for (uint j = 0; j < 254; j++) {
  		printf("%f ", NUMERIC_VAL(output[i][j]));
  	}
  	printf("\n");
  }
  
  if (pass) {
    printf("PASSED\n");
  }
  else {
    printf("FAILED\n");
  }

  return 0;

}



#endif