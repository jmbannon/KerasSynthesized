#ifndef TEST_CONVOLUTION2_HPP
#define TEST_CONVOLUTION2_HPP

#include "HLS/hls.h"
#include <stdio.h>
#include <math.h>
#include "../tensor3.hpp"
#include "../tensor4.hpp"
#include "../convolution.hpp"


int test_convolution() {
  Numeric arr_weights[3][3] = {
    { 0.0f, 1.0f, 2.0f },
    { 0.0f, 1.0f, 2.0f },
    { 0.0f, 1.0f, 2.0f }
  };

  tensor3 input;
  tensor3_init(&input, 256, 256, 3, ROW_MAJ);
  tensor3_set_data_sequential_row(&input);

  Numeric output[254][254];

  convolution6(mm_src_input, 0, mm_src_output, 0, mm_src_weights, 0, 5, 5);

  bool pass = true;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      if (!fcompare(exp_output[i][j], arr_output[i][j])) {
        return 1;
      }
    }
  }
  return 0;

}


#endif