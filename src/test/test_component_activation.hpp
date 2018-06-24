#ifndef TEST_COMPONENT_ACTIVATION_HPP
#define TEST_COMPONENT_ACTIVATION_HPP

#include "HLS/hls.h"
#include <stdio.h>
#include <math.h>
#include "../tensor3.hpp"
#include "../tensor4.hpp"
#include "../component_convolver.hpp"

int test_component_relu_6_6() {

  Numeric arr_input[6][6] = {
    { 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f },
    { 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f },
    { 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f },
    { 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f },
    { 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f },
    { 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f }
  };

  Numeric arr_output[6][6] = {
    { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },
    { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },
    { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },
    { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },
    { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },
    { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f }
  };

  Numeric exp_output[6][6] = {
    { 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f },
    { 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f },
    { 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f },
    { 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f },
    { 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f },
    { 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f }
  };

  Numeric input_bram_buffer[224];
  mm_src mm_src_output(arr_output, 6 * 6 * sizeof(Numeric));

  // Transfer arr_input into bram buffer of length 224
  int buffer_idx = 0;
  for (int i = 0; i < 6; ++i) {
    for (int j = 0; j < 6; ++j) {
      input_bram_buffer[buffer_idx++] = arr_input[i][j];
    }
  }

  activation7(input_bram_buffer, mm_src_output, 6, 6, true);

  for (int i = 0; i < 6; ++i) {
    for (int j = 0; j < 6; ++j) {
      if (!fcompare(exp_output[i][j], arr_output[i][j])) {
        return 1;
      }
    }
  }
  return 0;
}

int test_component_bn_relu_6_6() {
  Numeric gamma = -2.0f;
  Numeric beta = 3.0f;

  Numeric arr_input[6][6] = {
    { 2.0f, 1.0f, 0.0f, -1.0f, -2.0f, -3.0f },
    { 2.0f, 1.0f, 0.0f, -1.0f, -2.0f, -3.0f },
    { 2.0f, 1.0f, 0.0f, -1.0f, -2.0f, -3.0f },
    { 2.0f, 1.0f, 0.0f, -1.0f, -2.0f, -3.0f },
    { 2.0f, 1.0f, 0.0f, -1.0f, -2.0f, -3.0f },
    { 2.0f, 1.0f, 0.0f, -1.0f, -2.0f, -3.0f }
  };

  Numeric arr_output[6][6] = {
    { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },
    { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },
    { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },
    { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },
    { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },
    { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f }
  };

  Numeric exp_output[6][6] = {
    { 0.0f, 1.0f, 3.0f, 5.0f, 7.0f, 9.0f },
    { 0.0f, 1.0f, 3.0f, 5.0f, 7.0f, 9.0f },
    { 0.0f, 1.0f, 3.0f, 5.0f, 7.0f, 9.0f },
    { 0.0f, 1.0f, 3.0f, 5.0f, 7.0f, 9.0f },
    { 0.0f, 1.0f, 3.0f, 5.0f, 7.0f, 9.0f },
    { 0.0f, 1.0f, 3.0f, 5.0f, 7.0f, 9.0f }
  };

  Numeric input_bram_buffer[224];
  mm_src mm_src_output(arr_output, 6 * 6 * sizeof(Numeric));

  // Transfer arr_input into bram buffer of length 224
  int buffer_idx = 0;
  for (int i = 0; i < 6; ++i) {
    for (int j = 0; j < 6; ++j) {
      input_bram_buffer[buffer_idx++] = arr_input[i][j];
    }
  }

  bn_activation7(input_bram_buffer, mm_src_output, 6, 6, gamma, beta, true);

  for (int i = 0; i < 6; ++i) {
    for (int j = 0; j < 6; ++j) {
      if (!fcompare(exp_output[i][j], arr_output[i][j])) {
        return 1;
      }
    }
  }
  return 0;
}

#endif