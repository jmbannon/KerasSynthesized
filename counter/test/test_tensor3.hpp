#ifndef TEST_TENSOR3_HPP
#define TEST_TENSOR3_HPP

#include "HLS/hls.h"
#include <stdio.h>
#include <math.h>
#include "../tensor3.hpp"
#include "../tensor4.hpp"
#include "../convolution.hpp"

int test_tensor3(tensor3 *t, Numeric *expected_linear) {
  int ret;
  Numeric data[t->vol];

  for (int i = 0; i < t->vol; i++) {
    data[i] = i;
  }

  tensor3_set_data(t, data);

  for (uint i = 0; i < t->vol; i++) {
    if (t->data[i] != expected_linear[i]) {
      return 1;
    }
  }

  int idx = 0;
  for (uint i = 0; i < t->depth; i++) {
    for (uint j = 0; j < t->rows; j++) {
      for (uint k = 0; k < t->cols; k++) {
        if (tensor3_val(t, j, k, i) != idx++) {
          return 1;
        }
      }
    }
  }

  return 0;
}

int test_tensor3_row() {
  Numeric expected_linear[24] = { 0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23. };
  tensor3 t;

  tensor3_init(&t, 4, 2, 3, ROW_MAJ);
  return test_tensor3(&t, expected_linear);
}

int test_tensor3_col() {
  Numeric expected_linear[24] = { 0., 2., 4., 6., 1., 3., 5., 7., 8., 10., 12., 14., 9., 11., 13., 15., 16., 18., 20., 22., 17., 19., 21., 23. };
  tensor3 t;

  tensor3_init(&t, 4, 2, 3, COL_MAJ);
  return test_tensor3(&t, expected_linear);
}

int test_tensor3_dep() {
  Numeric expected_linear[24] = { 0., 8., 16., 1., 9., 17., 2., 10., 18., 3., 11., 19., 4., 12., 20., 5., 13., 21., 6., 14., 22., 7., 15., 23. };
  tensor3 t;

  tensor3_init(&t, 4, 2, 3, DEP_MAJ);
  return test_tensor3(&t, expected_linear);
}

#endif