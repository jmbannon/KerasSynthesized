#ifndef TEST_TILED_TENSOR3_HPP
#define TEST_TILED_TENSOR3_HPP

#include "HLS/hls.h"
#include <stdio.h>
#include <math.h>
#include "../tensor3.hpp"
#include "../tiled_tensor3.hpp"

int test_tiled_tensor3(tiled_tensor3 *t, Numeric *expected_linear) {
  int ret;
  Numeric data[t->size];

  for (int i = 0; i < t->size; i++) {
    data[i] = i;
  }

  tiled_tensor3_set_data(t, data);
  // for (uint i = 0; i < t->size; i++) {
  //   printf("%d %f\n", i, t->data[i]);
  // }
  // tiled_tensor3_print(t);

  for (uint i = 0; i < t->vol; i++) {
    // printf("%d | %f %f\n", i, t->data[i], expected_linear[i]);
    if (t->data[i] != expected_linear[i]) {
      return 1;
    }
  }

  int idx = 0;
  for (uint i = 0; i < t->depth; i++) {
    for (uint j = 0; j < t->rows; j++) {
      for (uint k = 0; k < t->cols; k++) {
        if (tiled_tensor3_val(t, j, k, i) != idx++) {
          return 1;
        }
      }
    }
  }

  return 0;
}

int test_tiled_tensor3_row_row() {
  Numeric expected_linear[96] = { 0., 1., 4., 5., 8., 9., 24., 25., 28., 29., 32., 33., 2., 3., 6., 7., 10., 11., 26., 27., 30., 31., 34., 35., 12., 13., 16., 17., 20., 21., 36., 37., 40., 41., 44., 45., 14., 15., 18., 19., 22., 23., 38., 39., 42., 43., 46., 47., 48., 49., 52., 53., 56., 57., 72., 73., 76., 77., 80., 81., 50., 51., 54., 55., 58., 59., 74., 75., 78., 79., 82., 83., 60., 61., 64., 65., 68., 69., 84., 85., 88., 89., 92., 93., 62., 63., 66., 67., 70., 71., 86., 87., 90., 91., 94., 95. };
  tiled_tensor3 t;

  tiled_tensor3_init(&t, 6, 4, 4, 3, 2, 2, ROW_MAJ, ROW_MAJ);
  return test_tiled_tensor3(&t, expected_linear);
}

int test_tiled_tensor3_col_col() {
  Numeric expected_linear[96] = { 0., 4., 8., 1., 5., 9., 24., 28., 32., 25., 29., 33., 12., 16., 20., 13., 17., 21., 36., 40., 44., 37., 41., 45., 2., 6., 10., 3., 7., 11., 26., 30., 34., 27., 31., 35., 14., 18., 22., 15., 19., 23., 38., 42., 46., 39., 43., 47., 48., 52., 56., 49., 53., 57., 72., 76., 80., 73., 77., 81., 60., 64., 68., 61., 65., 69., 84., 88., 92., 85., 89., 93., 50., 54., 58., 51., 55., 59., 74., 78., 82., 75., 79., 83., 62., 66., 70., 63., 67., 71., 86., 90., 94., 87., 91., 95. };
  tiled_tensor3 t;

  tiled_tensor3_init(&t, 6, 4, 4, 3, 2, 2, COL_MAJ, COL_MAJ);
  return test_tiled_tensor3(&t, expected_linear);
}

int test_tiled_tensor3_dep_dep() {
  Numeric expected_linear[24] = { 0., 8., 16., 1., 9., 17., 2., 10., 18., 3., 11., 19., 4., 12., 20., 5., 13., 21., 6., 14., 22., 7., 15., 23. };
  tiled_tensor3 t;

  tiled_tensor3_init(&t, 8, 4, 4, 4, 4, 2, DEP_MAJ, DEP_MAJ);
  return test_tiled_tensor3(&t, expected_linear);
}

#endif