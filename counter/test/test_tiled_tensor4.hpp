#ifndef TEST_TILED_TENSOR4_HPP
#define TEST_TILED_TENSOR4_HPP

#include "HLS/hls.h"
#include <stdio.h>
#include <math.h>
#include "../tiled_tensor4.hpp"
#include "../convolution.hpp"

int test_tiled_tensor4(tiled_tensor4 *t, Numeric *expected_linear) {
  int ret;
  Numeric data[t->vol];

  for (int i = 0; i < t->vol; i++) {
    data[i] = i;
  }

  tiled_tensor4_set_data(t, data);
  // tiled_tensor4_print(t);

  for (uint i = 0; i < t->vol; i++) {
    if (t->data[i] != expected_linear[i]) {
      return 1;
    }
  }

  for (uint i = 0; i < t->vol; i++) {
    if (t->data[i] != expected_linear[i]) {
      return 1;
    }
  }

  int idx = 0;
  for (uint c = 0; c < t->chans; c++) {
  	for (uint i = 0; i < t->depth; i++) {
  	  for (uint j = 0; j < t->rows; j++) {
  	    for (uint k = 0; k < t->cols; k++) {
  	      if (tiled_tensor4_val(t, j, k, i, c) != idx++) {
  	        return 1;
  	      }
  	    }
      }
    }
  }

  return 0;
}

int test_tiled_tensor4_row_row() {
  Numeric expected_linear[96] = { 0., 1., 4., 5., 8., 9., 24., 25., 28., 29., 32., 33., 2., 3., 6., 7., 10., 11., 26., 27., 30., 31., 34., 35., 12., 13., 16., 17., 20., 21., 36., 37., 40., 41., 44., 45., 14., 15., 18., 19., 22., 23., 38., 39., 42., 43., 46., 47., 48., 49., 52., 53., 56., 57., 72., 73., 76., 77., 80., 81., 50., 51., 54., 55., 58., 59., 74., 75., 78., 79., 82., 83., 60., 61., 64., 65., 68., 69., 84., 85., 88., 89., 92., 93., 62., 63., 66., 67., 70., 71., 86., 87., 90., 91., 94., 95. };
  tiled_tensor4 t;

  tiled_tensor4_init(&t, 6, 4, 4, 4, 3, 2, 2, 2, ROW_MAJ, ROW_MAJ);
  return test_tiled_tensor4(&t, expected_linear);
}

int test_tiled_tensor4_dep_dep() {
  Numeric expected_linear[96] = { 0., 4., 8., 1., 5., 9., 24., 28., 32., 25., 29., 33., 12., 16., 20., 13., 17., 21., 36., 40., 44., 37., 41., 45., 2., 6., 10., 3., 7., 11., 26., 30., 34., 27., 31., 35., 14., 18., 22., 15., 19., 23., 38., 42., 46., 39., 43., 47., 48., 52., 56., 49., 53., 57., 72., 76., 80., 73., 77., 81., 60., 64., 68., 61., 65., 69., 84., 88., 92., 85., 89., 93., 50., 54., 58., 51., 55., 59., 74., 78., 82., 75., 79., 83., 62., 66., 70., 63., 67., 71., 86., 90., 94., 87., 91., 95. };
  tiled_tensor4 t;

  tiled_tensor4_init(&t, 6, 4, 4, 4, 3, 2, 2, 2, DEP_MAJ, DEP_MAJ);
  return test_tiled_tensor4(&t, expected_linear);
}

int test_tiled_tensor4_chn_chn() {
  Numeric expected_linear[96] = { 0., 24., 1., 25., 4., 28., 5., 29., 8., 32., 9., 33., 48., 72., 49., 73., 52., 76., 53., 77., 56., 80., 57., 81., 2., 26., 3., 27., 6., 30., 7., 31., 10., 34., 11., 35., 50., 74., 51., 75., 54., 78., 55., 79., 58., 82., 59., 83., 12., 36., 13., 37., 16., 40., 17., 41., 20., 44., 21., 45., 60., 84., 61., 85., 64., 88., 65., 89., 68., 92., 69., 93., 14., 38., 15., 39., 18., 42., 19., 43., 22., 46., 23., 47., 62., 86., 63., 87., 66., 90., 67., 91., 70., 94., 71., 95. };
  tiled_tensor4 t;

  tiled_tensor4_init(&t, 6, 4, 4, 4, 3, 2, 2, 2, CHN_MAJ, CHN_MAJ);
  return test_tiled_tensor4(&t, expected_linear);
}

#endif