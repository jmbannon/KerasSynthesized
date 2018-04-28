#ifndef TEST_TENSOR4_HPP
#define TEST_TENSOR4_HPP

#include "HLS/hls.h"
#include <stdio.h>
#include <math.h>
#include "../tensor3.hpp"
#include "../tensor4.hpp"
#include "../convolution.hpp"

int test_tensor4(tensor4 *t, Numeric *expected_linear) {
  int ret;
  Numeric data[t->vol];

  for (int i = 0; i < t->vol; i++) {
    data[i] = i;
  }

  tensor4_set_data(t, data);

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
  	      if (tensor4_val(t, j, k, i, c) != idx++) {
  	        return 1;
  	      }
  	    }
      }
    }
  }

  return 0;
}

int test_tensor4_row() {
  Numeric expected_linear[48] = { 0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24., 25., 26., 27., 28., 29., 30., 31., 32., 33., 34., 35., 36., 37., 38., 39., 40., 41., 42., 43., 44., 45., 46., 47. };
  tensor4 t;

  tensor4_init(&t, 4, 2, 3, 2, ROW_MAJ);
  return test_tensor4(&t, expected_linear);
}

int test_tensor4_dep() {
  Numeric expected_linear[48] = { 0., 8., 16., 24., 32., 40., 1., 9., 17., 25., 33., 41., 2., 10., 18., 26., 34., 42., 3., 11., 19., 27., 35., 43., 4., 12., 20., 28., 36., 44., 5., 13., 21., 29., 37., 45., 6., 14., 22., 30., 38., 46., 7., 15., 23., 31., 39., 47. };
  tensor4 t;

  tensor4_init(&t, 4, 2, 3, 2, DEP_MAJ);
  return test_tensor4(&t, expected_linear);
}

int test_tensor4_chn() {
  Numeric expected_linear[48] = { 0., 24., 8., 32., 16., 40., 1., 25., 9., 33., 17., 41., 2., 26., 10., 34., 18., 42., 3., 27., 11., 35., 19., 43., 4., 28., 12., 36., 20., 44., 5., 29., 13., 37., 21., 45., 6., 30., 14., 38., 22., 46., 7., 31., 15., 39., 23., 47. };
  tensor4 t;

  tensor4_init(&t, 4, 2, 3, 2, CHN_MAJ);
  return test_tensor4(&t, expected_linear);
}

#endif