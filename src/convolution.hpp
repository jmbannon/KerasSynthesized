#ifndef CONVOLUTION_HPP
#define CONVOLUTION_HPP

#include "HLS/hls.h"
#include "tensor3.hpp"
#include "tensor4.hpp"
#include "vector.h"
#include "common.hpp"
#include <stdio.h>
#include <math.h>


Numeric test_weights[2][3][3][3] = 
      {{{{0., 1., 2.},
         {3., 4., 5.},
         {6., 7., 8.}},

        {{0., 1., 2.},
         {3., 4., 5.},
         {6., 7., 8.}},

        {{0., 1., 2.},
         {3., 4., 5.},
         {6., 7., 8.}}},


       {{{0., 1., 2.},
         {3., 4., 5.},
         {6., 7., 8.}},

        {{0., 1., 2.},
         {3., 4., 5.},
         {6., 7., 8.}},

        {{0., 1., 2.},
         {3., 4., 5.},
         {6., 7., 8.}}}};

Numeric test_bias[2] =
      {0., 1.};

Numeric test_input[3][5][5] = 
      {{{ 0.,  1.,  2.,  3.,  4.},
        { 5.,  6.,  7.,  8.,  9.},
        {10., 11., 12., 13., 14.},
        {15., 16., 17., 18., 19.},
        {20., 21., 22., 23., 24.}},

       {{ 0.,  1.,  2.,  3.,  4.},
        { 5.,  6.,  7.,  8.,  9.},
        {10., 11., 12., 13., 14.},
        {15., 16., 17., 18., 19.},
        {20., 21., 22., 23., 24.}},

       {{ 0.,  1.,  2.,  3.,  4.},
        { 5.,  6.,  7.,  8.,  9.},
        {10., 11., 12., 13., 14.},
        {15., 16., 17., 18., 19.},
        {20., 21., 22., 23., 24.}}};

// inline void convolution_raw(Numeric *input, Numeric *output, Numeric *kernel, Numeric *bias, uint strideX, uint strideY) {
  
// }

inline void convolution(tensor3 *input, tensor3 *output, tensor4 *kernel, vector *bias, uint strideX, uint strideY) {
  int rowsX = INT_DIV_CEIL(input->rows - kernel->rows + 1, strideX);
  int colsY = INT_DIV_CEIL(input->cols - kernel->cols + 1, strideY);

  for (int c = 0; c < kernel->chans; c++) {
	  for (int i = 0; i < input->depth; i++) {
	  	for (int j = 0; j < rowsX; j++) {
  		  for (int k = 0; k < colsY; k++) {

  		  	if (i == 0) {
  		  		ROW3_MAJ_VAL(output, j, k, c) = V_VAL(bias, c);
  		  	}
  		  	// Kernel multiplication
  		  	for (int dx = 0; dx < kernel->rows; dx++) {
  		  		for (int dy = 0; dy < kernel->cols; dy++) {
  		  			// printf("%d %d | %d %d\n", j, k, j * strideX, j * strideY);
  		  			ROW3_MAJ_VAL(output, j, k, c) += ROW4_MAJ_VAL(kernel, dx, dy, i, c) * ROW3_MAJ_VAL(input, (j * strideX) + dx, (k * strideY) + dy, i);
  		  		}
  		  	}
  	  	}
  	  }
	  }
  }
}

void test_convolution2() {
	int res;
	tensor3 input;
	tensor4 weights;
	tensor3 output;
	vector bias;

	res = tensor3_init(&input, 5, 5, 3, ROW_MAJ);
	res = tensor4_init(&weights, 3, 3, 3, 2, ROW_MAJ);
	res = tensor3_init(&output, 3, 3, 2, ROW_MAJ);
	res = vector_init(&bias, 2);

	res = tensor3_set_data(&input, (Numeric *)test_input);
	res = tensor4_set_data(&weights, (Numeric *)test_weights);
	res = vector_set_data(&bias, (Numeric *)test_bias);
	convolution(&input, &output, &weights, &bias, 1, 1);

	tensor3_print(&output);
}

#endif