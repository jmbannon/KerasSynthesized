#include "HLS/hls.h"
#include "tensor3.hpp"
#include <stdio.h>
#include <math.h>

inline void convolution(tensor3 *input, tensor3 *output, tensor3 *kernel, uint strideX, uint strideY) {
  for (int i = 0; i < input->depth; i++) {
  	for (int j = 0; j < input->rows; j++) {
	  for (int k = 0; k < input->cols; k++) {

	  	for (int dx = 0; dx < kernel->rows; dx++) {
	  		for (int dy = 0; dy < kernel->cols; dy++) {

	  			output[FIX] += kernel[dx][dy][i] * input[j + dx][k + dy][i]


	  		}
	  	}

  	  }
  	}
  }
}

