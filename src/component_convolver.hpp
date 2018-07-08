#ifndef COMPONENT_CONVOLVER_HPP
#define COMPONENT_CONVOLVER_HPP

#include "HLS/hls.h"
#include "HLS/ac_int.h"
#include "HLS/ac_fixed.h"
#include "HLS/ac_fixed_math.h"

#include "tensor3.hpp"
#include "tensor4.hpp"
#include "common.hpp"

using namespace ihc;


component
void convolution8(mm_src & restrict input,
                  mm_src & restrict output,
                  hls_avalon_slave_memory_argument(3*3*sizeof(Numeric)) Numeric * lweights,
                  hls_avalon_slave_memory_argument(BUFFER_SIZE*3*sizeof(Numeric)) Numeric * restrict bram_fifo,
                  hls_avalon_slave_memory_argument(BUFFER_SIZE*sizeof(Numeric)) Numeric * restrict bram_fifo_out0,
                  hls_avalon_slave_register_argument uint16 weight_offset,
                  hls_avalon_slave_register_argument uint16 rows,
                  hls_avalon_slave_register_argument uint16 cols,
                  hls_avalon_slave_register_argument uint16 buffer_cols,
                  hls_avalon_slave_register_argument uint3 paddingY,
                  hls_avalon_slave_register_argument uint3 paddingX) {
  #pragma max_concurrency 1
  for (uint16 m = 0; m < rows - 2; ++m) {
    #pragma max_concurrency 1
    for (uint16 batch_offset = 0; batch_offset < cols; batch_offset += (BUFFER_SIZE - 3)) {
      hls_register const uint16 output_offset = ((m + paddingY) * (buffer_cols)) + batch_offset + paddingX;

      // Loads data into registers and local storage
      #pragma ivdep
      #pragma loop_coalesce 2
      #pragma unroll 1
      #pragma max_concurrency 1
      for (uint3 ii = 0; ii < 3; ++ii) {
        hls_register const uint16 input_offset = (buffer_cols * (m + ii)) + batch_offset;
        hls_register const uint16 fifo_offset = (ii * BUFFER_SIZE);

        #pragma ivdep
        #pragma unroll 1
        for (uint8 j = 0; j < BUFFER_SIZE; ++j) {
          bram_fifo[fifo_offset + j] = input[input_offset + j];
        }
      }

      #pragma ivdep
      #pragma unroll 1
      for (uint8 j = 0; j < BUFFER_SIZE - 2; ++j) {
        bram_fifo_out0[j] = output[output_offset + j];
      }

      // Convolve on entire buffer
      #pragma unroll 1
      #pragma max_concurrency 1
      #pragma ivdep
      for (uint8 n = 0; n < BUFFER_SIZE - 3; ++n) {
        // Convolution
        #pragma unroll
        for (uint2 j = 0; j < 3; ++j) {
          bram_fifo_out0[n] += bram_fifo[(0 * BUFFER_SIZE) + n + j] * lweights[(0 * 3) + j];
          bram_fifo_out0[n] += bram_fifo[(1 * BUFFER_SIZE) + n + j] * lweights[(1 * 3) + j];
          bram_fifo_out0[n] += bram_fifo[(2 * BUFFER_SIZE) + n + j] * lweights[(2 * 3) + j];
        }
      }

      #pragma ivdep
      #pragma unroll 1
      for (uint8 n = 0; n < BUFFER_SIZE - 2; ++n) {
        if (batch_offset + n < cols - 2) {
          output[output_offset + n] = bram_fifo_out0[n];
        } else {
          output[output_offset + n] = 0.0;
        }
        
      }
    }
  }
}

component
void activation7(hls_avalon_slave_memory_argument(224*sizeof(Numeric)) Numeric * restrict input,
                mm_src & restrict output,
                const uint16 rows,
                const uint16 cols,
                const bool write_to_output) {
  for (uint6 i = 0; i < rows; ++i) {
    for (uint6 j = 0; j < cols; ++j) {
      Numeric zero = 0.0;
      input[(i * cols) + j] = MAX(input[(i * cols) + j], zero);
    }
  }

  if (write_to_output) {
    for (uint6 i = 0; i < rows; ++i) {
      for (uint6 j = 0; j < cols; ++j) {
        output[(i * cols) + j] = input[(i * cols) + j];
      }
    }
  }
}

component
void bn_activation7(hls_avalon_slave_memory_argument(224*sizeof(Numeric)) Numeric * restrict input,
                    mm_src & restrict output,
                    const uint16 rows,
                    const uint16 cols,
                    const Numeric gamma,
                    const Numeric beta,
                    const bool write_to_output) {
  for (uint6 i = 0; i < rows; ++i) {
    for (uint6 j = 0; j < cols; ++j) {
      Numeric value = (input[(i * cols) + j] * gamma) + beta;
      Numeric zero = 0.0;
      input[(i * cols) + j] = MAX(value, zero);
    }
  }

  if (write_to_output) {
    for (uint6 i = 0; i < rows; ++i) {
      for (uint6 j = 0; j < cols; ++j) {
        output[(i * cols) + j] = input[(i * cols) + j];
      }
    }
  }
}

component 
void pooling_max7(hls_avalon_slave_memory_argument(224*sizeof(Numeric)) Numeric * restrict input,
                  mm_src & restrict output,
                  const uint16 rows,
                  const uint16 cols) {

  const uint16 output_rows = rows / 2;
  const uint16 output_cols = cols / 2;

  // Pool columns together
  for (uint6 i = 0; i < rows; ++i) {
    for (uint6 j = 0; j < cols; j+=2) {
      
      hls_register const uint16 offset = (i * cols) + j;
      input[offset] = MAX(input[offset], input[offset + 1]);
    }
  }

  // Pool rows together
  for (uint6 i = 0; i < rows; i+=2) {
    for (uint6 j = 0; j < cols; j+=2) {
      
      hls_register const uint16 offset = (i * cols) + j;
      hls_register const uint16 offset_next = ((i + 1) * cols) + j;
      input[offset] = MAX(input[offset], input[offset_next]);
    }
  }

  for (uint6 i = 0; i < output_rows; ++i) {
    for (uint6 j = 0; j < output_cols; ++j) {
      output[(i * output_cols) + j] = input[(i * cols * 2) + (j * 2)];
    }
  }
}

component 
void pooling_avg7(hls_avalon_slave_memory_argument(224*sizeof(Numeric)) Numeric * restrict input,
                  mm_src & restrict output,
                  const uint16 rows,
                  const uint16 cols) {

  const uint16 output_rows = rows / 2;
  const uint16 output_cols = cols / 2;

  // Pool columns together, summing
  for (uint6 i = 0; i < rows; ++i) {
    for (uint6 j = 0; j < cols; j+=2) {
      
      hls_register const uint16 offset = (i * cols) + j;
      input[offset] = input[offset] + input[offset + 1];
    }
  }

  // Pool rows together, summing then averaging
  for (uint6 i = 0; i < rows; i+=2) {
    for (uint6 j = 0; j < cols; j+=2) {
      
      hls_register const uint16 offset = (i * cols) + j;
      hls_register const uint16 offset_next = ((i + 1) * cols) + j;
      input[offset] = (input[offset] + input[offset_next]) / 4;
    }
  }

  for (uint6 i = 0; i < output_rows; ++i) {
    for (uint6 j = 0; j < output_cols; ++j) {
      output[(i * output_cols) + j] = input[(i * cols * 2) + (j * 2)];
    }
  }
}

#endif