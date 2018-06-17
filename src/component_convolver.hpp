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
void convolution7(mm_src & restrict input,
                  mm_src & restrict output,
                  mm_src & restrict weights,
                  hls_avalon_slave_memory_argument(BUFFER_SIZE*3*sizeof(Numeric)) Numeric * restrict bram_fifo,
                  hls_avalon_slave_memory_argument(BUFFER_SIZE*sizeof(Numeric)) Numeric * restrict bram_fifo_out0,
                  const uint16 weight_offset,
                  const uint16 rows,
                  const uint16 cols) {
  // convolver weights
  hls_register Numeric lweights[3][3];

  // loads weights (test within function)
  #pragma loop_coalesce 2
  #pragma unroll 1
  for (uint2 i = 0; i < 3; ++i) {
    #pragma unroll 1
    for (uint2 j = 0; j < 3; ++j) {
      lweights[i][j] = weights[weight_offset + (i * 3) + j];
    }
  }

  #pragma max_concurrency 1
  for (uint16 m = 0; m < rows - 2; ++m) {
    #pragma max_concurrency 1
    for (uint16 batch_offset = 0; batch_offset < cols; batch_offset += (BUFFER_SIZE - 3)) {

      
      hls_register const uint16 output_offset = (m * (cols - 2)) + batch_offset;

      // convolver registers
      hls_register Numeric shift_registers0[3];
      hls_register Numeric shift_registers1[3];
      hls_register Numeric shift_registers2[3];

      // Loads data into registers and local storage
      #pragma ivdep
      #pragma loop_coalesce 2
      #pragma unroll 1
      #pragma max_concurrency 1
      for (uint3 ii = 0; ii < 3; ++ii) {
        hls_register const uint16 input_offset = (cols * (m + ii)) + batch_offset;
        hls_register const uint16 fifo_offset = (ii * BUFFER_SIZE);

        #pragma ivdep
        #pragma unroll 1
        for (uint6 j = 0; j < BUFFER_SIZE && j + batch_offset < cols; ++j) {
          bram_fifo[fifo_offset + j] = input[input_offset + j];
        }
      }

      #pragma ivdep
      #pragma unroll 1
      for (uint6 j = 0; j < BUFFER_SIZE - 3 && j + batch_offset < cols - 2; ++j) {
        bram_fifo_out0[j + 3] = output[output_offset + j];
      }

      // Convolve on entire buffer
      #pragma unroll 1
      #pragma max_concurrency 1
      for (uint6 n = 0; n < BUFFER_SIZE; ++n) {
        // Convolution
        if (n > 2) {
          hls_register Numeric tmp_out = 0;
          #pragma unroll
          for (uint2 j = 0; j < 3; ++j) {
            bram_fifo_out0[n] += shift_registers0[2 - j] * lweights[0][j];
            bram_fifo_out0[n] += shift_registers1[2 - j] * lweights[1][j];
            bram_fifo_out0[n] += shift_registers2[2 - j] * lweights[2][j];
          }
        }

        // Shift register values
        #pragma unroll
        for (uint2 j = 2; j > 0; --j) {
          shift_registers0[j] = shift_registers0[j - 1];
        }
        shift_registers0[0] = bram_fifo[(BUFFER_SIZE * 0) + n];

        #pragma unroll
        for (uint2 j = 2; j > 0; --j) {
          shift_registers1[j] = shift_registers1[j - 1];
        }
        shift_registers1[0] = bram_fifo[(BUFFER_SIZE * 1) + n];

        #pragma unroll
        for (uint2 j = 2; j > 0; --j) {
          shift_registers2[j] = shift_registers2[j - 1];
        }
        shift_registers2[0] = bram_fifo[(BUFFER_SIZE * 2) + n];
      }

      #pragma ivdep
      #pragma unroll 1
      for (uint6 n = 0; n < BUFFER_SIZE - 3 && n + batch_offset < cols - 2; ++n) {
        output[output_offset + n] = bram_fifo_out0[n + 3];
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
      input[(i * cols) + j] = MAX(input[(i * cols) + j], 0.0);
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
                    const Numeric beta,
                    const Numeric gamma,
                    const bool write_to_output) {
  for (uint6 i = 0; i < rows; ++i) {
    for (uint6 j = 0; j < cols; ++j) {
      input[(i * cols) + j] = MAX((input[(i * cols) + j] * gamma) + beta, 0.0);
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