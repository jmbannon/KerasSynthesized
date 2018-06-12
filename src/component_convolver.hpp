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
void convolution6(mm_master<Numeric, aspace<2>, align<16>, awidth<16>, latency<0>, maxburst<4>, dwidth<64>, waitrequest<true> > &input,
                  const uint16 input_offset,
                  mm_master<Numeric, aspace<3>, align<16>, awidth<16>, latency<0>, maxburst<4>, dwidth<64>, waitrequest<true> > &output,
                  const uint16 output_offset,
                  mm_master<Numeric, aspace<4>, align<16>, awidth<16>, latency<0>, maxburst<4>, dwidth<64>, waitrequest<true> > &weights,
                  const uint16 weight_offset,
                  const uint16 rows,
                  const uint16 cols) {
  #pragma ivdep
  for (uint16 m = 0; m < rows - 2; m++) {

    // Local input storage
    Numeric bram_fifo_in[3][256];
    Numeric bram_fifo_out0[256];
    
    #pragma unroll
    for (uint2 i = 0; i < 3; ++i) {
      uint32 offset = input_offset + (cols * (m + i));
      #pragma ivdep
      for (uint16 j = 0; j < cols; ++j) {
        bram_fifo_in[i][j] = input[offset + j];
      }
    }

    #pragma ivdep
    for (uint16 n = 0; n < cols - 2; ++n) {

      Numeric lweights[3][3];

      #pragma unroll
      for (uint2 i = 0; i < 3; ++i) {
        #pragma unroll
        for (uint2 j = 0; j < 3; j++) {
          lweights[i][j] = weights[weight_offset + (i * 3) + j];
        }
      }

      bram_fifo_out0[n] = 0;

      #pragma unroll
      for (uint2 i = 0; i < 3; ++i) {
        #pragma unroll
        for (uint2 j = 0; j < 3; ++j) {
          bram_fifo_out0[n] += bram_fifo_in[i][n + j] * lweights[i][j];
        }
      }


    }

    #pragma ivdep
    for (uint16 n = 0; n < cols - 2; ++n) {
      output[output_offset + (m * (cols - 2)) + n] += bram_fifo_out0[n];
    }
  }
}

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

#endif