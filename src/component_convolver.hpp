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
                  hls_avalon_slave_memory_argument(BUFFER_SIZE*sizeof(Numeric)) Numeric * restrict bram_fifo0,
                  hls_avalon_slave_memory_argument(BUFFER_SIZE*sizeof(Numeric)) Numeric * restrict bram_fifo1,
                  hls_avalon_slave_memory_argument(BUFFER_SIZE*sizeof(Numeric)) Numeric * restrict bram_fifo2,
                  hls_avalon_slave_memory_argument(BUFFER_SIZE*sizeof(Numeric)) Numeric * restrict bram_fifo_out0,
                  hls_avalon_slave_register_argument uint16 weight_offset,
                  hls_avalon_slave_register_argument uint16 rows,
                  hls_avalon_slave_register_argument uint16 cols,
                  hls_avalon_slave_register_argument uint8 paddingY,
                  hls_avalon_slave_register_argument uint8 paddingX) {
  #pragma max_concurrency 1
  for (uint16 m = 0; m < rows - 2; ++m) {
    #pragma max_concurrency 1
    #pragma unroll 1
    for (uint16 batch_offset = 0; batch_offset < cols; batch_offset += (BUFFER_SIZE - 3)) {
                                                // current row     // number of columns         //offset
      hls_register const uint16 output_offset = ((m + paddingY) * (cols - 2 + (paddingX * 2))) + batch_offset + paddingX;

      // convolver registers
      Numeric shift_registers0[3];
      Numeric shift_registers1[3];
      Numeric shift_registers2[3];

      // Loads data into registers and local storage
      #pragma ivdep
      #pragma unroll 1
      #pragma max_concurrency 1
      for (uint8 j = 0; j < BUFFER_SIZE && j + batch_offset < cols; ++j) {
        bram_fifo0[j] = input[(cols * (m + 0)) + batch_offset + j];
        bram_fifo1[j] = input[(cols * (m + 1)) + batch_offset + j];
        bram_fifo2[j] = input[(cols * (m + 2)) + batch_offset + j];
      }

      #pragma ivdep
      #pragma unroll 1
      for (uint8 j = 0; j < BUFFER_SIZE - 3 && j + batch_offset < cols - 2; ++j) {
        bram_fifo_out0[j + 3] = output[output_offset + j];
      }

      // Convolve on entire buffer
      #pragma unroll 1
      #pragma max_concurrency 1
      for (uint8 n = 0; n < BUFFER_SIZE && n + batch_offset <= cols; ++n) {
        // Convolution
        if (n > 2) {
          hls_register Numeric tmp_out = 0;
          #pragma unroll
          for (uint2 j = 0; j < 3; ++j) {
            bram_fifo_out0[n] += shift_registers0[2 - j] * lweights[(0 * 3) + j];
            bram_fifo_out0[n] += shift_registers1[2 - j] * lweights[(1 * 3) + j];
            bram_fifo_out0[n] += shift_registers2[2 - j] * lweights[(2 * 3) + j];
          }
        }

        // Shift register values
        #pragma unroll
        for (uint2 j = 2; j > 0; --j) {
          shift_registers0[j] = shift_registers0[j - 1];
        }
        shift_registers0[0] = bram_fifo0[n];

        #pragma unroll
        for (uint2 j = 2; j > 0; --j) {
          shift_registers1[j] = shift_registers1[j - 1];
        }
        shift_registers1[0] = bram_fifo1[n];

        #pragma unroll
        for (uint2 j = 2; j > 0; --j) {
          shift_registers2[j] = shift_registers2[j - 1];
        }
        shift_registers2[0] = bram_fifo2[n];
      }

      #pragma ivdep
      #pragma unroll 1
      for (uint8 n = 0; n < BUFFER_SIZE - 3 && n + batch_offset < cols - 2; ++n) {
        output[output_offset + n] = bram_fifo_out0[n + 3];
      }
    }
  }
}

#define PE_ARRAY_ROWS 3
#define PE_ARRAY_COLS 3

// OOM
component
void convolution9(mm_src & restrict input,
                  mm_src & restrict output,
                  hls_avalon_slave_memory_argument(3*3*sizeof(Numeric)) Numeric * lweights,
                  // output buffer should fill up a single tile
                  tiled_tensor3 input_tensor, // change to raw params
                  hls_avalon_slave_register_argument uint8 depth,
                  hls_avalon_slave_register_argument uint8 nr_kernels,
                  hls_avalon_slave_register_argument uint8 paddingY,
                  hls_avalon_slave_register_argument uint8 paddingX) {


  for (uint8 i = 0; i < input_tensor->rows_t; ++i) {
    for (uint8 j = 0; j < input_tensor->cols_t; ++j) {


      Numeric *tile_in = tiled_tensor3_tile(input_tensor, i, j, depth);
      Numeric *tile_out = tiled_tensor3_tile(output_tensor, i, j, depth);

      Numeric PE_ARR[PE_ARRAY_ROWS][PE_ARRAY_COLS];
      for (uint4 ii = 0; ii < PE_ARRAY_ROWS; ++ii) {
        for (uint4 jj = 0; jj < PE_ARRAY_COLS; ++jj) {
          PE_ARR[ii][jj] = tile_in[ii][jj];
        }
      }

      uint8 curr_row = 0;
      uint8 curr_col = 0;
      do {


        //////////////////////////////////////////////////////////////////////////
        // CONVOLUTION

        for (uint4 ii = 0; ii < PE_ARRAY_ROWS; ++ii) {
          for (uint4 jj = 0; jj < PE_ARRAY_COLS; ++jj) {
            tile_out[ii + curr_row][jj + curr_col] = PE_ARR[ii][jj] * lweights[curr_row][curr_col];
          }
        }

        //
        //////////////////////////////////////////////////////////////////////////
        // TRAVERSAL

        // Traversing right
        if (curr_row % 2 == 0 && curr_col + 1 < input_tensor->tile_rows) {
          ++curr_col;

          // shift PE array left, load new elements
          #pragma unroll
          for (uint2 m = 0; m < PE_ARRAY_ROWS; ++m) {
            #pragma unroll
            for (uint2 n = 1; n < PE_ARRAY_COLS; ++n) {
              PE_ARR[m][n - 1] = PE_ARR[m][n];
            }
            PE_ARR[m][PE_ARRAY_COLS - 1] = tile_in[m + curr_row][PE_ARRAY_COLS - 1 + curr_col];
          }

        // Traversing left
        } else if (curr_row % 2 == 1 && curr_col - 1 >= 0) { 
          --curr_col;

          // shift PE array right, load new elements
          #pragma unroll
          for (uint2 m = 0; m < PE_ARRAY_ROWS; ++m) {
            #pragma unroll
            for (uint2 n = PE_ARRAY_COLS; n > 0; --n) {
              PE_ARR[m][n] = PE_ARR[m][n - 1];
            }
            PE_ARR[m][0] = tile_in[m + curr_row][curr_col];
          }
        // Traversing down on left or right
        } else if (curr_col == 0 || curr_col == input_tensor->tile_rows - 1) {
          ++curr_row;

          // shift PE array up, load new elements
          #pragma unroll
          for (uint2 n = 0; m < PE_ARRAY_COLS; ++n) {
            #pragma unroll
            for (uint2 m = 1; m < PE_ARRAY_ROWS; ++m) {
              PE_ARR[m - 1][n] = PE_ARR[m][n];
            }
            PE_ARR[PE_ARRAY_ROWS - 1][n] = tile_in[PE_ARRAY_ROWS - 1 + curr_row][n + curr_col];
          }
          
        }

        //
        //////////////////////////////////////////////////////////////////////////



      } while (curr_row != tile_in->tile_rows - 1 && (curr_col != -1 || curr_col != tile_in->tile_rows));


    }
  }

}

// POOM
//
// component
// void convolution9(mm_src & restrict input,
//                   mm_src & restrict output,
//                   hls_avalon_slave_memory_argument(3*3*sizeof(Numeric)) Numeric * lweights,
//                   // output buffer should fill up a single tile
//                   tiled_tensor3 input_tensor, // change to raw params
//                   hls_avalon_slave_register_argument uint8 nr_kernels,
//                   hls_avalon_slave_register_argument p,
//                   hls_avalon_slave_register_argument uint8 paddingY,
//                   hls_avalon_slave_register_argument uint8 paddingX) {
//   // Load single tile's channels successively
//   // Partial sums are reused until tile of output map is complete
//   // store final output maps after pooling

//   for (uint8 i = 0; i < input_tensor.rows_t; ++i) {
//     for (uint8 j = 0; j < input_tensor.cols_t; ++j) {
//       for (uint8 k = 0; k < nr_kernels / p; k += p) {             // p: Number of kernel maps to compute in parallel to account for stride, typically stride^2
//         for (uint8 l = 0; l < input_tensor.depth_t; ++l) {      
          

//           // load input tiles (i,j,k+0) .. (i,j,k+p-1)
//           // load kernels k+0 .. k+p-1

//           // zigzag on row
//           //   multiply (i,j,k+0) w0_0, (i,j,k+1) w0_1, .. 
//           //   store in output buffer
//           //   shift right/down/left (the zigzag) by stride, repeat
          
//         }
//       }
//     }
//   }

// }

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