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

#define BUFFER_SIZE 32
#define BUFFER_LOAD_PIPELINE 1

component
void convolution7(mm_src & restrict input,
                  mm_src & restrict output,
                  mm_src & restrict weights,
                  hls_avalon_slave_memory_argument(BUFFER_SIZE*3*sizeof(Numeric)) Numeric * restrict bram_fifo,
                  // hls_avalon_slave_memory_argument(BUFFER_SIZE*sizeof(Numeric)) Numeric * restrict bram_fifo_in1,
                  // hls_avalon_slave_memory_argument(BUFFER_SIZE*sizeof(Numeric)) Numeric * restrict bram_fifo_in2,
                  hls_avalon_slave_memory_argument(BUFFER_SIZE*sizeof(Numeric)) Numeric * restrict bram_fifo_out0,
                  const uint16 weight_offset,
                  const uint16 rows,
                  const uint16 cols) {
  // convolver weights
  register Numeric lweights[3][3];

  // loads weights (test within function)
  #pragma loop_coalesce 2
  #pragma unroll 1
  for (uint2 i = 0; i < 3; ++i) {
    #pragma unroll 1
    for (uint2 j = 0; j < 3; ++j) {
      lweights[i][j] = weights[weight_offset + (i * 3) + j];
    }
  }

  #pragma ivdep
  #pragma loop_coalesce 2
  #pragma max_concurrency 1
  for (uint16 m = 0; m < rows - 2; ++m) {
    #pragma ivdep
    #pragma max_concurrency 1
    for (uint16 batch_offset = 0; batch_offset < cols; batch_offset += (BUFFER_SIZE - 2)) {

      // convolver registers
      register Numeric shift_registers0[3];
      register Numeric shift_registers1[3];
      register Numeric shift_registers2[3];

      // Loads data into registers and local storage
      #pragma ivdep safelen(1)
      #pragma unroll 1
      #pragma max_concurrency 1
      for (uint3 ii = 0; ii < 3; ++ii) {
        #pragma ivdep safelen(1)
        #pragma unroll 1
        #pragma max_concurrency 1
        for (uint16 j = 0; j < BUFFER_SIZE && j < cols; j += 4) {
          #pragma unroll
          for (uint3 k = 0; k < 4; ++k) {
            bram_fifo[(ii * BUFFER_SIZE) + j + k] = input[(cols * (m + ii)) + batch_offset + j + k];
          }
        }
      }

      // Convolve on entire buffer
      #pragma max_concurrency 1
      for (uint16 n = 0; n < BUFFER_SIZE && n <= cols; ++n) {
        // Convolution
        if (n > 2) {
          register Numeric tmp_out = 0;
          #pragma unroll
          for (uint2 j = 0; j < 3; ++j) {
            // printf("%f ", NUMERIC_VAL(shift_registers[i][j]));
            tmp_out += shift_registers0[2 - j] * lweights[0][j];
            tmp_out += shift_registers1[2 - j] * lweights[1][j];
            tmp_out += shift_registers2[2 - j] * lweights[2][j];
          }
          // printf("\n");
          bram_fifo_out0[n] = tmp_out;
          // printf("\nout = %f\n", NUMERIC_VAL(tmp_out));
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
      #pragma unroll 4
      for (uint16 n = 0; n < BUFFER_SIZE - 3 && n < cols - 2; ++n) {
        // printf("n = %d\n", UINT_VAL(n));
        output[(m * (cols - 2)) + n] += bram_fifo_out0[n + 3];
      }
    }
  }
}

// int convolution_3_3(tensor3 *input,
//                     tensor3 *output,
//                     tensor4 *kernel,
//                     mm_src &input
//                     mm_src &output) {

//   int p_inter = 2;
//   int p_intra = 2;



//   for (int c = 0; c < INT_DIV_CEIL(kernel->chans, p_inter); c += p_inter) {

//     for (int i = 0; i < INT_DIV_CEIL(input->depth, p_intra); i += p_intra) {

//       for (int cii = 0; cii < p_inter; cii++) {


//         for (int iii = 0; iii < p_intra; iii++) {

//           // Load p_inter kernels with p_intra depths

//           // Load p_intra input depths
//           // i = 0
//           // i = 1

//           for (int j = 0; j < input->rows; j++) {
//             for (int k = 0; k < input->cols; k++) {
//               // write to 3 (kernel size) streams
//               fstream_in_arr[iii].write( /* input[j][k][i + iii] */ );
//             }
//           }

//           // enqueue, convolution6(row0, row1, row2, tmpOut, kernel[:][:][i + iii][c + cii], input->cols)
//           // enqueue, output_to_input(tmpOut, tmpIn)
//           // enqueue, store_to_output2(tmpIn, fstream_out_arr[iii])
//         }
//       }
//       // executeAll
//     }
//   }
//   return 0;
// }

// int main() {
//   // (transposed for col-wise)
//   float arr_weights[3][3] = {
//     { 1.0f, 1.0f, 1.0f },
//     { 2.0f, 2.0f, 2.0f },
//     { 3.0f, 3.0f, 3.0f }
//   };

//   float arr_input[5][5] = {
//     { 0.0f, 0.0f, 0.0f, 1.0f, 1.0f },
//     { 0.0f, 0.0f, 0.0f, 1.0f, 1.0f },
//     { 0.0f, 0.0f, 0.0f, 1.0f, 1.0f },
//     { 0.0f, 0.0f, 0.0f, 1.0f, 1.0f },
//     { 0.0f, 0.0f, 0.0f, 1.0f, 1.0f }
//   };

//   float exp_output[3][3] = {
//     { 0.0f, 6.0f, 12.0f },
//     { 0.0f, 6.0f, 12.0f },
//     { 0.0f, 6.0f, 12.0f }
//   };

//   float arr_output[9];

//   fstream_in in1;
//   fstream_in in2;
//   fstream_in in3;
//   fstream_out output_stream;

//   for (int m = 0; m < 3; m++) {
//     for (int i = m; i < (m + 3) && i < 5; i++) {
//       for (int j = 0; j < 5; ++j) {
//         in1.write(arr_input[i + 0][j]);
//         in2.write(arr_input[i + 1][j]);
//         in3.write(arr_input[i + 2][j]);
//       }
//     }
//   }

//   // convolution6(in1, in2, in3, output_stream, (float *)arr_weights, 5);

//   bool pass = true;
//   for (int i = 0; i < 3; ++i) {
//     for (int j = 0; j < 3; ++j) {
//       float output = output_stream.read();
//       printf("%f %f\n", exp_output[i][j], output);
//       if (!fcompare(exp_output[i][j], output)) {
//         pass = false;
//       }
//       //printf("%f ", output_stream.read());
//     }
//     //printf("\n");
//   }

//   if (pass) {
//     printf("PASSED\n");
//   }
//   else {
//     printf("FAILED\n");
//   }

//   return 0;

// }

int main() {
  Numeric arr_weights[3][3] = {
    { 1.0f, 1.0f, 1.0f },
    { 2.0f, 2.0f, 2.0f },
    { 3.0f, 3.0f, 3.0f }
  };

  Numeric arr_input[5][5] = {
    { 0.0f, 0.0f, 0.0f, 1.0f, 1.0f },
    { 0.0f, 0.0f, 0.0f, 1.0f, 1.0f },
    { 0.0f, 0.0f, 0.0f, 1.0f, 1.0f },
    { 0.0f, 0.0f, 0.0f, 1.0f, 1.0f },
    { 0.0f, 0.0f, 0.0f, 1.0f, 1.0f }
  };

  Numeric arr_output[3][3] = {
    { 0.0f, 0.0f, 0.0f },
    { 0.0f, 0.0f, 0.0f },
    { 0.0f, 0.0f, 0.0f }
  };

  Numeric exp_output[3][3] = {
    { 0.0f, 6.0f, 12.0f },
    { 0.0f, 6.0f, 12.0f },
    { 0.0f, 6.0f, 12.0f }
  };

  mm_src mm_src_weights(arr_weights, 9 * 4);
  mm_src mm_src_input(arr_input, 25 * 4);
  mm_src mm_src_output(arr_output, 9 * 4);

  Numeric bram_fifo_in0[BUFFER_SIZE * 3];
  // Numeric bram_fifo_in1[BUFFER_SIZE];
  // Numeric bram_fifo_in2[BUFFER_SIZE];
  Numeric bram_fifo_out0[BUFFER_SIZE];

  convolution7(mm_src_input, mm_src_output, mm_src_weights, bram_fifo_in0, bram_fifo_out0, 0, 5, 5);

  bool pass = true;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      printf("%f %f\n", NUMERIC_VAL(exp_output[i][j]), NUMERIC_VAL(arr_output[i][j]));
      if (!fcompare(exp_output[i][j], arr_output[i][j])) {
        pass = false;
      }
      //printf("%f ", output_stream.read());
    }
    //printf("\n");
  }

  if (pass) {
    printf("PASSED\n");
  }
  else {
    printf("FAILED\n");
  }

  return 0;

}
