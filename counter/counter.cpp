#include "HLS/hls.h"
#include "HLS/ac_fixed.h"
#include "HLS/ac_fixed_math.h"

#include "tensor3.hpp"
#include "tensor4.hpp"
#include "common.hpp"
#include <stdio.h>
// #include <math.h>

#define SIZE 10

using namespace ihc;

typedef ac_fixed<16, 8, true> float16;

typedef mm_master<float16, dwidth<32>, awidth<10>, latency<100> > mm_src;

typedef stream_in<float, buffer<SIZE> > fstream_in;
typedef stream_out<float> fstream_out;

bool fcompare(float16 a, float16 b) {
    return fabs(a.to_double() - b.to_double()) < 1e-6f;
}


component
void convolution6(mm_src &input,
                  const uint input_offset,
                  mm_src &output,
                  const uint output_offset,
                  mm_src &weights,
                  const uint weight_offset,
                  const uint rows,
                  const uint cols) {
  #pragma ivdep
  for (int m = 0; m < rows - 2; m++) {

    // Local input storage
    float16 bram_fifo_in[3][256];
    float16 bram_fifo_out0[256];
    
    #pragma unroll
    for (int i = 0; i < 3; i++) {

      #pragma ivdep
      for (int j = 0; j < cols; j++) {
        bram_fifo_in[i][j] = input[input_offset + (cols * (m + i)) + j];
      }
    }

    #pragma ivdep
    for (int n = 0; n < cols - 2; n++) {

      float16 lweights[3][3];

      #pragma unroll
      for (int i = 0; i < 3; i++) {
        #pragma unroll
        for (int j = 0; j < 3; j++) {
          lweights[i][j] = weights[weight_offset + (i * 3) + j];
        }
      }

      bram_fifo_out0[n] = 0;

      #pragma unroll
      for (int i = 0; i < 3; i++) {
        #pragma unroll
        for (int j = 0; j < 3; j++) {
          bram_fifo_out0[n] += bram_fifo_in[i][n + j] * lweights[i][j];
        }
      }


    }

    #pragma ivdep
    for (int n = 0; n < cols - 2; n++) {
      output[output_offset + (m * (cols - 2)) + n] += bram_fifo_out0[n];
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
  float16 arr_weights[3][3] = {
    { 1.0f, 1.0f, 1.0f },
    { 2.0f, 2.0f, 2.0f },
    { 3.0f, 3.0f, 3.0f }
  };

  float16 arr_input[5][5] = {
    { 0.0f, 0.0f, 0.0f, 1.0f, 1.0f },
    { 0.0f, 0.0f, 0.0f, 1.0f, 1.0f },
    { 0.0f, 0.0f, 0.0f, 1.0f, 1.0f },
    { 0.0f, 0.0f, 0.0f, 1.0f, 1.0f },
    { 0.0f, 0.0f, 0.0f, 1.0f, 1.0f }
  };

  float16 arr_output[3][3] = {
    { 0.0f, 0.0f, 0.0f },
    { 0.0f, 0.0f, 0.0f },
    { 0.0f, 0.0f, 0.0f }
  };

  float16 exp_output[3][3] = {
    { 0.0f, 6.0f, 12.0f },
    { 0.0f, 6.0f, 12.0f },
    { 0.0f, 6.0f, 12.0f }
  };

  mm_src mm_src_weights(arr_weights, 9 * 4);
  mm_src mm_src_input(arr_input, 25 * 4);
  mm_src mm_src_output(arr_output, 9 * 4);


  convolution6(mm_src_input, 0, mm_src_output, 0, mm_src_weights, 0, 5, 5);

  bool pass = true;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      printf("%lf %lf\n", exp_output[i][j].to_double(), arr_output[i][j].to_double());
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
