#include "HLS/hls.h"
#include "tensor3.hpp"
#include "tensor4.hpp"
#include "common.hpp"
#include <stdio.h>
#include <math.h>

#define SIZE 10

using namespace ihc;

typedef mm_master<float, dwidth<32>, awidth<10>, latency<0> > mm_src_t;
typedef stream_in<float, buffer<SIZE> > fstream_in;
typedef stream_out<float> fstream_out;

bool fcompare(float a, float b) {
    return fabs(a - b) < 1e-6f;
}

hls_max_concurrency(6)
component
void convolution6(fstream_in &row0,
                  fstream_in &row1,
                  fstream_in &row2,
                  fstream_out &out,
                  hls_avalon_slave_memory_argument(9*sizeof(float)) float *lweights,
                  hls_stable_argument int in_size) {
  #pragma unroll 1
  for (int m = 0; m < 3; m++) {
    // Local weights loaded in registers
    float registers[9];

    registers[1] = row0.read();
    registers[4] = row1.read();
    registers[7] = row2.read();

    registers[0] = row0.read();
    registers[3] = row1.read();
    registers[6] = row2.read();

    for (int i = 2; i < in_size; i++) {
      // Sums of each col in convolver
      float sum[3];

      registers[2] = registers[1];
      registers[5] = registers[4];
      registers[8] = registers[7];
      sum[2] = (registers[2] * lweights[2]) + (registers[5] * lweights[5]) + (registers[8] * lweights[8]);

      registers[1] = registers[0];
      registers[4] = registers[3];
      registers[7] = registers[6];
      sum[1] = (registers[1] * lweights[1]) + (registers[4] * lweights[4]) + (registers[7] * lweights[7]);

      registers[0] = row0.read();
      registers[3] = row1.read();
      registers[6] = row2.read();
      sum[0] = (registers[0] * lweights[0]) + (registers[3] * lweights[3]) + (registers[6] * lweights[6]);
      
      out.write(sum[0] + sum[1] + sum[2]);
    }
  }
}

// int convolution_3_3(tensor3 *input, tensor3 *output, tensor4 *kernel) {
//   fstream_in in1;
//   fstream_in in2;
//   fstream_in in3;
//   fstream_out output_stream;

  
//   int convolver_parallelism = 6;

//   for (int c = 0; c < kernel->chans; c++) {
//     for (int i = 0; i < input->depth; i++) {
//       for (int j = 0; j < input->rows; j++) {
//         for (int k = 0; k < input->cols; k++) {


//           if (i == 0) {
//             ROW3_MAJ_VAL(output, j, k, c) = V_VAL(bias, c);
//           }
//           // Kernel multiplication
//           for (int dx = 0; dx < kernel->rows; dx++) {
//             for (int dy = 0; dy < kernel->cols; dy++) {
//               // printf("%d %d | %d %d\n", j, k, j * strideX, j * strideY);
//               ROW3_MAJ_VAL(output, j, k, c) += ROW4_MAJ_VAL(kernel, dx, dy, i, c) * ROW3_MAJ_VAL(input, (j * strideX) + dx, (k * strideY) + dy, i);
//             }
//           }
//         }
//       }
//     }
//   }

// }

int main() {
  // (transposed for col-wise)
  float arr_weights[3][3] = {
    { 1.0f, 1.0f, 1.0f },
    { 2.0f, 2.0f, 2.0f },
    { 3.0f, 3.0f, 3.0f }
  };

  float arr_input[5][5] = {
    { 0.0f, 0.0f, 0.0f, 1.0f, 1.0f },
    { 0.0f, 0.0f, 0.0f, 1.0f, 1.0f },
    { 0.0f, 0.0f, 0.0f, 1.0f, 1.0f },
    { 0.0f, 0.0f, 0.0f, 1.0f, 1.0f },
    { 0.0f, 0.0f, 0.0f, 1.0f, 1.0f }
  };

  float exp_output[3][3] = {
    { 0.0f, 6.0f, 12.0f },
    { 0.0f, 6.0f, 12.0f },
    { 0.0f, 6.0f, 12.0f }
  };

  float arr_output[9];

  fstream_in in1;
  fstream_in in2;
  fstream_in in3;
  fstream_out output_stream;

  for (int m = 0; m < 3; m++) {
    for (int i = m; i < (m + 3) && i < 5; i++) {
      for (int j = 0; j < 5; ++j) {
        in1.write(arr_input[i + 0][j]);
        in2.write(arr_input[i + 1][j]);
        in3.write(arr_input[i + 2][j]);
      }
    }
  }

  convolution6(in1, in2, in3, output_stream, (float *)arr_weights, 5);

  bool pass = true;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      float output = output_stream.read();
      printf("%f %f\n", exp_output[i][j], output);
      if (!fcompare(exp_output[i][j], output)) {
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

