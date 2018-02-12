#include "HLS/hls.h"
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

component
void convolution5(fstream_in &in0,
                  fstream_in &in1,
                  fstream_in &in2,
                  fstream_out &out,
                  hls_avalon_slave_memory_argument(9*sizeof(float)) float *weights,
                  hls_stable_argument int in_size) {
  // Includes ends for pipelining even if output is wasted
  // to avoid conditionals
  float outputs[7];

  // Sums of each col in convolver
  float sum[3];

  // Local weights loaded in registers
  float lweights[9];

  #pragma unroll
  for (int i = 0; i < 9; i++) {
    lweights[i] = weights[i];
  }

  #pragma unroll 1
  for (int m = 0; m < 3; m++) {

    #pragma unroll
    for (int i = 0; i < 7; i++) {
      outputs[i] = 0;
    }

    #pragma unroll
    for (int i = 0; i < in_size; i++) {
      float input0 = in0.read();
      float input1 = in1.read();
      float input2 = in2.read();

      sum[0]  = input0 * lweights[0];
      sum[0] += input1 * lweights[1];
      sum[0] += input2 * lweights[2];
      outputs[2 + i] += sum[0];

      sum[1]  = input0 * lweights[3];
      sum[1] += input1 * lweights[4];
      sum[1] += input2 * lweights[5];
      outputs[2 + i - 1] += sum[1];

      sum[2]  = input0 * lweights[6];
      sum[2] += input1 * lweights[7];
      sum[2] += input2 * lweights[8];
      outputs[2 + i - 2] += sum[2];
    }
    for (int i = 2; i < 5; i++) {
      out.write(outputs[i]);
    }
  }
}

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
    { 0.0f, 9.0f, 15.0f },
    { 0.0f, 9.0f, 15.0f },
    { 0.0f, 9.0f, 15.0f }
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

  convolution5(in1, in2, in3, output_stream, (float *)arr_weights, 5);

  bool pass = true;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      if (!fcompare(exp_output[i][j], output_stream.read())) {
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

