#include "HLS/hls.h"
#include <stdio.h>

using namespace ihc;

#define SIZE 100
typedef mm_master<float, dwidth<32>, awidth<10>, latency<0> > mm_src_t;
typedef stream_in<float, buffer<SIZE> > fstream_in;
typedef stream_out<float> fstream_out;


component
void convolution5(fstream_in &in0,
                  fstream_in &in1,
                  fstream_in &in2,
                  fstream_out &out,
                  hls_avalon_slave_memory_argument(9*sizeof(float)) float *weights,
                  hls_stable_argument int in_size) {
  //float input[3];
  float sum[3];
  float outputs[7];
  float lweights[9];

  #pragma unroll
  for (int i = 0; i < 9; i++) {
    lweights[i] = weights[i];
  }

  #pragma unroll 1
  for (int m = 0; m < 3; m++) {
    // Includes ends for pipelining even if output is wasted
    // to avoid conditionals

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

  float arr_output[9];

/*
  fstream_in input_stream;
  fstream_out output_stream;

  for(int i = 0; i < 5; ++i) {
    for(int j = 0; j < 5; ++j) {
      input_stream.write(arr_input[i][j]);
    }
  }

  convolution3(input_stream, output_stream, (float *)arr_weights);
*/

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
      printf("%f ", output_stream.read());
    }
    printf("\n");
  }

  if (pass) {
    printf("PASSED\n");
  }
  else {
    printf("FAILED\n");
  }

  return 0;

}

