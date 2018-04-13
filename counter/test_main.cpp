#include "HLS/hls.h"
#include <stdio.h>
#include <math.h>
#include "tensor3.hpp"
#include "tensor4.hpp"
#include "convolution.hpp"

#define SIZE 24

int test_tensor3() {
  Numeric data[SIZE];
  for (int i = 0; i < SIZE; i++) {
    data[i] = i;
  }

  int ret;
  tensor3 t_row, t_col, t_dep;

  ret = tensor3_init(&t_row, 4, 2, 3, ROW_MAJ);
  tensor3_set_data(&t_row, data);

  ret = tensor3_init(&t_col, 4, 2, 3, COL_MAJ);
  tensor3_set_data(&t_col, data);

  ret = tensor3_init(&t_dep, 4, 2, 3, DEP_MAJ);
  tensor3_set_data(&t_dep, data);

  printf("row linear:\n");
  for (int i = 0; i < t_row.vol; i++) {
    printf("%f ", t_row.data[i]);
  }
  printf("\nrow indexed:\n");
  tensor3_print(&t_row);

  printf("\ncol linear:\n");
  for (int i = 0; i < t_col.vol; i++) {
    printf("%f ", t_col.data[i]);
  }
  printf("\ncol indexed:\n");
  tensor3_print(&t_col);

  printf("\ndep linear:\n");
  for (int i = 0; i < t_dep.vol; i++) {
    printf("%f ", t_dep.data[i]);
  }
  printf("\ndep indexed:\n");
  tensor3_print(&t_dep);

  printf("\n\n\n");

  return 0;
}

int test_tensor4() {
  int ret;
  tensor4 t_row, t_dep, t_chn;

  ret = tensor4_init(&t_row, 4, 2, 3, 2, ROW_MAJ);

  Numeric data[t_row.size];
  for (int i = 0; i < t_row.size; i++) {
    data[i] = i;
  }
  tensor4_set_data(&t_row, data);

  ret = tensor4_init(&t_dep, 4, 2, 3, 2, DEP_MAJ);
  tensor4_set_data(&t_dep, data);

  ret = tensor4_init(&t_chn, 4, 2, 3, 2, CHN_MAJ);
  tensor4_set_data(&t_chn, data);

  printf("row linear:\n");
  for (int i = 0; i < t_row.size; i++) {
    printf("%f ", t_row.data[i]);
  }
  printf("\nrow indexed:\n");
  tensor4_print(&t_row);

  printf("\ndep linear:\n");
  for (int i = 0; i < t_dep.size; i++) {
    printf("%f ", t_dep.data[i]);
  }
  printf("\ndep indexed:\n");
  tensor4_print(&t_dep);

  printf("\nchan linear:\n");
  for (int i = 0; i < t_chn.size; i++) {
    printf("%f ", t_chn.data[i]);
  }
  printf("\nchan indexed:\n");
  tensor4_print(&t_dep);

  printf("\n\n\n");  

  return 0;
}

int main() {
  // test_tensor4();
  test_convolution();
  return 0;
}

