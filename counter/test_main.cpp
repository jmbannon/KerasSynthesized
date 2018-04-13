#include "HLS/hls.h"
#include <stdio.h>
#include <math.h>
#include "tensor3.hpp"
#include "convolution.hpp"

#define SIZE 24

int main() {
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

  printf("\ncol linear:");
  for (int i = 0; i < t_col.vol; i++) {
    printf("%f ", t_col.data[i]);
  }
  printf("\ncol indexed:\n");
  tensor3_print(&t_col);

  printf("\ndep linear:");
  for (int i = 0; i < t_dep.vol; i++) {
    printf("%f ", t_dep.data[i]);
  }
  printf("\ndep indexed:\n");
  tensor3_print(&t_dep);

  printf("\n\n\n");

  test_convolution();
  

  return 0;
}

