#include "HLS/hls.h"
#include <stdio.h>
#include <math.h>
#include <hdf5.h>
#include "tensor3.hpp"

#define SIZE 24

int main() {
  Numeric data[SIZE];
  for (int i = 0; i < SIZE; i++) {
    data[i] = i;
  }

  int ret;
  tensor3 t_row, t_col, t_dep;

  ret = tensor3_init(&t_row, 4, 2, 3, ROW_MAJ);
  tensor3_init_data(&t_row, data);

  ret = tensor3_init(&t_col, 4, 2, 3, COL_MAJ);
  tensor3_init_data(&t_col, data);

  ret = tensor3_init(&t_dep, 4, 2, 3, DEP_MAJ);
  tensor3_init_data(&t_dep, data);

  for (int i = 0; i < t_row.vol; i++) {
    printf("%f ", t_row.data[i]);
  }
  printf("\n");
  for (int i = 0; i < t_col.vol; i++) {
    printf("%f ", t_col.data[i]);
  }
  printf("\n");
  for (int i = 0; i < t_dep.vol; i++) {
    printf("%f ", t_dep.data[i]);
  }
  printf("\n\n\n");

  hid_t file1;
  file1 = H5Fopen("../dependencies/frugally-deep/small.h5", H5F_ACC_RDWR, H5P_DEFAULT);
  return 0;
}

