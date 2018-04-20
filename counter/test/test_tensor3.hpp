#ifndef TEST_TENSOR3_HPP
#define TEST_TENSOR3_HPP

#include "HLS/hls.h"
#include <stdio.h>
#include <math.h>
#include "../tensor3.hpp"
#include "../tensor4.hpp"
#include "../convolution.hpp"

int test_tensor3_row() {
  const uint rows = 4;
  const uint cols = 2;
  const uint depth = 3;
  const uint vol = 24;

  int ret;
  tensor3 t_row;
  Numeric data[vol];
  
  for (int i = 0; i < vol; i++) {
    data[i] = i;
  }

  ret = tensor3_init(&t_row, 4, 2, 3, ROW_MAJ);
  tensor3_set_data(&t_row, data);

  for (uint i = 0; i < vol; i++) {
    if (t_row.data[i] != i) {
      return 1;
    }
  }

  int idx = 0;
  for (uint i = 0; i < t_row.depth; i++) {
    for (uint j = 0; j < t_row.rows; j++) {
      for (uint k = 0; k < t_row.cols; k++) {
        if (tensor3_val(&t_row, j, k, i) != idx++) {
          return 1;
        }
      }
    }
  }

  return 0;
}

int test_tensor3_col() {
  const uint rows = 4;
  const uint cols = 2;
  const uint depth = 3;
  const uint vol = 24;

  int ret;
  tensor3 t_col;
  Numeric data[vol];
  
  for (int i = 0; i < vol; i++) {
    data[i] = i;
  }

  ret = tensor3_init(&t_col, 4, 2, 3, COL_MAJ);
  tensor3_set_data(&t_col, data);

  Numeric expected_linear[24] = { 0., 2., 4., 6., 1., 3., 5., 7., 8., 10., 12., 14., 9., 11., 13., 15., 16., 18., 20., 22., 17., 19., 21., 23. };
  for (uint i = 0; i < vol; i++) {
    if (t_col.data[i] != expected_linear[i]) {
      return 1;
    }
  }

  int idx = 0;
  for (uint i = 0; i < t_col.depth; i++) {
    for (uint j = 0; j < t_col.rows; j++) {
      for (uint k = 0; k < t_col.cols; k++) {
        if (tensor3_val(&t_col, j, k, i) != idx++) {
          return 1;
        }
      }
    }
  }

  return 0;
}

int test_tensor3_dep() {
  const uint rows = 4;
  const uint cols = 2;
  const uint depth = 3;
  const uint vol = 24;

  int ret;
  tensor3 t_dep;
  Numeric data[vol];
  
  for (int i = 0; i < vol; i++) {
    data[i] = i;
  }

  ret = tensor3_init(&t_dep, 4, 2, 3, DEP_MAJ);
  tensor3_set_data(&t_dep, data);

  Numeric expected_linear[24] = { 0., 8., 16., 1., 9., 17., 2., 10., 18., 3., 11., 19., 4., 12., 20., 5., 13., 21., 6., 14., 22., 7., 15., 23. };
  for (uint i = 0; i < vol; i++) {
    if (t_dep.data[i] != expected_linear[i]) {
      return 1;
    }
  }

  int idx = 0;
  for (uint i = 0; i < t_dep.depth; i++) {
    for (uint j = 0; j < t_dep.rows; j++) {
      for (uint k = 0; k < t_dep.cols; k++) {
        if (tensor3_val(&t_dep, j, k, i) != idx++) {
          return 1;
        }
      }
    }
  }

  return 0;
}

// int test_tensor3(char *err_msg) {
//   const uint rows = 4;
//   const uint cols = 2;
//   const uint depth = 3;
//   const uint vol = row * cols * depth;

//   int ret;
//   tensor3 t_row, t_col, t_dep;
//   Numeric data[vol];
  
//   for (int i = 0; i < vol; i++) {
//     data[i] = i;
//   }

//   ret = tensor3_init(&t_row, 4, 2, 3, ROW_MAJ);
//   tensor3_set_data(&t_row, data);

//   ret = tensor3_init(&t_col, 4, 2, 3, COL_MAJ);
//   tensor3_set_data(&t_col, data);

//   ret = tensor3_init(&t_dep, 4, 2, 3, DEP_MAJ);
//   tensor3_set_data(&t_dep, data);

//   printf("row linear:\n");
//   for (int i = 0; i < t_row.vol; i++) {
//     printf("%f ", t_row.data[i]);
//   }
//   printf("\nrow indexed:\n");
//   tensor3_print(&t_row);

//   printf("\ncol linear:\n");
//   for (int i = 0; i < t_col.vol; i++) {
//     printf("%f ", t_col.data[i]);
//   }
//   printf("\ncol indexed:\n");
//   tensor3_print(&t_col);

//   printf("\ndep linear:\n");
//   for (int i = 0; i < t_dep.vol; i++) {
//     printf("%f ", t_dep.data[i]);
//   }
//   printf("\ndep indexed:\n");
//   tensor3_print(&t_dep);

//   printf("\n\n\n");

//   return 0;
// }

#endif