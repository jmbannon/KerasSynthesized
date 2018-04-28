#ifndef KERNEL_HPP
#define KERNEL_HPP

#include "HLS/hls.h"
#include "tensor4.hpp"
#include "common.hpp"
#include <stdio.h>
#include <math.h>


// ROW-COL-DEP-CHAN
#define ROW4_MAJ_IDX_RAW(rows, cols, depth, row, col, dep, ch) (((ch) * (depth) * (rows) * (cols)) + ((dep) * (rows) * (cols)) + ((row) * (cols)) + (col))
#define ROW4_MAJ_IDX(t, row, col, dep, ch) ROW4_MAJ_IDX_RAW((t)->rows, (t)->cols, (t)->depth, row, col, dep, ch)
#define ROW4_MAJ_VAL(t, row, col, dep, ch) ((t)->data[ROW4_MAJ_IDX((t), (row), (col), (dep), (ch))])

// DEP-CHAN-ROW-COL
#define DEP4_MAJ_IDX_RAW(cols, depth, chans, row, col, dep, ch) (((row) * (cols) * (chans) * (depth)) + ((col) * (chans) * (depth)) + ((ch) * (depth)) + (dep))
#define DEP4_MAJ_IDX(t, row, col, dep, ch) DEP4_MAJ_IDX_RAW((t)->cols, (t)->depth, (t)->chans, row, col, dep, ch)
#define DEP4_MAJ_VAL(t, row, col, dep, ch) ((t)->data[DEP4_MAJ_IDX((t), (row), (col), (dep))])

// CHAN-DEP-ROW-COL
#define CHN4_MAJ_IDX_RAW(cols, depth, chans, row, col, dep, ch) (((row) * (cols) * (chans) * (depth)) + ((col) * (chans) * (depth)) + ((dep) * (chans)) + (ch))
#define CHN4_MAJ_IDX(t, row, col, dep, ch) CHN4_MAJ_IDX_RAW((t)->cols, (t)->depth, (t)->chans, row, col, dep, ch)
#define CHN4_MAJ_VAL(t, row, col, dep, ch) ((t)->data[DEP4_MAJ_IDX((t), (row), (col), (dep))])

typedef struct tensor4_ {
  Numeric *data;

  uint chans;
  uint depth;
  uint rows;
  uint cols;

  uint vol;

  Major maj;
} tensor4;

int tensor4_init(tensor4 *tensor, uint rows, uint cols, uint depth, uint chans, Major maj) {
  tensor->vol = rows * cols * depth * chans;

  tensor->rows = rows;
  tensor->cols = cols;
  tensor->depth = depth;
  tensor->chans = chans;
  tensor->maj = maj;
  tensor->data = (float *)malloc(tensor->vol * sizeof(Numeric));
  if (tensor->data == NULL) {
    return 1;
  }
  return 0;
}

inline uint tensor4_idx_raw(Major maj, uint rows, uint cols, uint depth, uint chans, uint row, uint col, uint dep, uint ch) {
  switch(maj) {
    case ROW_MAJ: return ROW4_MAJ_IDX_RAW(rows, cols, depth, row, col, dep, ch);
    case DEP_MAJ: return DEP4_MAJ_IDX_RAW(cols, depth, chans, row, col, dep, ch);
    case CHN_MAJ: return CHN4_MAJ_IDX_RAW(cols, depth, chans, row, col, dep, ch);
    default: printf("ERROR! GET LIBRARY\n"); return 0;
  }
}

inline uint tensor4_idx(tensor4 *t, uint row, uint col, uint dep, uint ch) {
  return tensor4_idx_raw(t->maj, t->rows, t->cols, t->depth, t->chans, row, col, dep, ch);
}


int tensor4_set_data_raw(Numeric *t, Numeric *data, Major maj, uint rows, uint cols, uint depth, uint chans) {
  uint idx = 0;

  for (uint c = 0; c < chans; c++) {
    for (uint i = 0; i < depth; i++) {
      for (uint j = 0; j < rows; j++) {
        for (uint k = 0; k < cols; k++) {
          t[tensor4_idx_raw(maj, rows, cols, depth, chans, j, k, i, c)] = data[idx++];
        }
      }
    }
  }

  return 0;
}

// Assumes input data is row major
int tensor4_set_data(tensor4 *t, Numeric *data) {
  return tensor4_set_data_raw(t->data, data, t->maj, t->rows, t->cols, t->depth, t->chans);
}


inline Numeric tensor4_val(tensor4 *t, uint row, uint col, uint dep, uint ch) {
  return t->data[tensor4_idx(t, row, col, dep, ch)];
}

void tensor4_print(tensor4 *t) {
	for (uint c = 0; c < t->chans; c++) {
	  for (uint i = 0; i < t->depth; i++) {
	    for (uint j = 0; j < t->rows; j++) {
	      for (uint k = 0; k < t->cols; k++) {
	        printf("%f, ", tensor4_val(t, j, k, i, c));
	      }
	      printf("\n");
	    }
	    printf("\n");
	  }
	  printf("\n");
	}
}

#endif