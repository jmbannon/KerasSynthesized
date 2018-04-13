#ifndef TENSOR3_HPP
#define TENSOR3_HPP

#include "HLS/hls.h"
#include <stdio.h>
#include <math.h>

typedef float Numeric;

enum Major { ROW_MAJ, COL_MAJ, DEP_MAJ, CHN_MAJ };

#define ROW_MAJ_IDX(t, row, col, dep) (((dep) * (t)->rows * (t)->cols) + ((row) * (t)->cols) + (col))
#define ROW_MAJ_VAL(t, row, col, dep) ((t)->data[ROW_MAJ_IDX((t), (row), (col), (dep))])

#define COL_MAJ_IDX(t, row, col, dep) (((dep) * (t)->rows * (t)->cols) + ((col) * (t)->rows) + (row))
#define COL_MAJ_VAL(t, row, col, dep) ((t)->data[COL_MAJ_IDX((t), (row), (col), (dep))])

// dep-maj is followed by 2d row-maj
#define DEP_MAJ_IDX(t, row, col, dep) (((row) * (t)->cols * (t)->depth) + ((col) * (t)->depth) + (dep))
#define DEP_MAJ_VAL(t, row, col, dep) ((t)->data[DEP_MAJ_IDX((t), (row), (col), (dep))])

using namespace ihc;

typedef struct tensor3_ {
  Numeric *data;
  uint depth;
  uint rows;
  uint cols;
  uint vol;

  Major maj;
} tensor3;

typedef struct tiled_tensor3_ {
  Numeric *data;
  uint depth;
  uint rows;
  uint cols;
  uint vol;

  uint h_tile;
  uint w_tile;
  uint d_tile;

  uint repl;

  Major maj;
  Major tile_maj;
} tiled_tensor3;

int tensor3_init(tensor3 *tensor, uint rows, uint cols, uint depth, Major maj) {
  tensor->vol = rows * cols * depth;
  tensor->rows = rows;
  tensor->cols = cols;
  tensor->depth = depth;
  tensor->maj = maj;
  tensor->data = (float *)malloc(tensor->vol * sizeof(Numeric));
  if (tensor->data == NULL) {
    return 1;
  }
  return 0;
}

// Assumes input data is row major
int tensor3_set_data(tensor3 *t, Numeric *data) {
  uint idx = 0;
  switch(t->maj) {
    case ROW_MAJ:
      for (uint i = 0; i < t->vol; i++) {
        t->data[i] = data[i];
      }
      return 0;

    case COL_MAJ:
      for (uint i = 0; i < t->depth; i++) {
        for (uint j = 0; j < t->cols; j++) {
          for (uint k = 0; k < t->rows; k++) {
            t->data[idx++] = data[ROW_MAJ_IDX(t, k, j, i)];
          }
        }
      }
      return 0;

    case DEP_MAJ:
      for (uint i = 0; i < t->rows; i++) {
        for (uint j = 0; j < t->cols; j++) {
          for (uint k = 0; k < t->depth; k++) {
            t->data[idx++] = data[ROW_MAJ_IDX(t, i, j, k)];
          }
        }
      }
      return 0;
    default:
      return 1;
  }
}

inline uint tensor3_idx(tensor3 *t, uint row, uint col, uint dep) {
  switch(t->maj) {
    case ROW_MAJ: return ROW_MAJ_IDX(t, row, col, dep);
    case COL_MAJ: return COL_MAJ_IDX(t, row, col, dep);
    case DEP_MAJ: return DEP_MAJ_IDX(t, row, col, dep);
    default: printf("ERROR! GET LIBRARY\n"); return 0;
  }
}

inline Numeric tensor3_val(tensor3 *t, uint row, uint col, uint dep) {
  return t->data[tensor3_idx(t, row, col, dep)];
}

void tensor3_print(tensor3 *t) {
  for (uint i = 0; i < t->depth; i++) {
    for (uint j = 0; j < t->rows; j++) {
      for (uint k = 0; k < t->cols; k++) {
        printf("%f, ", tensor3_val(t, j, k, i));
      }
      printf("\n");
    }
    printf("\n");
  }
}

#endif