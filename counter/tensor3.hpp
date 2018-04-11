#ifndef TENSOR3_HPP
#define TENSOR3_HPP

#include "HLS/hls.h"
#include <stdio.h>
#include <math.h>

typedef float Numeric;

enum Major { ROW_MAJ, COL_MAJ, DEP_MAJ };

#define ROW_MAJ_IDX(t, row, col, dep) (((dep) * (t)->rows * (t)->cols) + ((row) * (t)->cols) + (col))
#define ROW_MAJ_VAL(t, row, col, dep) ((t)->data[ROW_MAJ_IDX((t), (row), (col), (dep))])

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
            t->data[idx++] = data[(i * t->rows * t->cols) + (k * t->cols) + j];
          }
        }
      }
      return 0;

    case DEP_MAJ:
      for (uint i = 0; i < t->rows; i++) {
        for (uint j = 0; j < t->cols; j++) {
          for (uint k = 0; k < t->depth; k++) {
            t->data[idx++] = data[(k * t->rows * t->cols) + (i * t->cols) + j];
          }
        }
      }
      return 0;
    default:
      return 1;
  }
}

void tensor3_print(tensor3 *t) {
  for (uint i = 0; i < t->depth; i++) {
    for (uint j = 0; j < t->rows; j++) {
      for (uint k = 0; k < t->cols; k++) {
        printf("%f, ", ROW_MAJ_VAL(t, j, k, i));
      }
      printf("\n");
    }
    printf("\n");
  }
}

#endif