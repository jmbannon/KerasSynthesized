#ifndef TILED_TENSOR3_HPP
#define TILED_TENSOR3_HPP

#include "HLS/hls.h"
#include "common.hpp"
#include <stdio.h>
#include <math.h>

// ROW-COL-DEP
#define ROW3_MAJ_IDX(t, row, col, dep) (((dep) * (t)->rows * (t)->cols) + ((row) * (t)->cols) + (col))
#define ROW3_MAJ_VAL(t, row, col, dep) ((t)->data[ROW3_MAJ_IDX((t), (row), (col), (dep))])

// COL-ROW-DEP
#define COL3_MAJ_IDX(t, row, col, dep) (((dep) * (t)->rows * (t)->cols) + ((col) * (t)->rows) + (row))
#define COL3_MAJ_VAL(t, row, col, dep) ((t)->data[COL3_MAJ_IDX((t), (row), (col), (dep))])

// DEP-ROW-COL
#define DEP3_MAJ_IDX(t, row, col, dep) (((row) * (t)->cols * (t)->depth) + ((col) * (t)->depth) + (dep))
#define DEP3_MAJ_VAL(t, row, col, dep) ((t)->data[DEP3_MAJ_IDX((t), (row), (col), (dep))])

using namespace ihc;

typedef struct tiled_tensor3_ {
  Numeric *data;
  uint depth;
  uint rows;
  uint cols;

  uint vol;

  uint tile_depth;
  uint tile_rows;
  uint tile_cols;

  uint depth_t;
  uint rows_t;
  uint cols_t;

  uint tile_vol;

  uint repl;

  Major maj;
  Major tile_maj;
} tiled_tensor3;

int tiled_tensor3_init(tiled_tensor3 *tensor, uint rows, uint cols, uint depth, uint tile_depth, uint tile_rows, uint tile_cols, Major maj, Major tile_maj) {
  tensor->vol = rows * cols * depth;
  tensor->rows = rows;
  tensor->cols = cols;
  tensor->depth = depth;

  tensor->tile_rows = tile_rows;
  tensor->tile_cols = tile_cols;
  tensor->tile_depth = tile_depth;

  tensor->rows_t = INT_DIV_CEIL(rows, tile_rows);
  tensor->cols_t = INT_DIV_CEIL(cols, tile_cols);
  tensor->depth_t = INT_DIV_CEIL(depth, tile_depth);
  tensor->tile_vol = tensor->rows_t * tensor->cols_t * tensor->depth_t;


  tensor->maj = maj;
  tensor->tile_maj = tile_maj;
  tensor->data = (float *)malloc(tensor->tile_vol * sizeof(Numeric));
  if (tensor->data == NULL) {
    return 1;
  }
  return 0;
}

// Assumes input data is row major
int tiled_tensor3_set_data(tiled_tensor3 *t, Numeric *data) {
  uint idx = 0;
  switch(t->maj) {
    case ROW_MAJ:
      for (uint i = 0; i < t->depth_t; i++) {
        for (uint j = 0; j < j->rows_t; j++) {
          for (uint k = 0; k < j->cols_t; k++) {

            
            
          }
        }
      }
      for (uint i = 0; i < t->vol; i++) {
        t->data[i] = data[i];
      }
      return 0;

    case COL_MAJ:
      for (uint i = 0; i < t->depth; i++) {
        for (uint j = 0; j < t->cols; j++) {
          for (uint k = 0; k < t->rows; k++) {
            t->data[idx++] = data[ROW3_MAJ_IDX(t, k, j, i)];
          }
        }
      }
      return 0;

    case DEP_MAJ:
      for (uint i = 0; i < t->rows; i++) {
        for (uint j = 0; j < t->cols; j++) {
          for (uint k = 0; k < t->depth; k++) {
            t->data[idx++] = data[ROW3_MAJ_IDX(t, i, j, k)];
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
    case ROW_MAJ: return ROW3_MAJ_IDX(t, row, col, dep);
    case COL_MAJ: return COL3_MAJ_IDX(t, row, col, dep);
    case DEP_MAJ: return DEP3_MAJ_IDX(t, row, col, dep);
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