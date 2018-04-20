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

  // Actual data dimensions
  uint depth;
  uint rows;
  uint cols;
  uint vol;

  // inner tile dimensions
  uint tile_depth;
  uint tile_rows;
  uint tile_cols;
  uint tile_vol;
  Major tile_maj;

  // outer tile dimensions
  uint depth_t;
  uint rows_t;
  uint cols_t;
  uint vol_t;
  Major maj_t;

  uint repl;
} tiled_tensor3;

int tiled_tensor3_init(tiled_tensor3 *tensor, uint rows, uint cols, uint depth, uint tile_depth, uint tile_rows, uint tile_cols, Major maj_t, Major tile_maj) {
  tensor->rows = rows;
  tensor->cols = cols;
  tensor->depth = depth;
  tensor->vol = rows * cols * depth;

  tensor->tile_rows = tile_rows;
  tensor->tile_cols = tile_cols;
  tensor->tile_depth = tile_depth;
  tensor->tile_vol = tile_rows * tile_cols * tile_depth;
  tensor->tile_maj = tile_maj;

  tensor->rows_t = INT_DIV_CEIL(rows, tile_rows);
  tensor->cols_t = INT_DIV_CEIL(cols, tile_cols);
  tensor->depth_t = INT_DIV_CEIL(depth, tile_depth);
  tensor->vol_t = tensor->rows_t * tensor->cols_t * tensor->depth_t;
  tensor->maj_t = maj_t;

  tensor->data = (Numeric *)malloc(tensor->tile_vol * tensor->vol_t * sizeof(Numeric));
  if (tensor->data == NULL) {
    return 1;
  }
  return 0;
}

Numeric* tiled_tensor3_tile(tiled_tensor3 *t, uint row_t, uint col_t, uint dep_t) {
  uint tile_idx = tensor3_idx_raw(t->tile_maj, t->rows_t, t->cols_t, t->depth_t, row_t, col_t, dep_t);
  return &t->data[tile_idx * t->tile_vol];
}

// Assumes input data is row major
int tiled_tensor3_set_data(tiled_tensor3 *t, Numeric *data) {
  uint idx = 0;
  switch(t->maj) {
    case ROW_MAJ:
      for (uint i = 0; i < t->depth_t; i++) {
        for (uint j = 0; j < j->rows_t; j++) {
          for (uint k = 0; k < j->cols_t; k++) {

            Numeric *tile = tiled_tensor3_tile(t, j, k, i);
            for (uint ix = 0; ix < t->tile_depth; ix++) {
              for (uint jx = 0; jx < t->tile_rows; jx++) {
                for (uint kx = 0; kx < t->tile_cols; kx++) {
                  tile[]
                }
              }
            }

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