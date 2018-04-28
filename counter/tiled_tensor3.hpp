#ifndef TILED_TENSOR3_HPP
#define TILED_TENSOR3_HPP

#include "HLS/hls.h"
#include "common.hpp"
#include <stdio.h>
#include <math.h>

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

  uint size;
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

  tensor->size = tensor->tile_vol * tensor->vol_t;
  tensor->data = (Numeric *)malloc(tensor->size * sizeof(Numeric));
  if (tensor->data == NULL) {
    return 1;
  }
  return 0;
}

Numeric* tiled_tensor3_tile(tiled_tensor3 *t, uint row_t, uint col_t, uint depth_t) {
  uint idx_t = tensor3_idx_raw(t->tile_maj, t->rows_t, t->cols_t, t->depth_t, row_t, col_t, depth_t);
  return &t->data[idx_t * t->tile_vol];
}

inline uint tiled_tensor3_idx_raw(Major maj_t, uint rows_t, uint cols_t, uint depth_t, 
                                  Major tile_maj, uint tile_rows, uint tile_cols, uint tile_depth, 
                                  uint row, uint col, uint dep) {

  uint tile_vol = tile_rows * tile_cols * tile_depth;
  uint row_t_idx = INT_DIV_CEIL(row, tile_rows);
  uint col_t_idx = INT_DIV_CEIL(col, tile_cols);
  uint depth_t_idx = INT_DIV_CEIL(dep, tile_depth);

  uint idx_t = tensor3_idx_raw(maj_t, rows_t, cols_t, depth_t, row_t_idx, col_t_idx, depth_t_idx);
  uint tile_idx = tensor3_idx_raw(tile_maj, tile_rows, tile_cols, tile_depth, row % tile_rows, col % tile_cols, dep % tile_depth);
  return (idx_t * tile_vol) + tile_idx;
}

inline uint tiled_tensor3_idx(tiled_tensor3 *t, uint row, uint col, uint dep) {
  return tiled_tensor3_idx_raw(t->maj_t, t->rows_t, t->cols_t, t->depth_t, t->tile_maj, t->tile_rows, t->tile_cols, t->tile_depth, row, col, dep);
}

inline Numeric tiled_tensor3_val(tiled_tensor3 *t, uint row, uint col, uint dep) {
  return t->data[tiled_tensor3_idx(t, row, col, dep)];
}

int tiled_tensor3_set_data(tiled_tensor3 *t, Numeric *data) {
  uint idx = 0;

  for (uint i = 0; i < t->depth; i++) {
    for (uint j = 0; j < t->rows; j++) {
      for (uint k = 0; k < t->cols; k++) {
        t->data[tiled_tensor3_idx(t, j, k, i)] = data[idx++];
      }
    }
  }

  return 0;
}

void tiled_tensor3_print(tiled_tensor3 *t) {
  for (uint i = 0; i < t->depth; i++) {
    for (uint j = 0; j < t->rows; j++) {

      if (j > 0 && j % t->tile_rows == 0) {
        for (uint k = 0; k < t->cols; k++) {
          printf(" - ");
        }
        printf("\n");
      }

      for (uint k = 0; k < t->cols; k++) {
        if (k > 0 && k % t->tile_cols == 0) {
          printf(" | ");
        }
        printf("%f, ", tiled_tensor3_val(t, j, k, i));
      }
      printf("\n");
    }
    printf("\n");
  }
}

#endif