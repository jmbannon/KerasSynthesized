#ifndef TILED_TENSOR4_HPP
#define TILED_TENSOR4_HPP

#include "HLS/hls.h"
#include "common.hpp"
#include <stdio.h>
#include <math.h>

using namespace ihc;

typedef struct tiled_tensor3_ {
  Numeric *data;

  // Actual data dimensions
  uint chans;
  uint depth;
  uint rows;
  uint cols;
  uint vol;

  // inner tile dimensions
  uint tile_chans;
  uint tile_depth;
  uint tile_rows;
  uint tile_cols;
  uint tile_vol;
  Major tile_maj;

  // outer tile dimensions
  uint chans_t;
  uint depth_t;
  uint rows_t;
  uint cols_t;
  uint vol_t;
  Major maj_t;

  uint size;
} tiled_tensor4;

int tiled_tensor4_init(tiled_tensor4 *tensor, uint rows, uint cols, uint depth, uint chans, uint tile_depth, uint tile_rows, uint tile_cols, uint tile_chans, Major maj_t, Major tile_maj) {
  tensor->rows = rows;
  tensor->cols = cols;
  tensor->depth = depth;
  tensor->chans = chans;
  tensor->vol = rows * cols * depth * chans;

  tensor->tile_rows = tile_rows;
  tensor->tile_cols = tile_cols;
  tensor->tile_depth = tile_depth;
  tensor->tile_chans = tile_chans,
  tensor->tile_vol = tile_rows * tile_cols * tile_depth * tile_chans;
  tensor->tile_maj = tile_maj;

  tensor->rows_t = INT_DIV_CEIL(rows, tile_rows);
  tensor->cols_t = INT_DIV_CEIL(cols, tile_cols);
  tensor->depth_t = INT_DIV_CEIL(depth, tile_depth);
  tensor->chans_t = INT_DIV_CEIL(chans, tile_chans);
  tensor->vol_t = tensor->rows_t * tensor->cols_t * tensor->depth_t * tensor->chans_t;
  tensor->maj_t = maj_t;

  tensor->size = tensor->tile_vol * tensor->vol_t;
  tensor->data = (Numeric *)malloc(tensor->size * sizeof(Numeric));
  if (tensor->data == NULL) {
    return 1;
  }
  return 0;
}

inline Numeric* tiled_tensor4_tile(tiled_tensor4 *t, uint row_t, uint col_t, uint dep_t, uint ch_t) {
  uint idx_t = tensor4_idx_raw(t->tile_maj, t->rows_t, t->cols_t, t->depth_t, t->chans_t, row_t, col_t, dep_t, ch_t);
  return &t->data[idx_t * t->tile_vol];
}

inline uint tiled_tensor4_idx_raw(Major maj_t, uint rows_t, uint cols_t, uint depth_t, uint chans_t,
                                  Major tile_maj, uint tile_rows, uint tile_cols, uint tile_depth, uint tile_chans,
                                  uint row, uint col, uint dep, uint ch) {

  uint tile_vol = tile_rows * tile_cols * tile_depth * tile_chans;
  uint row_t = INT_DIV_CEIL(row, tile_rows);
  uint col_t = INT_DIV_CEIL(col, tile_cols);
  uint dep_t = INT_DIV_CEIL(dep, tile_depth);
  uint ch_t = INT_DIV_CEIL(ch, tile_chans);

  uint idx_t = tensor4_idx_raw(maj_t, rows_t, cols_t, depth_t, chans_t, row_t, col_t, dep_t, ch_t);
  uint tile_idx = tensor4_idx_raw(tile_maj, tile_rows, tile_cols, tile_depth, tile_chans, row % tile_rows, col % tile_cols, dep % tile_depth, ch % tile_chans);
  return (idx_t * tile_vol) + tile_idx;
}

inline uint tiled_tensor4_idx(tiled_tensor4 *t, uint row, uint col, uint dep, uint ch) {
  return tiled_tensor4_idx_raw(t->maj_t, t->rows_t, t->cols_t, t->depth_t, t->chans_t, t->tile_maj, t->tile_rows, t->tile_cols, t->tile_depth, t->tile_chans, row, col, dep, ch);
}

inline Numeric tiled_tensor4_val(tiled_tensor4 *t, uint row, uint col, uint dep, uint ch) {
  return t->data[tiled_tensor4_idx(t, row, col, dep, ch)];
}

int tiled_tensor4_set_data(tiled_tensor4 *t, Numeric *data) {
  uint idx = 0;

  for (uint c = 0; c < t->chans; c++) {
    for (uint i = 0; i < t->depth; i++) {
      for (uint j = 0; j < t->rows; j++) {
        for (uint k = 0; k < t->cols; k++) {
          t->data[tiled_tensor4_idx(t, j, k, i, c)] = data[idx++];
        }
      }
    }
  }

  return 0;
}

void tiled_tensor4_print(tiled_tensor4 *t) {
  for (uint c = 0; c < t->chans; c++) {
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
    printf("\n");
  }
}

#endif