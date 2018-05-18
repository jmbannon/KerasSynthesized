#ifndef TENSOR3_HPP
#define TENSOR3_HPP

#include "HLS/hls.h"
#include "common.hpp"
#include <stdio.h>
#include <math.h>

// ROW-COL-DEP
#define ROW3_MAJ_IDX_RAW(rows, cols, row, col, dep) (((dep) * (rows) * (cols)) + ((row) * (cols)) + (col))
#define ROW3_MAJ_IDX(t, row, col, dep) (ROW3_MAJ_IDX_RAW((t)->rows, (t)->cols, (row), (col), (dep)))
#define ROW3_MAJ_VAL(t, row, col, dep) ((t)->data[ROW3_MAJ_IDX((t), (row), (col), (dep))])

// COL-ROW-DEP
#define COL3_MAJ_IDX_RAW(rows, cols, row, col, dep) (((dep) * (rows) * (cols)) + ((col) * (rows)) + (row))
#define COL3_MAJ_IDX(t, row, col, dep) (COL3_MAJ_IDX_RAW((t)->rows, (t)->cols, (row), (col), (dep)))
#define COL3_MAJ_VAL(t, row, col, dep) ((t)->data[COL3_MAJ_IDX((t), (row), (col), (dep))])

// DEP-ROW-COL
#define DEP3_MAJ_IDX_RAW(cols, depth, row, col, dep) (((row) * (cols) * (depth)) + ((col) * (depth)) + (dep))
#define DEP3_MAJ_IDX(t, row, col, dep) (DEP3_MAJ_IDX_RAW((t)->cols, (t)->depth, (row), (col), (dep)))
#define DEP3_MAJ_VAL(t, row, col, dep) ((t)->data[DEP3_MAJ_IDX((t), (row), (col), (dep))])

using namespace ihc;

typedef struct tensor3_ {
  Numeric *data;
  uint depth;
  uint rows;
  uint cols;
  uint vol;

  Major maj;
} tensor3;

int tensor3_init(tensor3 *tensor, uint rows, uint cols, uint depth, Major maj) {
  tensor->vol = rows * cols * depth;
  tensor->rows = rows;
  tensor->cols = cols;
  tensor->depth = depth;
  tensor->maj = maj;
  tensor->data = (Numeric *)malloc(tensor->vol * sizeof(Numeric));
  if (tensor->data == NULL) {
    return 1;
  }
  return 0;
}

int tensor3_init_padding(tensor3 *tensor, uint rows, uint cols, uint depth, Major maj, uint paddingY, uint paddingX) {
  return tensor3_init(tensor, rows + (2 * paddingY), cols + (2 * paddingX), depth, maj);
}

inline uint tensor3_idx_raw(Major maj, uint rows, uint cols, uint depth, uint row, uint col, uint dep) {
  switch(maj) {
    case ROW_MAJ: return ROW3_MAJ_IDX_RAW(rows, cols, row, col, dep);
    case COL_MAJ: return COL3_MAJ_IDX_RAW(rows, cols, row, col, dep);
    case DEP_MAJ: return DEP3_MAJ_IDX_RAW(cols, depth, row, col, dep);
    default: printf("ERROR! GET LIBRARY\n"); return 0;
  }
}

inline uint tensor3_idx(tensor3 *t, uint row, uint col, uint dep) {
  return tensor3_idx_raw(t->maj, t->rows, t->cols, t->depth, row, col, dep);
}

inline Numeric tensor3_val_raw(Numeric *t, Major maj, uint rows, uint cols, uint depth, uint row, uint col, uint dep) {
  return t[tensor3_idx_raw(maj, rows, cols, depth, row, col, dep)];
}

inline Numeric tensor3_val(tensor3 *t, uint row, uint col, uint dep) {
  return tensor3_val_raw(t->data, t->maj, t->rows, t->cols, t->depth, row, col, dep);
}


int tensor3_set_data_raw(Numeric *t, Numeric *data, Major maj, uint rows, uint cols, uint depth, uint paddingY, uint paddingX) {
  uint idx = 0;
  uint vol = rows * cols * depth;

  for (uint i = 0; i < depth; i++) {
    // Pads top rows
    for (uint j = 0; j < paddingY; j++) {
      for (uint k = 0; k < cols; k++) {
        t[tensor3_idx_raw(maj, rows, cols, depth, j, k, i)] = 0.;
      }
    }

    // Pads columns and sets data
    for (uint j = paddingY; j < (rows - paddingY); j++) {
      for (uint k = 0; k < paddingX; k++) {
        t[tensor3_idx_raw(maj, rows, cols, depth, j, k, i)] = 0.;
      }
      for (uint k = paddingX; k < (cols - paddingX); k++) {
        t[tensor3_idx_raw(maj, rows, cols, depth, j, k, i)] = data[idx++];
      }
      for (uint k = (cols - paddingX); k < cols; k++) {
        t[tensor3_idx_raw(maj, rows, cols, depth, j, k, i)] = 0.;
      }
    }

    // Pads bottom rows
    for (uint j = (rows - paddingY); j < rows; j++) {
      for (uint k = 0; k < cols; k++) {
        t[tensor3_idx_raw(maj, rows, cols, depth, j, k, i)] = 0.;
      }
    }
  }

  return 0;
}

// Assumes input data is row major
int tensor3_set_data(tensor3 *t, Numeric *data) {
  return tensor3_set_data_raw(t->data, data, t->maj, t->rows, t->cols, t->depth, 0, 0);
}

int tensor3_set_data_padding(tensor3 *t, Numeric *data, uint paddingX, uint paddingY) {
  return tensor3_set_data_raw(t->data, data, t->maj, t->rows, t->cols, t->depth, paddingX, paddingY);
}

void tensor3_print(tensor3 *t) {
  for (uint i = 0; i < t->depth; i++) {
    for (uint j = 0; j < t->rows; j++) {
      for (uint k = 0; k < t->cols; k++) {
        printf("%f, ", NUMERIC_VAL(tensor3_val(t, j, k, i)));
      }
      printf("\n");
    }
    printf("\n");
  }
}

#endif