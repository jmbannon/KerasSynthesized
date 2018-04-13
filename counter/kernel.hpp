#ifndef KERNEL_HPP
#define KERNEL_HPP

#include "HLS/hls.h"
#include "tensor3.hpp"
#include <stdio.h>
#include <math.h>


// ROW-COL-DEP-CHAN
#define T4_ROW_MAJ_IDX(t, row, col, dep, ch) (((ch) * (t)->depth * (t)->rows * (t)->cols) + ((dep) * (t)->rows * (t)->cols) + ((row) * (t)->cols) + (col))
#define T4_ROW_MAJ_VAL(t, row, col, dep, ch) ((t)->data[T4_ROW_MAJ_IDX((t), (row), (col), (dep), (ch))])

// DEP-CHAN-ROW-COL
// ((col * t->rows * t->chans * t->depth) + (row * t->chans * t->depth) + (dep * t->chans) + ch
#define T4_DEP_MAJ_IDX(t, row, col, dep, ch) (((row) * (t)->cols * (t)->chans * (t)->depth) + ((col) * (t)->chans * (t)->depth) + ((ch) * (t)->depth) + (dep))
#define T4_DEP_MAJ_VAL(t, row, col, dep, ch) ((t)->data[T4_DEP_MAJ_IDX((t), (row), (col), (dep))])

typedef struct tensor4_ {
  Numeric *data;

  uint chans;
  uint depth;
  uint rows;
  uint cols;

  uint vol;
  uint size;

  Major maj;
} tensor4;

int tensor4_init(tensor4 *tensor, uint rows, uint cols, uint depth, uint chans, Major maj) {
  tensor->vol = rows * cols * depth;
  tensor->size = tensor->vol * chans;

  tensor->rows = rows;
  tensor->cols = cols;
  tensor->depth = depth;
  tensor->chans = chans;
  tensor->maj = maj;
  tensor->data = (float *)malloc(tensor->size * sizeof(Numeric));
  if (tensor->data == NULL) {
    return 1;
  }
  return 0;
}

// Assumes input data is row major
int tensor4_set_data(tensor4 *t, Numeric *data) {
  uint idx = 0;
  switch(t->maj) {
    case ROW_MAJ:
      for (uint i = 0; i < t->size; i++) {
        t->data[i] = data[i];
      }
      return 0;
	case DEP_MAJ:
      for (uint i = 0; i < t->rows; i++) {
        for (uint j = 0; j < t->cols; j++) {
          for (uint k = 0; k < t->chans; k++) {
          	for (uint l = 0; l < t->depth; l++) {
              t->data[idx++] = data[T4_ROW_MAJ_IDX(t, i, j, l, k)];
        	}
          }
        }
      }
      return 0;
    case CHN_MAJ:
      for (uint i = 0; i < t->rows; i++) {
        for (uint j = 0; j < t->cols; j++) {
          for (uint k = 0; k < t->depth; k++) {
          	for (uint l = 0; l < t->chans; l++) {
              t->data[idx++] = data[T4_ROW_MAJ_IDX(t, i, j, k, l)];
        	}
          }
        }
      }
      return 0;
    default:
      return 1;
  }
}

#endif