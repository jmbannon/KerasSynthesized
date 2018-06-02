#ifndef VECTOR_HPP
#define VECTOR_HPP

#include "common.hpp"

#define V_VAL(v, i) ((v)->data[i])

typedef struct vector_ {
  Numeric *data;
  uint size;
} vector;

int vector_init(vector *v, uint size) {
  v->size = size;
  v->data = (float *)malloc(size * sizeof(Numeric));
  if (v->data == NULL) {
    return 1;
  }
  return 0;
}

// Assumes input data is row major
int vector_set_data(vector *v, Numeric *data) {
  for (uint i = 0; i < v->size; i++) {
    v->data[i] = data[i];
  }
  return 0;
}

void vector_print(vector *v) {
  for (uint i = 0; i < v->size; i++) {
    printf("%f, ", V_VAL(v, i));
  }
  printf("\n");
}

#endif