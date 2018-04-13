#ifndef KERNEL_HPP
#define KERNEL_HPP

#include "HLS/hls.h"
#include <stdio.h>
#include <math.h>

typedef struct kernel_ {
  Numeric *data;

  uint channels;
  uint depth;
  uint rows;
  uint cols;
  uint vol;

  Major maj;
} kernel;

#endif