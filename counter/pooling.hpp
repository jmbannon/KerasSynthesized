#include "HLS/hls.h"
#include "tensor3.hpp"
#include <stdio.h>
#include <math.h>

inline Numeric max_pooling_2d_2(Numeric x, Numeric y) {
  return x > y ? x : y;
}

