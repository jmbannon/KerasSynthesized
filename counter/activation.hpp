#include "HLS/hls.h"
#include "tensor3.hpp"
#include <stdio.h>
#include <math.h>

inline Numeric activation_relu(Numeric x) {
  return x > 0.0 ? x : 0.0;
}

