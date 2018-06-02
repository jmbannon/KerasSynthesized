#ifndef TENSOR_UTILS_HPP
#define TENSOR_UTILS_HPP

#include "HLS/hls.h"
#include "HLS/ac_fixed.h"

using namespace ihc;


//////////////////////////////////////////////////////////////////////////////////////////
#if FPGA_COMPILE

#include "HLS/ac_fixed_math.h"

#define NUMERIC_VAL(val) ((val).to_double())
#define UINT_VAL(val) ((val).to_long())
typedef ac_fixed<16, 8, true> Numeric;
typedef mm_master<Numeric, align<16>, latency<0>, dwidth<64> > mm_src;

bool fcompare(Numeric a, Numeric b) {
    return fabs(a.to_double() - b.to_double()) < 1e-6f;
}

#else ///////////////////////////////////////////////////////////////////////////////////////////////

#include "math.h"

#define NUMERIC_VAL(val) (val)
#define UINT_VAL(val) (val)

typedef float Numeric;
typedef mm_master<Numeric, latency<0> > mm_src;

bool fcompare(Numeric a, Numeric b) {
    return fabs(a - b) < 1e-6f;
}

#endif
//////////////////////////////////////////////////////////////////////////////////////////

#define BUFFER_SIZE 32
#define BUFFER_LOAD_PIPELINE 1

#ifndef CONVOLVER_TEST_INPUT_SIZE
#define CONVOLVER_TEST_INPUT_SIZE 5
#endif

enum Major { ROW_MAJ, COL_MAJ, DEP_MAJ, CHN_MAJ };

#define INT_DIV_CEIL(a, b) ((a) / (b) + ((a) % (b) > 0))

#endif