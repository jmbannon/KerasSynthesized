#ifndef TENSOR_UTILS_HPP
#define TENSOR_UTILS_HPP

typedef float Numeric;

enum Major { ROW_MAJ, COL_MAJ, DEP_MAJ, CHN_MAJ };

#define INT_DIV_CEIL(a, b) ((a) / (b) + ((a) % (b) > 0))

#endif