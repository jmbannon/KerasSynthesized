#include "test/test_component_convolver.hpp"

int main() {
  int ret = test_component_convolver_5_5();

  if (ret == 0) {
    printf("PASSED\n");
  } else {
    printf("FAILED\n");
  }

  return 0;

}
