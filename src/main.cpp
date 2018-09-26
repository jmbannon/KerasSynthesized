#include <iostream>
#include <string>
#include <sstream>
#include "test/test_component_convolver.hpp"
#include "test/test_component_oom.hpp"

using namespace std;

string dimension_str(uint dim) {
  stringstream ss;
  ss << dim;
  return ss.str() + string("_") + ss.str();
}

int main() {
  const string test_component_3_3_convolver_name = 
    string("test_component_3_3_convolver_") 
      + dimension_str(CONVOLVER_INPUT_SIZE) 
      + string("_padding_") 
      + dimension_str(CONVOLVER_PADDING_SIZE);

  cout << test_component_3_3_convolver_name << endl;

  int ret = test_component_oom_5_5();
  if (ret == 0) {
    printf("PASSED\n");
  } else {
    printf("FAILED\n");
  }

  return 0;

}
