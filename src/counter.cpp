#include <iostream>
#include <string>
#include <sstream>
#include "test/test_component_convolver.hpp"

using namespace std;

int main() {
  stringstream ss;
  ss << CONVOLVER_TEST_INPUT_SIZE;
  string conv_input_size = ss.str();
  const string test_name = string("test_component_3_3_convolver_") + conv_input_size + string("_") + conv_input_size;
  cout << test_name << endl;

  int ret = test_component_3_3_convolver_variable();

  if (ret == 0) {
    printf("PASSED\n");
  } else {
    printf("FAILED\n");
  }

  return 0;

}
