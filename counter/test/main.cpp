#include <iostream>
#include <string>
#include "test_tensor3.hpp"
#include "test_tensor4.hpp"
#include "test_tiled_tensor3.hpp"

using namespace std;

#define MAX_TEST_NAME 40
#define MAX_NUM_TESTS 100

typedef struct _test {
	string name;
	int (*test_function)();
} unittest;

unittest tests[MAX_NUM_TESTS];
int test_count = 0;

void unittest_add(string name, 
			      int (*test_function)())
{
	unittest test = { .name = name, .test_function = test_function };
	tests[test_count++] = test;
}

void unittest_add_all() {
	unittest_add("test_tensor3_row", test_tensor3_row);
	unittest_add("test_tensor3_col", test_tensor3_col);
	unittest_add("test_tensor3_dep", test_tensor3_dep);

	unittest_add("test_tensor4_row", test_tensor3_row);
	unittest_add("test_tensor4_dep", test_tensor3_col);
	unittest_add("test_tensor4_chn", test_tensor3_dep);

	unittest_add("test_tensor3_row_row", test_tensor3_row_row);
	unittest_add("test_tensor3_col_col", test_tensor3_col_col);
	unittest_add("test_tensor3_dep_dep", test_tensor3_dep_dep);
}

int main() {
	unittest_add_all();

	int res = 0;
	int failures = 0;
	for (int i = 0; i < test_count; i++) {
		res = (*tests[i].test_function) ();
		if (res != 0) {
			cout << "FAILURE  -  " << tests[i].name << endl;
			++failures;
		} else {
			cout << "SUCCESS  -  " << tests[i].name << endl;
		}
	}
	if (failures > 0) {
		cout << "Number of test failures: " << failures << endl;
	} else {
		cout << "All tests passed!" << endl;
	}
	return 0;
}
