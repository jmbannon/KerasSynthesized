SOURCE_MAIN   := main.cpp
TEST_MAIN     := test/main.cpp
SOURCE_FILES  := 

HLS_CXX_FLAGS := 
CXX := i++
override CXXFLAGS := $(CXXFLAGS) --fp-relaxed -I/usr/include/c++/4.4.7 -I/usr/include/c++/4.4.7/x86_64-linux-gnu -L/home/jb/intelFPGA_lite/17.1/hls/linux64/lib/dspba/linux64
VERBOSE := 1

ifeq ($(CONVOLVER_INPUT_SIZE),)
CONVOLVER_INPUT_SIZE := 5
endif

ifeq ($(CONVOLVER_PADDING_SIZE),)
CONVOLVER_PADDING_SIZE := 0
endif

ifeq ($(CONVOLVER_PADDING_SIZE),)
CONVOLVER_PADDING_SIZE := 0
endif


TEST_FLAGS := -D CONVOLVER_INPUT_SIZE=$(CONVOLVER_INPUT_SIZE) -D CONVOLVER_PADDING_SIZE=$(CONVOLVER_PADDING_SIZE)
TEST_X86_FLAGS := -D FPGA_COMPILE=0 $(TEST_FLAGS)
TEST_FPGA_FLAGS := -D FPGA_COMPILE=1 $(TEST_FLAGS)

ifneq ($(COMPONENT),)
  TEST_FPGA_FLAGS := --component $(COMPONENT) -D FPGA_COMPILE=1 $(TEST_FLAGS)
endif



# OS-dependant tools
ifeq ($(OS),Windows_NT)
  RM  := rd /S /Q
else
  RM  := rm -rfv
endif

ifeq ($(MAKECMDGOALS),)
  $(info No target specified, defaulting to test-x86-64)
  $(info Available targets: test-x86-64, test-fpga, test-gpp, clean)
endif

# Any tools installed with HLS can be found relative to the location of i++
HLS_INSTALL_DIR := $(shell which i++ | sed 's|/bin/i++||g')

# Run the i++ x86 test by default
.PHONY: default
default: test-x86

# Run the testbench and the component as a regular program
.PHONY: test-x86
test-x86: CXXFLAGS := $(CXXFLAGS) $(HLS_CXX_FLAGS) $(TEST_X86_FLAGS) -march=x86-64 -o test-x86-64
test-x86: $(SOURCE_FILES) $(TEST_MAIN)
	$(CXX) -v -g --debug-log $(SOURCE_FILES) $(TEST_MAIN) $(CXXFLAGS)
	@echo "+----------------------------------------+"
	@echo "| Run ./test-x86-64 to execute the test. |"
	@echo "+----------------------------------------+"

# Run a simulation with the C testbench and verilog component
.PHONY: test-fpga
ifeq ($(VERBOSE),1)
  test-fpga: CXXFLAGS := $(CXXFLAGS) -v
endif
test-fpga: CXXFLAGS := $(CXXFLAGS) $(HLS_CXX_FLAGS) $(TEST_FPGA_FLAGS) -march="Cyclone V" -o test-fpga
test-fpga: $(SOURCE_FILES) $(SOURCE_MAIN)
	$(CXX) -v -g --debug-log $(SOURCE_FILES) $(SOURCE_MAIN) $(CXXFLAGS)
	@echo "+--------------------------------------+"
	@echo "| Run ./test-fpga to execute the test. |"
	@echo "+--------------------------------------+"

# Run the testbench and the component as a regular program
.PHONY: test-fpga-x86
test-fpga-x86: CXXFLAGS := $(CXXFLAGS) $(HLS_CXX_FLAGS) $(TEST_FPGA_FLAGS) -march=x86-64 -o test-x86-64
test-fpga-x86: $(SOURCE_FILES) $(SOURCE_MAIN)
	$(CXX) -v -g --debug-log $(SOURCE_FILES) $(SOURCE_MAIN) $(CXXFLAGS)
	@echo "+----------------------------------------+"
	@echo "| Run ./test-x86-64 to execute the test. |"
	@echo "+----------------------------------------+"

# Clean up temprary and delivered files
CLEAN_FILES := test-gpp \
               test-gpp.prj \
               test-fpga \
               test-fpga.prj \
               test-x86-64 \
               test-x86-64.prj
clean:
	-$(RM) $(CLEAN_FILES)
