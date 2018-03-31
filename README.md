# KerasSynthesized
An API that translates Keras models to Intel HLS compatible FPGA implementations

## Dev Environment Setup Ubuntu 16.04 LTS

### Software
Install [Quartus Prime Lite 17.1](http://dl.altera.com/?edition=lite&platform=linux&download_manager=direct) with ModelSim and devices you intend to use.

### Libraries
#### GCC
Requires GCC/G++ 4.4 and 4.7. Use `update-alternatives` to interchange between GCC/G++ versions.

Add the following to `/etc/apt/sources.list`:
```
deb http://dk.archive.ubuntu.com/ubuntu/ trusty main universe
deb http://dk.archive.ubuntu.com/ubuntu/ trusty-updates main universe
```

Install with the following commands:
```
sudo apt-get update
sudo apt-get install gcc-4.4 g++-4.4
sudo apt-get install gcc-4.7 g++-4.7

sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-5 100 --slave /usr/bin/g++ g++ /usr/bin/g++-5
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.7 60 --slave /usr/bin/g++ g++ /usr/bin/g++-4.7
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.4 50 --slave /usr/bin/g++ g++ /usr/bin/g++-4.4

sudo update-alternatives --config gcc
```
#### HDF5
Install HDF5 using the `2. Quick Installation` method [found here](https://support.hdfgroup.org/ftp/HDF5/current/src/unpacked/release_docs/INSTALL).

### Compile Paths
Run the following before compiling:
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/hdf5/lib
export PATH=$PATH:/path/to/intelFPGA_lite/17.1/modelsim_ase/linux
source /path/to/intelFPGA_lite/17.1/init_hls.sh
```

### Compile Arguments
Include the following arguments when compiling source code:
```
-I/usr/local/hdf5/include -I/usr/include/c++/4.4.7 -I/usr/include/c++/4.4.7/x86_64-linux-gnu -L/path/to/intelFPGA_lite/17.1/hls/linux64/lib/dspba/linux64 -L/usr/local/hdf5/lib -lhdf5_hl -lhdf5
```
