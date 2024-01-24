export PATH=/usr/local/cuda-11.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.1/lib64:$LD_LIBRARY_PATH
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
#/home/inspur/hxw/cmake/cmake-3.27.2-linux-x86_64/bin/cmake 
#/home/inspur/hxw/helib_pack