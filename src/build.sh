make clean
make -f makefile.cuda

g++ -o flappie -std=c++11 -I/usr/include/hdf5/serial  -I/usr/local/cuda-10.2/include -std=c++11 -Wno-sign-compare -march=native  -Wno-format  -g  -fpermissive   -DUSE_SSE2  -D__USE_MISC -D_POSIX_SOURCE -DNDEBUG -o flappie decode.cpp layers.cpp networks.cpp nnfeatures.cpp flappie_common.cpp flappie_matrix.cpp flappie_output.cpp flappie_structures.cpp util.cpp fast5_interface.cpp flappie.cpp grugpu.o  -L/usr/local/cuda-10.2/lib64 -lhdf5_serial -lpthread -lblas -lcublas -lcudart
