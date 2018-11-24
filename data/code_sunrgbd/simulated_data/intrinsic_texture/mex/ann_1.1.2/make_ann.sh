#! /bin/bash

make realclean
make linux-g++
cp lib/libANN.a ..
cd ..
matlab -r compile
