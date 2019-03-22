#!/bin/sh

SEQUENCE_PATH=/home/kv/slam/dataset/sequences/02

cd ../build
make test_slam -j3

if [ $? -eq 0 ]; then
  echo "******************"
  ./tests/slam/test_slam ${SEQUENCE_PATH}
else
  echo "Compiler or linker error!"
fi
