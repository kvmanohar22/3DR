#!/bin/sh

SEQUENCE_PATH=/home/kv/slam/dataset/sequences/02

cd ../release
make test_slam -j3

if [ $? -eq 0 ]; then
  echo "******************"
  # ./tests/slam/test_slam ${SEQUENCE_PATH}
  ./tests/slam/test_slam ${SEQUENCE_PATH} --logtostderr --v=3 /home/kv/Desktop/log
else
  echo "Compiler or linker error!"
fi
