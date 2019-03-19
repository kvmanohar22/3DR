#!/bin/sh

SEQUENCE_PATH=/home/kv/slam/dataset/sequences/00

cd ../build
make test_slam

if [ $? -eq 0 ]; then
  echo "******************"
  ./tests/slam/test_slam ${SEQUENCE_PATH}
else
  echo "Error occured in compilation"
fi

