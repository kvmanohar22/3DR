#!/bin/sh

cd ../build
make test_slam

SEQUENCE_PATH=/home/kv/slam/dataset/sequences/00

if [ $? -eq 0 ]; then
  ./tests/slam/test_slam ${SEQUENCE_PATH}
else
  echo "Error occured in compilation"
fi

