#!/bin/sh

SEQUENCE_PATH=/home/kv/slam/datasets/seq00even
export GLOG_logtostderr=1

cd /home/kv/slam/graphics/3DR/build
make test_pipeline -j3

if [ $? -eq 0 ]; then
  echo "\n\n"
  ../bin/test_pipeline ${SEQUENCE_PATH} --logtostderr=1
else
  echo "Compiler or linker error!"
fi

