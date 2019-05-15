#!/bin/sh

export GLOG_logtostderr=1
cd /home/kv/slam/graphics/3DR/build
make test_init_map_generalized -j3
if [ $? -eq 0 ]; then
  echo "\n\n"
  ../bin/test_init_map_generalized --logtostderr=1
else
  echo "Compiler or linker error!"
fi

