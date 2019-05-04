#!/bin/sh

cd /home/kv/slam/graphics/3DR/build
make test_init_map -j3

if [ $? -eq 0 ]; then
  echo "******************"
  ../bin/test_init_map
else
  echo "Compiler or linker error!"
fi

