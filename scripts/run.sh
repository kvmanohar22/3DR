#!/bin/sh

cd ../build
make slam

if [ $? -eq 0 ]; then
  ./slam
else
  echo "Error occured in compilation"
fi

