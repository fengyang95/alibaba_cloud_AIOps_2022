#!/bin/bash

privatetag=$1
tag="v$(date +%Y%m%d)"
if [ -n "$privatetag" ]; then
  tag=$tag-$privatetag
fi

echo "docker build -t aiops2022:$tag "
docker build -t aiops2022:$tag .
