#!/bin/bash

if [ ! -d "/jsonschemabench" ]; then
  git clone https://github.com/epfl-dlab/jsonschemabench.git /jsonschemabench
else
  cd /jsonschemabench && git pull
fi

conda activate default
