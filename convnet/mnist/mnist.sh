#!/usr/bin/env bash

PYTHONPATH=.:../.. python -u mnist.py 2>&1 | tee convnet.log