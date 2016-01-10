#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

PYTHONPATH=$DIR:$DIR/../ python $DIR/server.py --port=8888 --host=127.0.0.1