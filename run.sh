#!/bin/bash

cd ~/kart-main
python -m venv env
source env/bin/activate
python kart4.py

deactivate
rm -rf env