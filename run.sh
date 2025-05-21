#!/bin/bash

cd ~/kart-main/v2
python -m venv env
source env/bin/activate
pip install pygame requests
python kart2.py

deactivate
rm -rf env