#!/bin/bash

cd ~/kart-main
python -m venv env
source env/bin/activate
pip install pygame requests
python kart4.py

deactivate
rm -rf env