#!/bin/bash

echo "Сегодня: $(date)"
echo "Свободное место на диске:"
df -h / | tail -1

curl -L -o gkart.zip https://github.com/mitrichevgeorge/kart/archive/refs/heads/main.zip
unzip gkart.zip
rm -f gkart.zip
cd kart-main
