#!/bin/bash

cd ~
echo "Сегодня: $(date)"
echo "Свободное место на диске:"
df -h / | tail -1

curl -L -o gkart.zip https://github.com/mitrichevgeorge/kart/archive/refs/heads/main.zip
unzip gkart.zip
rm -f gkart.zip

mkdir -p ~/bin
mv kart-main/run.sh ~/bin/gkart
chmod +x ~/bin/gkart

mkdir -p ~/.local/share/applications
cp kart-main/gkart.desktop ~/.local/share/applications/
chmod +x ~/.local/share/applications/gkart.desktop
cp kart-main/gkart.desktop ~/'Рабочий стол'/
chmod +x ~/Desktop/gkart.desktop

echo -e "\033c"
echo "Установка завершена"