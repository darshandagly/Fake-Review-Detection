#!/bin/sh
pip3 install pandas
pip3 install nltk
pip3 install numpy
pip3 install sklearn
pip3 install matplotlib
pip3 install seaborn
pip3 install tqdm
python3 -m nltk.downloader stopwords
chmod 777 main.py
python3 main.py