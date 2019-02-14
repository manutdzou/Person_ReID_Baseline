#!/bin/sh

python main.py train ./config/cuhk03_softmax.yaml
python main.py train ./config/cuhk03_softmax_triplet.yaml

python main.py test ./config/cuhk03_softmax.yaml
python main.py test ./config/cuhk03_softmax_triplet.yaml

python main.py test ./config/cuhk03_softmax.yaml True
python main.py test ./config/cuhk03_softmax_triplet.yaml True

