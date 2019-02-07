#!/bin/sh

python main.py train ./config/market_softmax.yaml
python main.py train ./config/market_triplet.yaml
python main.py train ./config/market_softmax_triplet.yaml

python main.py train ./config/dukemtmc_softmax.yaml
python main.py train ./config/dukemtmc_triplet.yaml
python main.py train ./config/dukemtmc_softmax_triplet.yaml