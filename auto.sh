#!/bin/sh

# python train.py ./config/cuhk_softmax.yaml
# python train.py ./config/cuhk_softmax_triplet.yaml

# python test.py ./config/cuhk_softmax.yaml
# python test.py ./config/cuhk_softmax_triplet.yaml

# python test.py ./config/cuhk_softmax.yaml True
# python test.py ./config/cuhk_softmax_triplet.yaml True

python test.py ./config/market_softmax.yaml
python test.py ./config/market_softmax_triplet.yaml

python test.py ./config/market_softmax.yaml True
python test.py ./config/market_softmax_triplet.yaml True

python test.py ./config/duke_softmax.yaml
python test.py ./config/duke_softmax_triplet.yaml

python test.py ./config/duke_softmax.yaml True
python test.py ./config/duke_softmax_triplet.yaml True

python test.py ./config/ntu_softmax.yaml
python test.py ./config/ntu_softmax_triplet.yaml

python test.py ./config/ntu_softmax.yaml True
python test.py ./config/ntu_softmax_triplet.yaml True

python test.py ./config/msmt_softmax.yaml
python test.py ./config/msmt_softmax_triplet.yaml

python test_cross_dataset.py ./config/ntu_softmax.yaml Market1501
python test_cross_dataset.py ./config/ntu_softmax_triplet.yaml Market1501
python test_cross_dataset.py ./config/ntu_softmax.yaml DukeMTMC
python test_cross_dataset.py ./config/ntu_softmax_triplet.yaml DukeMTMC
python test_cross_dataset.py ./config/ntu_softmax.yaml CUHK03
python test_cross_dataset.py ./config/ntu_softmax_triplet.yaml CUHK03

python test_cross_dataset.py ./config/market_softmax.yaml NTUCampus
python test_cross_dataset.py ./config/market_softmax_triplet.yaml NTUCampus
python test_cross_dataset.py ./config/market_softmax.yaml DukeMTMC
python test_cross_dataset.py ./config/market_softmax_triplet.yaml DukeMTMC
python test_cross_dataset.py ./config/market_softmax.yaml CUHK03
python test_cross_dataset.py ./config/market_softmax_triplet.yaml CUHK03

python test_cross_dataset.py ./config/duke_softmax.yaml NTUCampus
python test_cross_dataset.py ./config/duke_softmax_triplet.yaml NTUCampus
python test_cross_dataset.py ./config/duke_softmax.yaml Market1501
python test_cross_dataset.py ./config/duke_softmax_triplet.yaml Market1501
python test_cross_dataset.py ./config/duke_softmax.yaml CUHK03
python test_cross_dataset.py ./config/duke_softmax_triplet.yaml CUHK03

python test_cross_dataset.py ./config/cuhk_softmax.yaml NTUCampus
python test_cross_dataset.py ./config/cuhk_softmax_triplet.yaml NTUCampus
python test_cross_dataset.py ./config/cuhk_softmax.yaml Market1501
python test_cross_dataset.py ./config/cuhk_softmax_triplet.yaml Market1501
python test_cross_dataset.py ./config/cuhk_softmax.yaml DukeMTMC
python test_cross_dataset.py ./config/cuhk_softmax_triplet.yaml DukeMTMC

python test_cross_dataset.py ./config/msmt_softmax.yaml NTUCampus
python test_cross_dataset.py ./config/msmt_softmax_triplet.yaml NTUCampus
python test_cross_dataset.py ./config/msmt_softmax.yaml Market1501
python test_cross_dataset.py ./config/msmt_softmax_triplet.yaml Market1501
python test_cross_dataset.py ./config/msmt_softmax.yaml DukeMTMC
python test_cross_dataset.py ./config/msmt_softmax_triplet.yaml DukeMTMC


python test_cross_dataset.py ./config/duke_softmax.yaml MSMT17
python test_cross_dataset.py ./config/duke_softmax_triplet.yaml MSMT17
python test_cross_dataset.py ./config/market_softmax.yaml MSMT17
python test_cross_dataset.py ./config/market_softmax_triplet.yaml MSMT17
python test_cross_dataset.py ./config/ntu_softmax.yaml MSMT17
python test_cross_dataset.py ./config/ntu_softmax_triplet.yaml MSMT17
python test_cross_dataset.py ./config/cuhk_softmax.yaml MSMT17
python test_cross_dataset.py ./config/cuhk_softmax_triplet.yaml MSMT17