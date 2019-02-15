# Basic Person ReID Baseline and Project Template

## Requirements
- [python 3](https://www.python.org/downloads/)
- [pytorch 1.0 + torchvision](https://pytorch.org/)
- [yacs](https://github.com/rbgirshick/yacs) Yet Another Configuration System
- [fire](https://github.com/google/python-fire) Automatically generating command line interfaces (CLIs)

Install all dependences libraries
``` bash
pip3 install -r requirements.txt
```

## Datasets
Person Re-ID datasets defined in the `DATASETS.NAMES` of the yaml config file will be download automatically into the `datasets` folder.
Currently support:
* [CUHK03](http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html)
* [Market1501](http://www.liangzheng.org/Project/project_reid.html)
* [DukeMTMC](https://github.com/layumi/DukeMTMC-reID_evaluation)
* [MSMT17](https://www.pkuvmc.com/publications/msmt17.html)

## Training:
``` bash
python train.py ./config/market_softmax.yaml
```

## Testing:
``` bash
### No Re-Ranking
python test.py ./config/market_softmax.yaml

### With Re-Ranking
python test.py ./config/market_softmax.yamln True
```

## Testing Cross Dataset:
``` bash
### Market1501 -> DukeMTMC
python test_cross_dataset.py ./config/market_softmax.yaml DukeMTMC
```

## Results

|            |   Softmax   | Softmax+Triplet |
|     ---    |     --      | --              |
| CUHK03     | 61.8 (58.7) | 63.6 (60.2)     |
| Market1501 | 91.3 (77.8) | 92.8 (82.0)     |
| DukeMTMC   | 84.1 (67.7) | 86.2 (73.0)     |
| MSMT17     | 71.6 (43.9) | 74.0 (47.5)     |


## File and Folder Structure
```
├──  config
│    └── defaults.py  - here's the default config file.
│    └── market_softmax.yml  - here's the specific config file for specific model or dataset.
│ 
│
├──  data_loader  
│    └── datasets_importer  - here's the datasets folder that is responsible for all data handling.
│        └── BaseDataset.py  - Generate and show basic statistics of the dataset in terminal.
│        └── ImageDataset.py  - PIL read the images and gernerate PyTorch Dataset Object
│        └── market1501.py  - Data handler for dataset Market1501
│
│    └── transforms  - here's the data preprocess folder is responsible for all data augmentation.
│        └── transforms.py  - initialization of data transformation of the network
│        └── RandomErasing.py  - Custom-made RandomErasing process for data augmentation
│ 
│    └── samplers  - here's the id samplering function for triplet training
│        └── triplet_sampler.py
│ 
│    └── data_loader.py  - here's the file to make dataloader.
│
│
├──  datasets  
│    └── PersonReID_Dataset_Downloader.py  - here's the file to automatic download the dataset
│    └── Market1501  - here will be the folder storing the downloaded dataset
│
│
├──  evaluation
│   ├── evaluation.py   - this file to compute the CMC and mAP result.
│   └── re_ranking.py   - this file is the re_ranking function.
│
│
├── logger  - this folder is to create a logger and store the training process.
│
│
├── loss  - this folder is the loss function for the network.
│   └── make_loss.py
│   └── triplet_loss.py  - Custom-made Triplet Loss function
│  
│
├── models  - this folder contains models of the project.
│   └── BasicModule.py     - Re-package the Pytorch Model with save and load models function
│   └── ResNet50.py        - Model with ResNet50 as backbone
│
│
├── optimizer - this folder contains optimizer of the project.
│
├── scheduler - this folder contains learning rate scheduler
|
├── utils       
│   └── check_jupyter_run.py - if it is running on the jupyter use the notebook version of tqdm
│   
│ 
├── train.py                - here's the train the network
│    
└── test.py                 - here's the test the network performance   
│
└── test_cross_dataset.pu	- test the performance in cross-dataset scenario
```