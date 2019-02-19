# Basic Person ReID Baseline and Project Template

**A Basic Person ReID Baseline and a PyTorch Template for NTU ROSE Person ReID Project.**

I am not a big fan of pytorch [ignite](https://github.com/pytorch/ignite)(Too high level). So I have rewrite L1aoXingyu's [reid_baseline](https://github.com/L1aoXingyu/reid_baseline) following the basic pytorch training and testing logtic flow. As a basic reid baseline, I remove most of tricks and custom-made scheduler, except the bash hard triplet loss and random erasing. Evething elso are all pytorch native build-in functions. 

## Requirements
- [python 3](https://www.python.org/downloads/)
- [pytorch 1.0 + torchvision](https://pytorch.org/)
- [yacs](https://github.com/rbgirshick/yacs) Yet Another Configuration System
- [fire](https://github.com/google/python-fire) Automatically generating command line interfaces (CLIs)

Install all dependences libraries
``` bash
pip3 install -r requirements.txt
```

## Configs

Use different yaml config files for different experiment settings. All the config files are store in folder `config`. Please use different `OUTPUT_DIR` names for different experiments to avoid conflit and accidentally files overwritten.


## Datasets
This code support CUHK03, Market1501, DukeMTMC and MSMT17 datasets. All these dataset should be defined in the `DATASETS.NAMES` of the config file, our code will be download the corresponding dataset automatically (into the `datasets` folder). As this fuction require access to __Google Drive__, it will not work in China. 
Currently support:
* [CUHK03](http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html)
* [Market1501](http://www.liangzheng.org/Project/project_reid.html)
* [DukeMTMC](https://github.com/layumi/DukeMTMC-reID_evaluation)
* [MSMT17](https://www.pkuvmc.com/publications/msmt17.html)


## Training:
* Batch Size 128 uses around 12.01G GPU memory (Only recommend for Titian GPU and above on server interface)
* Batch Size 64 is the most suitable size for GTX 1080ti

``` bash
python train.py ./config/market_softmax.yaml

### Change GPU
python train.py ./config/market_softmax.yaml --DEVICE=cuda:5
```

## Testing:
* Default testing batch size is 256. Reduce to accommodate your GPU memory size.

``` bash
### No Re-Ranking
python test.py ./config/market_softmax.yaml

### Change GPU
python test.py ./config/market_softmax.yaml --DEVICE=cuda:5

### With Re-Ranking
python test.py ./config/market_softmax.yaml --RE_RANKING=True
```

## Testing Cross Dataset:
``` bash
### Market1501 -> DukeMTMC
python test_cross_dataset.py ./config/market_softmax.yaml DukeMTMC
```

## Results

##### __Batch Size 128__: Rank1  (mAP)

|            |   Softmax   | Softmax+Triplet |Softmax+Re-ranking|Softmax+Triplet+Re-ranking |
|     ---    |     --      | --              |--                |--                         |
| CUHK03     | 61.8 (58.7) | 63.6 (60.2)     |68.2 (70.0)       |72.6 (73.9)                |
| Market1501 | 91.3 (77.8) | 92.8 (82.0)     |90.6 (85.7)       |93.3 (90.1)                |
| DukeMTMC   | 84.1 (67.7) | 86.2 (73.0)     |85.3 (79.6)       |88.2 (83.5)                |
| MSMT17     | 71.6 (43.9) | 74.0 (47.5)     |-                 |-                          |

##### __Batch Size 64__: Rank1  (mAP)

|            |   Softmax   | Softmax+Triplet |Softmax+Re-ranking|Softmax+Triplet+Re-ranking |
|     ---    |     --      | --              |--                |--                         |
| CUHK03     | 56.1 (52.4) | 65.6 (61.8)     |64.2 (64.9)       |74.6 (75.5)                |
| Market1501 | 91.6 (78.7) | 93.2 (82.0)     |90.8 (85.9)       |93.8 (90.2)                |
| DukeMTMC   | 83.4 (66.6) | 86.4 (72.4)     |84.9 (79.5)       |88.1 (83.0)                |
| MSMT17     | 69.0 (40.1) | -     |-                 |-                          |




## File and Folder Structure
```
├──  checkpoint - here's store all the training models checkpoints and testing results
│    └── Market1501
│        └── Softmax_BS64
│            └── log.txt                 - training log
│            └── ResNet50_epo120.pth     - saved model checkpoint parameters
│            └── result.txt              - testing result
│            └── result_re-ranking.txt   - testing result with re-ranking
│ 
│
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

## Issues
- [ ] Re-Ranking is not working on MSMT17. Currently under investigating 