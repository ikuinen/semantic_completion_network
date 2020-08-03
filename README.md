# A Simple PyTorch Implementation of the Semantic Completion Networks 

The paper accepted by AAAI-2020 is available here [pdf](https://arxiv.org/abs/1911.08199).

+ Train on the ActivityNet Dataset:
```
python train.py --config-path config/activitynet/main.json
```

## Prerequisites
- pytorch
- python3
- gensim
- opencv-python
- h5py


## A link to pre-extracted features
The C3D visual features from [google drive](https://drive.google.com/drive/folders/1D3nav3TKZmYNHvSLBgDt1vpBUXoV2MRv?usp=sharing) and save it to the `data/` folder.