# KAN4TSF

KAN4TSF is an official PyTorch implementation of "Are KANs Effective for Time Series Forecasting?". 
Although it is called KAN4TSF, it also supports time series forecasting methods of various network structures (CNN, Linear, Transformer and others).

## Usage

### Environment
Step by Step with Conda:
```shell
conda create -n kan4tsf python=3.10
conda activate kan4tsf
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
python -m pip install lightning
```

or you can just:
```shell
pip install -r requirements.txt
```

### Code and Data
ETT can be downloaded within this project, and other datasets will be released soon.

### Running
```shell
python train.py -c config/reproduce_conf/DLinear/96for96.py
```

## Cite
If you find this repo useful, please cite our paper:
```
@inproceedings{han2023are,
  title={Are KANs Effective for Time Series Forecasting?},
  author={Xiao Han, Xinfeng Zhang, Yiling Wu, Zhenduo Zhang and Zhe Wu},
  booktitle={arXiv},
  year={2024},
}
```
