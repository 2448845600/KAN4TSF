# KAN4TSF

KAN4TSF is an official PyTorch implementation of [KAN4TSF: Are KAN and KAN-based models Effective for Time Series Forecasting?](https://arxiv.org/pdf/2408.11306). 
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
ETTh1 and ETTm1 can be downloaded within this project, and other datasets can be downloaded from ![BaiduPan](链接: https://pan.baidu.com/s/18NKge4dsMIuGQFom7n2S2w?pwd=zumh 提取码: zumh 复制这段内容后打开百度网盘手机App，操作更方便哦).

### Running
```shell
python train.py -c config/reproduce_conf/RMoK/ETTh1_96for96.py
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
