import os

import lightning.pytorch as pl
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class GeneralTSFDataset(Dataset):
    """
    General TSF Dataset.
    """

    def __init__(self, data_root, dataset_name, hist_len, pred_len, data_split, freq, mode):
        self.data_dir = os.path.join(data_root, dataset_name)
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.train_len, self.val_len, self.test_len = data_split
        self.freq = freq

        self.mode = mode
        assert mode in ['train', 'valid', 'test'], "mode {} mismatch, should be in [train, valid, test]".format(mode)

        mode_map = {'train': 0, 'valid': 1, 'test': 2}
        self.set_type = mode_map[mode]
        self.var, self.time_marker = self.__read_data__()

    def __read_data__(self):
        norm_feature_path = os.path.join(self.data_dir, 'feature.npz')
        norm_feature = np.load(norm_feature_path)

        norm_var = norm_feature['norm_var']
        norm_time_marker = norm_feature['norm_time_marker']

        border1s = [0, self.train_len, self.train_len + self.val_len]
        border2s = [self.train_len, self.train_len + self.val_len, self.train_len + self.val_len + self.test_len]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        norm_var = norm_var[border1:border2]
        norm_time_marker = norm_time_marker[border1:border2]

        norm_var = norm_var[:, :, np.newaxis]  # (L, N, C)
        norm_time_marker = norm_time_marker[:, np.newaxis, :]  # L, -1, C, C = [tod, dow, dom, doy]
        return norm_var, norm_time_marker

    def __getitem__(self, index):
        hist_start = index
        hist_end = index + self.hist_len
        pred_end = hist_end + self.pred_len
        var_x = self.var[hist_start:hist_end, ...]
        marker_x = self.time_marker[hist_start:hist_end, ...]

        var_y = self.var[hist_end:pred_end, ...]
        marker_y = self.time_marker[hist_end:pred_end, ...]
        return var_x, marker_x, var_y, marker_y

    def __len__(self):
        return len(self.var) - (self.hist_len + self.pred_len) + 1


def data_provider(config, mode):
    return GeneralTSFDataset(
        data_root=config['data_root'],
        dataset_name=config['dataset_name'],
        hist_len=config['hist_len'],
        pred_len=config['pred_len'],
        data_split=config['data_split'],
        freq=config['freq'],
        mode=mode,
    )


class DataInterface(pl.LightningDataModule):

    def __init__(self, **kwargs):
        super().__init__()
        self.num_workers = kwargs['num_workers']
        self.batch_size = kwargs['batch_size']
        self.kwargs = kwargs

    def train_dataloader(self):
        train_set = data_provider(self.kwargs, mode='train')
        return DataLoader(train_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True,
                          drop_last=True)

    def val_dataloader(self):
        val_set = data_provider(self.kwargs, mode='valid')
        return DataLoader(val_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        test_set = data_provider(self.kwargs, mode='test')
        return DataLoader(test_set, batch_size=1, num_workers=self.num_workers, shuffle=False)
