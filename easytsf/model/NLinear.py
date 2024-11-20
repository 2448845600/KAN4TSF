import torch
import torch.nn as nn


class NLinear(nn.Module):
    def __init__(self, hist_len, pred_len, var_num, individual=False):
        super(NLinear, self).__init__()
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.var_num = var_num
        self.individual = individual

        if self.individual:
            self.Linear = nn.ModuleList()
            for i in range(self.var_num):
                self.Linear.append(nn.Linear(self.hist_len, self.pred_len))
        else:
            self.Linear = nn.Linear(self.hist_len, self.pred_len)

    def forward(self, var_x, marker_x):
        seq_last = var_x[:, -1:, :].detach()
        var_x = var_x - seq_last
        if self.individual:
            output = torch.zeros([var_x.size(0), self.pred_len, var_x.size(2)], dtype=var_x.dtype).to(var_x.device)
            for i in range(self.channels):
                output[:, :, i] = self.Linear[i](var_x[:, :, i])
            var_x = output
        else:
            var_x = self.Linear(var_x.permute(0, 2, 1)).permute(0, 2, 1)
        var_x = var_x + seq_last
        return var_x
