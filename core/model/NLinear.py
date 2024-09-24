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
        x = var_x[..., 0]  # (B, L, N)
        seq_last = x[:, -1:, :].detach()
        x = x - seq_last
        if self.individual:
            output = torch.zeros([x.size(0), self.pred_len, x.size(2)], dtype=x.dtype).to(x.device)
            for i in range(self.channels):
                output[:, :, i] = self.Linear[i](x[:, :, i])
            x = output
        else:
            x = self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = x + seq_last
        return x
