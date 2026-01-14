import torch
import torch.nn as nn

from Libs.layers.Autoformer_EncDec import series_decomp
import torch.nn.functional as F


class DLinear(nn.Module):

    def __init__(self, configs, individual=True):
        super(DLinear, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.seq_len
        self.DIL = configs.DIL
        self.decompsition = series_decomp(configs.moving_avg)
        self.individual = individual
        self.channels = configs.enc_in
        self.feat = 128
        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            for i in range(self.channels):
                self.Linear_Seasonal.append(
                    nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(
                    nn.Linear(self.seq_len, self.pred_len))

                self.Linear_Seasonal[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
                self.Linear_Trend[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

            self.Linear_Seasonal.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
            self.Linear_Trend.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))

        if self.DIL:
            self.embedding = nn.Linear(configs.enc_in * configs.seq_len, self.feat)
            self.decompose = nn.Linear(self.feat, self.feat)
            self.projectionDIL = nn.Linear(self.feat, configs.num_class)
        else:
            self.projection1 = nn.Linear(configs.enc_in * configs.seq_len, 128)#
            self.projection2 = nn.Linear(128, configs.num_class)

    def encoder(self, x):
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(
            0, 2, 1), trend_init.permute(0, 2, 1)
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                                          dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.pred_len],
                                       dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](
                    seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](
                    trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)
        x = seasonal_output + trend_output
        return x.permute(0, 2, 1)

    def get_different_class_embeddings(self, X, y):
        B, D = X.shape
        XX = torch.zeros_like(X)
        for i in range(B):
            mask = y != y[i]
            candidates = X[mask]
            if len(candidates) == 0:
                raise ValueError(f"No different class sample found for index {i} with class {y[i].item()}")
            else:
                rand_idx = torch.randint(0, candidates.shape[0], (1,))
                XX[i] = candidates[rand_idx]
        return XX

    def forward(self, x_enc, y):
        b = x_enc.shape[0]
        x_enc = x_enc.transpose(-1, -2)
        enc_out = self.encoder(x_enc)
        enc_out = enc_out.reshape(enc_out.shape[0], -1)
        if self.DIL:
            z = self.embedding(enc_out)
            mask_ivariant = F.sigmoid(self.decompose(z))
            mask_variant = F.sigmoid(-self.decompose(z))
            Ivariant = mask_ivariant * z
            Variant = mask_variant * z
            if self.training:
                DifClass = self.get_different_class_embeddings(Ivariant, y)
                perm = torch.randperm(b)
                Variant = Variant[perm]
            else:
                DifClass = torch.zeros_like(Ivariant).to(Ivariant.device)
                Variant = torch.zeros_like(Ivariant).to(Ivariant.device)

            outv = self.projectionDIL(Ivariant + Variant)
            outc = self.projectionDIL(Ivariant + DifClass)
            return outv, outc, Ivariant

        else:
            enc_out = self.projection1(enc_out)
            output=self.projection2(enc_out)
            return output, None, enc_out
