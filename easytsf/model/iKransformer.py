import torch
import torch.nn as nn

from easytsf.layer.kanlayer import KANInterface
from easytsf.layer.transformer import Encoder, EncoderLayer, FullAttention, iTransformer_Embedder


class KANAttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None, kan_type=None, kan_param=None):
        super(KANAttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        if kan_type == "JacobiKAN":
            self.query_projection = KANInterface(d_model, d_keys * n_heads, layer_type="JacobiKAN", degree=kan_param)
            self.key_projection = KANInterface(d_model, d_keys * n_heads, layer_type="JacobiKAN", degree=kan_param)
            self.value_projection = KANInterface(d_model, d_values * n_heads, layer_type="JacobiKAN", degree=kan_param)
            self.out_projection = KANInterface(d_values * n_heads, d_model, layer_type="JacobiKAN", degree=kan_param)
        elif kan_type == "TaylorKAN":
            self.query_projection = KANInterface(d_model, d_keys * n_heads, layer_type="TaylorKAN", order=kan_param)
            self.key_projection = KANInterface(d_model, d_keys * n_heads, layer_type="TaylorKAN", order=kan_param)
            self.value_projection = KANInterface(d_model, d_values * n_heads, layer_type="TaylorKAN", order=kan_param)
            self.out_projection = KANInterface(d_values * n_heads, d_model, layer_type="TaylorKAN", order=kan_param)
        elif kan_type == "KAN":
            self.query_projection = KANInterface(d_model, d_keys * n_heads, layer_type="KAN", n_grid=kan_param)
            self.key_projection = KANInterface(d_model, d_keys * n_heads, layer_type="KAN", n_grid=kan_param)
            self.value_projection = KANInterface(d_model, d_values * n_heads, layer_type="KAN", n_grid=kan_param)
            self.out_projection = KANInterface(d_values * n_heads, d_model, layer_type="KAN", n_grid=kan_param)
        elif kan_type == "WavKAN":
            self.query_projection = KANInterface(d_model, d_keys * n_heads, layer_type="WavKAN", )
            self.key_projection = KANInterface(d_model, d_keys * n_heads, layer_type="WavKAN", )
            self.value_projection = KANInterface(d_model, d_values * n_heads, layer_type="WavKAN", )
            self.out_projection = KANInterface(d_values * n_heads, d_model, layer_type="WavKAN", )
        elif kan_type == "MoK":
            self.query_projection = KANInterface(d_model, d_keys * n_heads, layer_type="TaylorKAN", order=kan_param)
            self.key_projection = KANInterface(d_model, d_keys * n_heads, layer_type="TaylorKAN", order=kan_param)
            self.value_projection = KANInterface(d_model, d_values * n_heads, layer_type="TaylorKAN", order=kan_param)
            self.out_projection = KANInterface(d_values * n_heads, d_model, layer_type="TaylorKAN", order=kan_param)
        else:
            raise NotImplementedError
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class iKransformer(nn.Module):
    def __init__(self, hist_len, pred_len, output_attention, d_model, dropout, factor, n_heads,
                 d_ff, activation, e_layers, kan_type, kan_param, use_out_kan):
        super(iKransformer, self).__init__()

        self.seq_len = hist_len
        self.pred_len = pred_len
        self.output_attention = output_attention
        # Embedding
        self.enc_embedding = iTransformer_Embedder(self.seq_len, d_model, dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    KANAttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout, output_attention=output_attention),
                        d_model, n_heads, kan_type=kan_type, kan_param=kan_param),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        # Decoder
        if use_out_kan:
            self.projection = KANInterface(d_model, self.pred_len, layer_type="JacobiKAN", degree=degree)
        else:
            self.projection = nn.Linear(d_model, self.pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, _, N = x_enc.shape

        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def forward(self, var_x, marker_x):
        x_enc = var_x[..., 0]
        x_mark_enc = marker_x[..., 0,]
        dec_out = self.forecast(x_enc, x_mark_enc)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]
