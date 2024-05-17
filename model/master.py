import torch
from torch import nn
from torch.nn.modules.linear import Linear
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.normalization import LayerNorm
import math
from transformers import AutoTokenizer, AutoConfig, AutoModel
import os


def _acquire_device(config):
    if config.use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu)
        device = torch.device('cuda:{}'.format(config.gpu))
    else:
        device = torch.device('cpu')
    return device


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:x.shape[1], :]


class SAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.temperature = math.sqrt(self.d_model / nhead)

        self.qtrans = nn.Linear(d_model, d_model, bias=False)
        self.ktrans = nn.Linear(d_model, d_model, bias=False)
        self.vtrans = nn.Linear(d_model, d_model, bias=False)

        attn_dropout_layer = []
        for i in range(nhead):
            attn_dropout_layer.append(Dropout(p=dropout))
        self.attn_dropout = nn.ModuleList(attn_dropout_layer)

        # input LayerNorm
        self.norm1 = LayerNorm(d_model, eps=1e-5)

        # FFN layerNorm
        self.norm2 = LayerNorm(d_model, eps=1e-5)
        self.ffn = nn.Sequential(
            Linear(d_model, d_model),
            nn.ReLU(),
            Dropout(p=dropout),
            Linear(d_model, d_model),
            Dropout(p=dropout)
        )

    def forward(self, x):
        x = self.norm1(x)
        q = self.qtrans(x).transpose(0, 1)
        k = self.ktrans(x).transpose(0, 1)
        v = self.vtrans(x).transpose(0, 1)

        dim = int(self.d_model / self.nhead)
        att_output = []
        for i in range(self.nhead):
            if i == self.nhead - 1:
                qh = q[:, :, i * dim:]
                kh = k[:, :, i * dim:]
                vh = v[:, :, i * dim:]
            else:
                qh = q[:, :, i * dim:(i + 1) * dim]
                kh = k[:, :, i * dim:(i + 1) * dim]
                vh = v[:, :, i * dim:(i + 1) * dim]

            atten_ave_matrixh = torch.softmax(torch.matmul(qh, kh.transpose(1, 2)) / self.temperature, dim=-1)
            if self.attn_dropout:
                atten_ave_matrixh = self.attn_dropout[i](atten_ave_matrixh)
            att_output.append(torch.matmul(atten_ave_matrixh, vh).transpose(0, 1))
        att_output = torch.concat(att_output, dim=-1)

        # FFN
        xt = x + att_output
        xt = self.norm2(xt)
        att_output = xt + self.ffn(xt)

        return att_output


class TAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.qtrans = nn.Linear(d_model, d_model, bias=False)
        self.ktrans = nn.Linear(d_model, d_model, bias=False)
        self.vtrans = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout = []
        if dropout > 0:
            for i in range(nhead):
                self.attn_dropout.append(Dropout(p=dropout))
            self.attn_dropout = nn.ModuleList(self.attn_dropout)

        # input LayerNorm
        self.norm1 = LayerNorm(d_model, eps=1e-5)
        # FFN layerNorm
        self.norm2 = LayerNorm(d_model, eps=1e-5)
        # FFN
        self.ffn = nn.Sequential(
            Linear(d_model, d_model),
            nn.ReLU(),
            Dropout(p=dropout),
            Linear(d_model, d_model),
            Dropout(p=dropout)
        )

    def forward(self, x):
        x = self.norm1(x)
        q = self.qtrans(x)
        k = self.ktrans(x)
        v = self.vtrans(x)

        dim = int(self.d_model / self.nhead)
        att_output = []
        for i in range(self.nhead):
            if i == self.nhead - 1:
                qh = q[:, :, i * dim:]
                kh = k[:, :, i * dim:]
                vh = v[:, :, i * dim:]
            else:
                qh = q[:, :, i * dim:(i + 1) * dim]
                kh = k[:, :, i * dim:(i + 1) * dim]
                vh = v[:, :, i * dim:(i + 1) * dim]
            atten_ave_matrixh = torch.softmax(torch.matmul(qh, kh.transpose(1, 2)), dim=-1)
            if self.attn_dropout:
                atten_ave_matrixh = self.attn_dropout[i](atten_ave_matrixh)
            att_output.append(torch.matmul(atten_ave_matrixh, vh))
        att_output = torch.concat(att_output, dim=-1)

        # FFN
        xt = x + att_output
        xt = self.norm2(xt)
        att_output = xt + self.ffn(xt)

        return att_output


class TemporalAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.trans = nn.Linear(d_model, d_model, bias=False)

    def forward(self, z):
        h = self.trans(z)  # [N, T, D]
        query = h[:, -1, :].unsqueeze(-1)
        lam = torch.matmul(h, query).squeeze(-1)  # [N, T, D] --> [N, T]
        lam = torch.softmax(lam, dim=1).unsqueeze(1)
        output = torch.matmul(lam, z)  # [N, 1, T], [N, T, D] --> [N, 1, D]
        return output


# from https://github.com/THUDM/P-tuning-v2
class PrefixEncoder(torch.nn.Module):
    r'''
    The torch.nn model to encode the prefix
    Input shape: (batch-size, prefix-length)
    Output shape: (batch-size, prefix-length, 2*layers*hidden)
    '''

    def __init__(self, config):
        super().__init__()
        self.prefix_projection = config.finbert_prefix_projection
        if self.prefix_projection:
            # Use a two-layer MLP to encode the prefix
            self.embedding = torch.nn.Embedding(config.finbert_prefix_len, config.finbert_hidden_size)
            self.trans = torch.nn.Sequential(
                torch.nn.Linear(config.finbert_hidden_size, config.finbert_prefix_hidden_size),
                torch.nn.Tanh(),
                torch.nn.Linear(config.finbert_prefix_hidden_size,
                                config.finbert_num_hidden_layers * 2 * config.finbert_hidden_size)
            )
        else:
            self.embedding = torch.nn.Embedding(config.finbert_prefix_len,
                                                config.finbert_num_hidden_layers * 2 * config.finbert_hidden_size)

    def forward(self, prefix: torch.Tensor):
        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.trans(prefix_tokens)
        else:
            past_key_values = self.embedding(prefix)
        return past_key_values


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()

        self.d_model = config.d_model
        self.d_feat = config.d_feat

        self.T_dropout_rate = config.T_dropout_rate
        self.S_dropout_rate = config.S_dropout_rate
        self.t_nhead = config.t_nhead
        self.s_nhead = config.s_nhead

        self.embedding_layers = nn.Sequential(
            # feature layer
            nn.Linear(config.d_feat, config.d_model),
            PositionalEncoding(config.d_model),
        )

        self.attention_layers = nn.Sequential(
            # intra-stock aggregation
            TAttention(d_model=config.d_model, nhead=config.t_nhead, dropout=config.T_dropout_rate),
            # inter-stock aggregation
            SAttention(d_model=config.d_model, nhead=config.s_nhead, dropout=config.S_dropout_rate),
            TemporalAttention(d_model=config.d_model),
            # decoder
            # nn.Linear(config.d_model, config.pred_len)
        )

        self.finbert_dropout = torch.nn.Dropout(config.finbert_hidden_dropout_prob)

        self.finbert_prefix_len = config.finbert_prefix_len
        self.finbert_n_layer = config.finbert_num_hidden_layers
        self.finbert_n_head = config.finbert_num_attention_heads
        self.finbert_n_embd = config.finbert_hidden_size // config.finbert_num_attention_heads
        self.finbert_max_position_embeddings = config.finbert_max_position_embeddings
        self.prefix_tokens = torch.arange(config.finbert_prefix_len).long()
        self.prefix_encoder = PrefixEncoder(config)
        self.batch_size = config.batch_size

        self.align_dim = 200
        self.stock_proj = nn.Linear(self.d_model, self.align_dim)
        self.tweet_proj = nn.Linear(config.finbert_hidden_size, self.align_dim)
        self.temp = config.temperature  # temperature

        self.tweet_decoder = nn.Linear(config.finbert_hidden_size, 2)
        self.stock_decoder = nn.Linear(self.d_model, 2)
        self.cat_decoder = nn.Linear(2 * self.align_dim, 2)
        self.device = _acquire_device(config)

    def get_prompt(self, batch_size):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.device)
        past_key_values = self.prefix_encoder(prefix_tokens)
        # bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(
            batch_size,
            self.finbert_prefix_len,
            self.finbert_n_layer * 2,
            self.finbert_n_head,
            self.finbert_n_embd
        )
        # [bsz,pre_seq_len,2*n_layer,self.n_head,self.n_embed]
        past_key_values = self.finbert_dropout(past_key_values)

        # [2*n_layer,bsz,self.n_head,pre_seq_len,self.n_embed]
        # [n_layer*[2,bsz,self.n_head,pre_seq_len,self.n_embed]]]
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)

        return past_key_values

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, x_tweet):
        # price
        x_enc = x_enc.squeeze(dim=0)  # [stock_num,time_step,feature_dim]
        x_mark_enc = x_mark_enc.squeeze(dim=0)
        src = x_enc[:, :, :]  # N, T, D
        data_embedding = self.embedding_layers(src)  # without temporal embed
        temporal_embedding_layer = nn.Linear(x_mark_enc.shape[-1], self.d_model).to(self.device)
        temporal_embedding = temporal_embedding_layer(x_mark_enc)
        # data_embedding += temporal_embedding
        data_output = self.attention_layers(data_embedding)
        data_output = data_output.squeeze(dim=1)  # [stock_num,hidden_size=64]

        # tweet
        # model = AutoModel.from_pretrained("model/finbert").to(self.device)
        # tweet_output = []
        # position_ids = torch.arange(self.finbert_max_position_embeddings).expand((1, -1)).to(self.device)
        # for stock_tweet in x_tweet:  # [stock_num,daliy_tweet_num,tweet]
        #     prompt_past_key_values = self.get_prompt(self.batch_size)
        #     prefix_attention_mask = torch.ones(self.batch_size, self.finbert_prefix_len).to(self.device)
        #     stock_tweet['attention_mask'] = torch.cat((prefix_attention_mask, stock_tweet['attention_mask']), dim=1)
        #     stock_tweet['past_key_values'] = prompt_past_key_values
        #     stock_tweet['position_ids'] = position_ids[:, stock_tweet['input_ids'].size()[1] - 1]
        #     tweet_output.append(model(**stock_tweet)[0][:, 0, :].squeeze(dim=0))
        # tweet_output = torch.stack(tweet_output)  # [stock_num,hidden_size=768]

        # # align
        # stock_align = torch.tanh(self.stock_proj(data_output))  # [stock_num,dim_align]
        # tweet_align = torch.tanh(self.tweet_proj(tweet_output))  # [stock_num,dim_align]

        # sim = (tweet_align @ stock_align.T) / self.temp  # [stock_num,stock_num] sim_ij=tweet_h_i*data_h_j
        # stock_num = sim.shape[0]

        # align_loss = -torch.log(
        #     torch.softmax(sim, dim=1)[torch.arange(stock_num), torch.arange(stock_num)]).mean() - torch.log(
        #     torch.softmax(sim, dim=0)[torch.arange(stock_num), torch.arange(stock_num)]).mean()

        # tweet_pred = self.tweet_decoder(tweet_output)
        data_pred = self.stock_decoder(data_output)
        # cat_pred = self.cat_decoder(torch.cat([tweet_align, stock_align], dim=-1))  # [stock_num,2]
        tweet_pred = data_pred
        cat_pred = data_pred
        align_loss = 0

        return tweet_pred, data_pred, cat_pred, align_loss
