import math
from copy import deepcopy

from torch import nn
import torch

class MAREAttention(nn.Module):
    def __init__(self, config):
        '''
        :param config: 模型config
        '''
        super().__init__()
        self.config = config

        if config.mlp_head:  # MLP
            self.mare_attention_query = nn.Sequential(
                nn.LayerNorm(config.hidden_size),
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.LayerNorm(config.hidden_size),
                nn.GELU(),
                nn.Linear(config.hidden_size, config.hidden_size * 2, bias=False),
            )
            self.mare_attention_key = nn.Sequential(
                nn.LayerNorm(config.hidden_size),
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.LayerNorm(config.hidden_size),
                nn.GELU(),
                nn.Linear(config.hidden_size, config.hidden_size * 2, bias=False),
            )
        else:  # linear
            self.mare_attention_query = nn.Sequential(
                nn.LayerNorm(config.hidden_size),
                nn.Linear(config.hidden_size, config.hidden_size * 2, bias=False)
            )
            self.mare_attention_key = nn.Sequential(
                nn.LayerNorm(config.hidden_size),
                nn.Linear(config.hidden_size, config.hidden_size * 2, bias=False)
            )

        with torch.no_grad():
            self.mare_attention_query[-1].weight.data.normal_(mean=0.0, std=0.02)
            self.mare_attention_key[-1].weight.data.normal_(mean=0.0, std=0.02)

            bias = nn.Parameter(torch.zeros((2, 1, 1), dtype=torch.float32), requires_grad=True)
            bias.data[0] = -config.mean_bias
            bias.data[1] = config.mean_bias

        self.register_parameter('bias', bias)

        self.bias._mare_initialized=True

        self.process_type = getattr(config, 'process_type', 0)


    def forward(self, hidden_states, length,
                tti_mask=None):
        # q = self.q(inputs) # (B, C, D)
        # print(dir(self.mare_attention_query))
        start_pos = 0
        end_pos = start_pos + length
        q = self.mare_attention_query(hidden_states[:, start_pos:end_pos]) # (B, C, 2D)
        k = self.mare_attention_key(hidden_states[:, end_pos:]) # (B, L, 2D)

        B, L, D = k.shape
        D //= 2
        C = q.shape[1]

        q = q.reshape(B, C, 2, D).permute(0, 2, 1, 3)
        k = k.reshape(B, L, 2, D).permute(0, 2, 1, 3)

        att_score = torch.einsum('band,bamd->banm', q, k) / math.sqrt(q.shape[-1]) # (B, 2, C, L)
        att_score = att_score + self.bias # (B, 2, C, L)

        att_mare_mask = torch.nn.functional.gumbel_softmax(att_score, tau=1, hard=True, dim=1)[:, 1] # (B, C, L)

        if tti_mask is not None:
            m = tti_mask[:, None, end_pos:].expand_as(att_mare_mask)
            att_mare_mask = 1-m + att_mare_mask * m

        if self.process_type == 2:
            mare_mask_with_cls = torch.zeros(B, L + C, L + C, dtype=torch.float32,
                                             device=att_mare_mask.device)  # (B, L, L)

            score_matrix = torch.einsum('bdn,bdm->bnm', att_mare_mask, att_mare_mask)  # (B, l, l)
            binarized_score_matrix = (score_matrix > 1e-3).float()

            # 右下方：mask*mask
            mare_mask_with_cls[:, end_pos:,
            end_pos:] = score_matrix + binarized_score_matrix - score_matrix.detach()  # straight through
            # 右上方：att_mare_mask
            mare_mask_with_cls[:, start_pos:end_pos, end_pos:] = att_mare_mask
            # 左下方：att_mare_mask.T
            mare_mask_with_cls[:, end_pos:, start_pos:end_pos] = att_mare_mask.transpose(-1, -2)
            # 左上方：eye
            mare_mask_with_cls[:, start_pos:end_pos, start_pos:end_pos] = torch.eye(length, device=att_mare_mask.device)

        else:
            raise ValueError('Not Implemented')

        return mare_mask_with_cls, att_score, mare_mask_with_cls[:, start_pos:end_pos]


def get_mare_predictor(config):
    # mare predictors
    if config.mare_att == 4:
        predictor = MAREAttention(config=config)
    else:
        raise NotImplementedError()

    return predictor
