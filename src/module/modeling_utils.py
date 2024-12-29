from dataclasses import dataclass
import torch
from transformers.file_utils import ModelOutput
from typing import Optional, Tuple, Union

from copy import deepcopy
from torch import nn

from transformers.modeling_outputs import MaskedLMOutput, QuestionAnsweringModelOutput

@dataclass
class BaseModelOutputWithPastAndCrossAttentionsMARE(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    attention_mask: Optional[torch.FloatTensor] = None
    mare_att_mask: Optional[torch.FloatTensor] = None
    mare_logits: Optional[torch.FloatTensor] = None
    all_disc: Optional[Tuple[torch.FloatTensor]] = None
    all_ee_preds: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class BaseModelOutputWithPoolingAndCrossAttentionsMARE(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    attention_mask: Optional[torch.FloatTensor] = None
    mare_att_mask: Optional[torch.FloatTensor] = None
    mare_logits: Optional[torch.FloatTensor] = None
    all_disc: Optional[Tuple[torch.FloatTensor]] = None
    all_ee_preds: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class SequenceClassifierOutputMARE(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    attention_mask: Optional[torch.FloatTensor] = None
    mare_att_mask: Optional[torch.FloatTensor] = None
    mare_logits: Optional[torch.FloatTensor] = None
    mare_loss: Optional[Union[torch.FloatTensor, Tuple[torch.FloatTensor]]] = None
    classification_loss: Optional[torch.FloatTensor] = None
    tokens_remained: Optional[Union[torch.FloatTensor, Tuple[torch.FloatTensor]]] = None
    layer_tokens_remained: Optional[Union[Tuple[torch.FloatTensor], Tuple[Tuple[torch.FloatTensor]]]] = None
    all_ee_preds: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class QuestionAnsweringModelOutputMARE(QuestionAnsweringModelOutput):
    attention_mask: Optional[torch.FloatTensor] = None
    mare_att_mask: Optional[torch.FloatTensor] = None
    mare_logits: Optional[torch.FloatTensor] = None
    mare_loss: Optional[torch.FloatTensor] = None
    classification_loss: Optional[torch.FloatTensor] = None
    tokens_remained: Optional[torch.FloatTensor] = None
    layer_tokens_remained: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class MaskedLMOutputMARE(MaskedLMOutput):
    attention_mask: Optional[torch.FloatTensor] = None
    mare_att_mask: Optional[torch.FloatTensor] = None
    mare_logits: Optional[torch.FloatTensor] = None
    mare_loss: Optional[torch.FloatTensor] = None
    classification_loss: Optional[torch.FloatTensor] = None
    tokens_remained: Optional[torch.FloatTensor] = None
    layer_tokens_remained: Optional[Tuple[torch.FloatTensor]] = None


def convert_softmax_mask_to_digit(mare_att_mask):
    # mare_att_mask [batch, from, to, seq_len]
    return (mare_att_mask == 0).to(dtype=torch.int64).unsqueeze(1).unsqueeze(1)




def get_multiple_layer(src_layer, num_layers, include_self=True):
    '''
    复制src_layer num_layers次，include self表示最终返回的layer列表是否含有src_layer自己
    :param src_layer: 需要被复制的层
    :param num_layers: 层数
    :param include_self: 最终的结果是否包括src_layer自己
    :return: nn.ModuleList(List)
    '''
    if num_layers <= 0: # base case
        return []

    layers = []
    for i in range(num_layers - int(include_self)):
        layers.append(deepcopy(src_layer))
    
    return nn.ModuleList([src_layer] * int(include_self) + layers)


def get_certain_layer(data_idx, default_layer, moe_layers):
    '''
    给定data_idx，取出对应当前任务的moe层
    :param data_idx: 任务idx
    :param moe_layers: 有哪些layers
    :return: nn.Module
    '''

    # # 0 -> default ; else -> moe_layers
    if data_idx == 0:
        return default_layer
    return moe_layers[data_idx - 1]


def get_con_loss(q, q_):
    # full_q = torch.cat([q_, q], dim=0)
    cos_sim = torch.einsum('nc,mc->nm',
                           q, q_)  # (B, B)
    ori_cos_sim = torch.einsum('nc,mc->nm',
                           q, q)  # (B, B)

    B = cos_sim.shape[0]
    label = torch.arange(B, device=cos_sim.device)

    con_loss = torch.nn.functional.cross_entropy(cos_sim, label)
    con_loss += torch.nn.functional.cross_entropy(ori_cos_sim, label)

    return con_loss / 2


def masked_mean(x, mask, dim=1):
    if len(mask.shape) == len(x.shape) - 1:
        mask = mask.unsqueeze(-1)
    return (x * mask).sum(dim) / mask.sum(dim)

def masked_max(x, mask, dim=1):
    if len(mask.shape) == len(x.shape) - 1:
        mask = mask.unsqueeze(-1)
    x = torch.masked_fill(x, ~mask.bool().expand_as(x), -10000.)
    return x.max(dim)[0]

def masked_softmax(x, mask, dim=1):
    if len(mask.shape) == len(x.shape) - 1:
        mask = mask.unsqueeze(-1)
    x = torch.masked_fill(x, ~mask.expand_as(x), -10000.)

    return torch.softmax(x, dim)

if __name__ == '__main__':
    a = torch.arange(10)
    b = torch.tensor([0,1,1,1,0,0,1,1,0,1])

    print(masked_mean(a, b, 0))
    print(torch.masked_fill(a, b, -10000)) # True -> value
