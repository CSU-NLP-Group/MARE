import argparse
import glob
import logging
import math
import random
import timeit
import shutil
import json
from copy import deepcopy
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from itertools import chain

from transformers import (
    WEIGHTS_NAME,
    AutoTokenizer,
    AdamW,
    BertConfig,
    BertTokenizer,
    T5Tokenizer,
    RobertaTokenizer,
    get_linear_schedule_with_warmup, T5TokenizerFast, BertTokenizerFast, RobertaTokenizerFast,
)

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter
from torchmetrics import Metric
from module.modeling_mare import MAREForSequenceClassification

from torch.utils.data import Dataset
from sklearn.metrics import precision_recall_fscore_support

from datasets import Dataset, config


from typing import TypedDict, Union

os.environ["HF_DATASETS_OFFLINE"] = 'FALSE'

logger = logging.getLogger(__name__)

ALL_MODELS = list()

MODEL_CLASSES = {
    'mare': (BertConfig, MAREForSequenceClassification, AutoTokenizer),
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def to_list(tensor):
    if isinstance(tensor, (list, tuple)):
        return [to_list(t) for t in tensor]
    else:
        return tensor.detach().cpu().tolist()


def check(name):
    return any(name.startswith(x) for x in ['beer', 'hotel', 'legal',
                                            'movie', 'propaganda',
                                            'multirc', 'movies',
                                            'fever', 'boolq',
                                            'evidence_inference', 'amazon',
                                            'aapd', 'hoc'])


def is_eraser(args):
    tasks = ('evidence_inference', 'boolq', 'movies', 'multirc', 'scifact',
             'esnli', 'esnli_flat', 'cose', 'cose_simplified')
    return any(x in args.task_name for x in tasks)


class TokenF1(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("true_positives", default=torch.zeros(1), dist_reduce_fx="sum")
        self.add_state("prediction_positive", default=torch.zeros(1), dist_reduce_fx="sum")
        self.add_state("target_positive", default=torch.zeros(1), dist_reduce_fx="sum")
        self.add_state("overall", default=torch.zeros(1), dist_reduce_fx="sum")

    def _get_top_mask(self, target: torch.Tensor):
        if len(target.shape) == 1:
            return int(-1 not in target)

        return [self._get_top_mask(e) for e in target]

    def update(self, preds: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
        top_mask = self._get_top_mask(target)

        top_mask = torch.tensor(top_mask, device=mask.device).unsqueeze(-1)

        mask = mask.long() * top_mask
        if mask.dim() < preds.dim():
            mask = mask.unsqueeze(1).expand_as(preds)

        assert preds.shape == target.shape == mask.shape, f'{preds.shape}, {target.shape}, {mask.shape}'
        predictions = mask * (preds > 0.5).long() * top_mask
        target = mask * target.long() * top_mask

        true_positives = predictions * target

        self.true_positives += true_positives.sum()
        self.prediction_positive += predictions.sum()
        self.target_positive += target.sum()
        self.overall += mask.sum()
        assert self.prediction_positive >= self.true_positives, f'{self.prediction_positive}, {self.true_positives}'
        assert self.target_positive >= self.true_positives, f'{self.target_positive}, {self.true_positives}'

    def compute(self):
        precision = self.true_positives / (self.prediction_positive + 1e-10)
        recall = self.true_positives / (self.target_positive + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        sparsity = self.target_positive / (self.overall + 1e-10)
        cur_sparsity = self.prediction_positive / (self.overall + 1e-10)

        return {'precision': precision, 'recall': recall, 'f1': f1, 'sparsity': sparsity, 'cur_sparsity': cur_sparsity}


class OptimizedTokenF1(Metric):
    def __init__(self, tokenizer):
        super().__init__()
        self.add_state("true_positives", default=torch.zeros(1), dist_reduce_fx="sum")
        self.add_state("prediction_positive", default=torch.zeros(1), dist_reduce_fx="sum")
        self.add_state("target_positive", default=torch.zeros(1), dist_reduce_fx="sum")
        self.add_state("overall", default=torch.zeros(1), dist_reduce_fx="sum")
        self.tokenizer = tokenizer

    def update(self, preds: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, input_ids: torch.Tensor):
        mask = mask.long()
        if mask.dim() < preds.dim():
            mask = mask.unsqueeze(0).expand_as(preds)
        if target.dim() < preds.dim():
            target = target.unsqueeze(0).expand_as(preds)
        assert preds.shape == target.shape == mask.shape, f'{preds.shape}, {target.shape}, {mask.shape}'
        predictions = mask * (preds > 0.5).long()
        target = mask * target.long()

        true_positives = 0
        target_positive = 0
        prediction_positive = 0
        overall = 0
        for pred, gold, ids in zip(predictions, target, input_ids):
            for i in range(len(pred)):
                if self.tokenizer.convert_ids_to_tokens(ids[i].item()).startswith('##'):
                    continue
                if gold[i] == 1 and pred[i] == 1:
                    true_positives += 1
                    target_positive += 1
                    prediction_positive += 1
                elif gold[i] == 1:
                    target_positive += 1
                elif pred[i] == 1:
                    prediction_positive += 1
                overall += 1

        self.true_positives += true_positives
        self.prediction_positive += prediction_positive
        self.target_positive += target_positive
        self.overall += overall
        assert self.prediction_positive >= self.true_positives, f'{self.prediction_positive}, {self.true_positives}'
        assert self.target_positive >= self.true_positives, f'{self.target_positive}, {self.true_positives}'

    def compute(self):
        precision = self.true_positives / (self.prediction_positive + 1e-10)
        recall = self.true_positives / (self.target_positive + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        sparsity = self.target_positive / (self.overall + 1e-10)
        cur_sparsity = self.prediction_positive / (self.overall + 1e-10)

        return {'precision': precision, 'recall': recall, 'f1': f1, 'sparsity': sparsity, 'cur_sparsity': cur_sparsity}


class Example(TypedDict):
    context: str
    label: int = None
    annotation: list = None
    sec_context: str = None

    #
    # def __init__(self,
    #              context,
    #              label=None,
    #              annotation=None,
    #              sec_context=None):
    #     super().__init__()
    #     self.context = context
    #     self.label = label
    #     self.annotation = annotation
    #     self.sec_context=sec_context


class InputFeatures(TypedDict):
    input_ids: list
    attention_mask: list
    labels: Union[int, list]
    token_type_ids: list = None
    annotation: list = None
    query_input_ids: list = None
    query_attention_mask: list = None

    # def __init__(self, input_ids, input_mask, label, token_type_ids=None, annotation=None):
    #     self.input_ids = input_ids
    #     self.input_mask = input_mask
    #     self.token_type_ids = token_type_ids
    #     self.label = label
    #     self.annotation = annotation


def balance(examples, num_samples=-1):
    import random

    if isinstance(examples[0]['label'], int):
        pos = [e for e in examples if e['label'] == 1]
        neg = [e for e in examples if e['label'] == 0]
    else:
        pos = [e for e in examples if e['label'][0] == 1]
        neg = [e for e in examples if e['label'][0] == 0]

    print(f'# pos: {len(pos)}, # neg: {len(neg)}')

    if num_samples > 0:
        pos = random.sample(pos, num_samples // 2)
        neg = random.sample(neg, num_samples // 2)
    else:
        if len(pos) > len(neg):
            pos = random.sample(pos, len(neg))
        else:
            neg = random.sample(neg, len(pos))

    print(f'# pos: {len(pos)}, # neg: {len(neg)}')
    examples = pos + neg

    return examples


def read_beer_examples(filename, aspect=0, pos_thres=0.6,
                       neg_thres=0.4, is_balance=False,
                       conditioned=False,
                       num_samples=-1, ):
    examples = []
    label_map = ['positive', 'negative']
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip().split()
            scores = line[:5]
            text = line[5:]
            scores = 1 if float(scores[aspect]) >= pos_thres else 0 if float(scores[aspect]) <= neg_thres else -1
            if scores == -1:
                continue

            examples.append(Example(context=' '.join(text), label=int(scores)))

    if is_balance:
        examples = balance(examples, num_samples)
    elif num_samples > 0:
        examples = random.sample(examples, num_samples)

    if conditioned:
        new_examples = []

        for example in examples:
            # 蕴含
            new_examples.append(Example(context=example.context, label=1,
                                        sec_context='Is the label of following sentence %s ?' % label_map[
                                            example.label]))
            # 冲突
            new_examples.append(Example(context=example.context, label=0,
                                        sec_context='Is the label of following sentence %s ?' % label_map[
                                            1 - example.label]))

        examples = new_examples

    return examples


def read_beer_annotation_examples(filename, aspect=0,
                                  pos_thres=0.6, neg_thres=0.4,
                                  conditioned=False, multiple=False):
    examples = []
    label_map = ['positive', 'negative']
    with (open(filename, 'r', encoding='utf-8') as f):
        for line in f:
            line = json.loads(line.strip())

            if multiple:
                scores = [float(y) for y in line['y']]
                scores = list(map(lambda x: 1 if x >= pos_thres else 0 if x <= neg_thres else 1, scores))

                annotations = []
                for aspect in range(5):
                    annotation = line[str(aspect)]  # list [[0, 3]]
                    if annotation:
                        annotation = [1 if any(a[0] <= i < a[1] for a in annotation) else 0 for i in
                                      range(len(line['x']))]
                    else:
                        annotation = [-1] * len(line['x'])
                    annotations.append(annotation)

                # overall first
                # annotations = annotations[-1:] + annotations[:-1]
                annotation = annotations
            else:
                scores = 1 if float(line["y"][aspect]) >= pos_thres else 0 if float(
                    line["y"][aspect]) <= neg_thres else -1
                if scores == -1:
                    continue

                annotation = line[str(aspect)]  # list [[0, 3]]
                if not annotation:
                    continue
                annotation = [1 if any(a[0] <= i < a[1] for a in annotation) else 0 for i in range(len(line['x']))]

            text = ' '.join(line['x'])

            if not conditioned:
                examples.append(Example(context=text, label=scores, annotation=annotation))
            else:
                examples.append(Example(context=text, label=1, annotation=annotation,
                                        sec_context='Is the label of following sentence %s ?' % label_map[int(scores)]))
                examples.append(Example(context=text, label=0, annotation=annotation,
                                        sec_context='Is the label of following sentence %s ?' % label_map[
                                            1 - int(scores)]))
    print(len(examples))
    return examples


def read_hotel_examples(filename, aspect=0, **kwargs):
    examples = []
    with open(filename, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            line = line.strip().split('\t')
            scores = line[1]
            text = line[-1].split()

            examples.append(Example(context=' '.join(text), label=int(scores)))

    return examples


def read_hotel_annotation_examples(filename, aspect=0, **kwargs):
    examples = []

    with open(filename, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            line = line.strip().split('\t')
            scores = line[1]
            text = line[2].split()
            annotation = [int(e) for e in line[3].split()]
            if not sum(annotation):
                continue
            assert len(annotation) == len(text), f'{len(annotation)}, {len(text)}'

            examples.append(Example(context=' '.join(text), label=int(scores), annotation=annotation))

    return examples


def convert_single_example(example, tokenizer, max_seq_length, max_chunks=1):
    def add_special_tokens(tokens, tokenizer, sec_context_tokens):
        cls_token = []
        sep_token = []
        if isinstance(tokenizer, (BertTokenizer, BertTokenizerFast)):  # for bert
            cls_token = [tokenizer.cls_token]
            sep_token = [tokenizer.sep_token]
        elif isinstance(tokenizer, (RobertaTokenizer, RobertaTokenizerFast)):  # for roberta
            cls_token = [tokenizer.bos_token]
            sep_token = [tokenizer.eos_token]
        elif isinstance(tokenizer, (T5Tokenizer, T5TokenizerFast)):  # for t5
            cls_token = []
            sep_token = [tokenizer.eos_token]
        else:
            print('nothing to do in add_special_tokens')

        token_type_ids = []
        ids = 0
        if sec_context_tokens:
            token_type_ids.extend([ids] * (len(sec_context_tokens) + 1))
            ids = ids + 1
            tokens = sec_context_tokens + sep_token + tokens

        tokens = cls_token + tokens + sep_token
        token_type_ids.extend([ids] * (len(tokens) - len(token_type_ids)))

        return tokens, token_type_ids

    def get_annotation(annotation, tokenizer, sec_context_tokens):
        sec_len = len(sec_context_tokens)
        multiple_sentence = len(sec_context_tokens) > 0

        prefix = [1] if not isinstance(tokenizer, (T5Tokenizer, T5TokenizerFast)) else []
        if sec_len > 0:
            # BERT single: 0 + 2 - 1 = 1
            # T5 single: 0 + 1 - 1 = 0
            prefix = [1] * (sec_len +
                            get_special_token_number(tokenizer,
                                                     multiple_sentence) - 1)

        annotation = annotation + [0]
        annotation = prefix + annotation
        if isinstance(tokenizer, (T5Tokenizer, T5TokenizerFast)):
            annotation = (([1] * sec_len + [0]) if multiple_sentence else []) + annotation

        return annotation

    def get_special_token_number(tokenizer, multiple_sentence: bool = False):
        if isinstance(tokenizer, (BertTokenizer, BertTokenizerFast)):  # for bert
            special_token_number = 2
        elif isinstance(tokenizer, (RobertaTokenizer, RobertaTokenizerFast)):  # for roberta
            special_token_number = 2
        elif isinstance(tokenizer, (T5Tokenizer, T5TokenizerFast)):  # for t5
            special_token_number = 1
        else:
            special_token_number = -int(multiple_sentence)

        return special_token_number + int(multiple_sentence)

    def convert_single_annotation(annotation, context_tokens, invalid_annot,
                                  sec_context_tokens, stacked_token, tokenizer):
        if not invalid_annot:
            new_annot = [0] * len(context_tokens)
            idx = 0
            n = len(new_annot)
            i = 0
            while i < n:
                for k in range(len(stacked_token[idx])):
                    if i + k >= n:
                        break
                    new_annot[i + k] = annotation[idx]
                if i + len(stacked_token[idx]) >= n:
                    break
                i += len(stacked_token[idx])
                idx += 1
            annotation = get_annotation(new_annot, tokenizer, sec_context_tokens)
        else:
            annotation = [-1] * len(context_tokens)
        return annotation

    def convert_single_chunk(context_tokens, example, max_seq_length, sec_context_tokens,
                             stacked_token, tokenizer):
        tokens, token_type_ids = add_special_tokens(context_tokens, tokenizer, sec_context_tokens)
        annotation = None
        # TODO: pad here
        if 'annotation' in example:
            if isinstance(example['annotation'][0], (list, tuple)):
                annotation = [convert_single_annotation(example['annotation'][i], context_tokens,
                                                        -1 in example['annotation'][i],
                                                        sec_context_tokens, stacked_token, tokenizer) for i in
                              range(len(example['annotation']))]
            else:
                annotation = convert_single_annotation(example['annotation'], context_tokens,
                                                       -1 in example['annotation'],
                                                       sec_context_tokens, stacked_token, tokenizer)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        # Zero-pad up to the sequence length.
        token_padding = [tokenizer.pad_token_id] * (max_seq_length - len(input_ids))
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += token_padding
        token_type_ids += padding
        input_mask += padding
        if 'annotation' in example:
            if isinstance(example['annotation'][0], (list, tuple)):
                for i, e in enumerate(example['annotation']):
                    if -1 not in e:
                        annotation[i] += padding
                    else:
                        annotation[i] += [-1] * (len(input_ids) - len(annotation[i]))
                    assert len(
                        annotation[
                            i]) == max_seq_length, f'{annotation[i]}, {len(input_ids)}, {len(annotation[i])}, {max_seq_length}'
            else:
                if -1 not in example['annotation']:
                    annotation += padding
                else:
                    annotation += [-1] * (len(input_ids) - len(annotation))
                assert len(
                    annotation) == max_seq_length, f'{annotation}, {len(input_ids)}, {len(annotation)}, {max_seq_length}'
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(token_type_ids) == max_seq_length
        label = example['label'] if 'label' in example else -1
        if isinstance(tokenizer, T5Tokenizer):
            lb = 'positive' if label else 'negative'
            # label = tokenizer.encode(f'the label is {lb}', add_special_tokens=True)
            label = tokenizer.encode(f'Prediction: {lb}', add_special_tokens=True)
        return annotation, input_ids, input_mask, label, token_type_ids

    stacked_token = [tokenizer.tokenize(e) for e in example['context'].split()]
    context_tokens = list(chain.from_iterable(stacked_token))

    sec_context_tokens = []
    if 'sec_context' in example:
        sec_token = [tokenizer.tokenize(e) for e in example['sec_context'].split()]
        sec_context_tokens = list(chain.from_iterable(sec_token))


    features = []
    for i in range(min(max_chunks, math.ceil(len(context_tokens) / max_seq_length))):
        tks = context_tokens[i * max_seq_length:(i + 1) * max_seq_length - len(sec_context_tokens)
                                                - get_special_token_number(tokenizer,
                                                                           len(sec_context_tokens) > 0)]
        annotation, input_id, input_mask, label, token_type_id = convert_single_chunk(tks,
                                                                                      example,
                                                                                      max_seq_length,
                                                                                      sec_context_tokens,
                                                                                      stacked_token, tokenizer)

        input_feature = InputFeatures(
            input_ids=input_id,
            attention_mask=input_mask,
            labels=label,
            token_type_ids=token_type_id,
            annotation=annotation,
        )
        if input_feature['annotation'] is None:
            input_feature.pop('annotation')

        features.append(input_feature)

    return features


def convert_examples_to_features(args, examples, tokenizer, max_seq_length, max_chunks=1) -> Dataset:
    """Loads a data file into a list of `InputBatch`s."""

    datasets = []
    for example in tqdm(examples):
        datasets.extend(convert_single_example(example, tokenizer, max_seq_length, max_chunks))

    dataset = Dataset.from_list(datasets)


    print(len(dataset))
    return dataset


def accuracy(out, labels):
    outputs = np.argmax(out, axis=-1)
    if len(outputs.shape) == 2:
        return np.sum(outputs[:, -1] == np.array(labels))
    return np.sum(outputs == labels)


def multilabel_accuracy(out, labels):
    # B, C, 2
    outputs = np.argmax(out, axis=-1)
    return np.sum(outputs == labels)


def load_and_cache_examples(args, tokenizer, prefix, aspect=None, task_name=None,
                            data_dir=None, evaluate=False):
    task_name = task_name or args.task_name
    data_dir = data_dir or args.data_dir
    if 'beer_cor' in data_dir:
        data_dir = 'data/beer'
    if args.local_rank not in [-1, 0] and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()
    if task_name.startswith('beer') and prefix in ['dev', 'eval', 'val', 'validation']:
        prefix = 'heldout'
    # Load data features from cache or dataset file
    if args.do_test and prefix == 'dev':
        prefix = 'test'
    input_dir = data_dir if data_dir else "."
    path = "cached_{}_{}".format(
        prefix,
        str(args.max_seq_length),
    )

    if any(task_name.startswith(x) for x in ['beer', 'hotel']):
        path += '_aspect_{}'.format(aspect)
    if 'cor' in task_name:
        path += '_cor'
    if 'full' in task_name:
        path += '_full'

    if args.annotation_ratio is not None and prefix == 'train' and 'beer' not in task_name and 'hotel' not in task_name:
        path += f'_{args.annotation_ratio:.2f}'

    is_balance = args.balance if prefix == 'train' else False
    path = path + ('' if is_balance else '_nobalance')

    path = path + ('' if args.max_chunks == 1 else f'_{args.max_chunks}')
    cached_features_file = os.path.join(
        input_dir,
        path
    )
    if args.do_lower_case:
        cached_features_file += '_lower'
    
    print(cached_features_file)
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        # features_and_dataset = torch.load(cached_features_file)
        # dataset = features_and_dataset["dataset"]

        dataset = Dataset.load_from_disk(cached_features_file)

    else:
        logger.info("Creating features from dataset file at %s", input_dir)
        if task_name.startswith('beer'):
            print(aspect)
            if prefix == 'test':
                examples = read_beer_annotation_examples(os.path.join(data_dir, 'annotations.json'),
                                                         aspect=aspect,
                                                         conditioned=args.conditioned,
                                                         multiple='multiple' in task_name or 'multilabel' in task_name)
            else:
                filename = os.path.join(data_dir, 'reviews.aspect' + task_name[-1] + '.' + prefix + ".txt")
                if 'cor' in task_name:
                    data_dir='data/beer'
                    filename = os.path.join(data_dir, 'reviews.260k.' + prefix + ".txt")
                    is_balance = True
                if 'full' in task_name:
                    filename = os.path.join(data_dir, 'reviews.260k.' + prefix + ".txt")
                    is_balance = False
                else:
                    examples = read_beer_examples(filename, aspect=aspect, is_balance=is_balance,
                                                  conditioned=args.conditioned, num_samples=args.num_samples)
        elif task_name.startswith('hotel'):
            hotel_map = ['hotel_Location', 'hotel_Service', 'hotel_Cleanliness']
            print(aspect)
            if prefix == 'test':
                examples = read_hotel_annotation_examples(
                    os.path.join(data_dir, 'annoated/' + hotel_map[int(task_name[-1])] + '.train'), aspect=aspect)
            else:
                filename = os.path.join(data_dir, 'hotel' + task_name[-1] + '/' + prefix + ".tsv")
                examples = read_hotel_examples(filename, aspect=aspect)
        else:
            assert (False)

        dataset: Dataset = convert_examples_to_features(args, examples=examples, tokenizer=tokenizer,
                                                        max_seq_length=args.max_seq_length,
                                                        max_chunks=args.max_chunks)

        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            # torch.save({"dataset": dataset}, cached_features_file)
            dataset.save_to_disk(cached_features_file)

    if args.local_rank == 0 and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    print(cached_features_file, len(dataset))
    return dataset.with_format('torch')


def construct_data(args, batch: dict, data_idx=None):
    # batch = tuple(t.to(args.device) for t in batch)
    # inputs = {
    #     "input_ids": batch[0].long(),
    #     "attention_mask": batch[1].long(),
    #     "token_type_ids": batch[2].long(),
    #     "labels": batch[3].long(),
    #     "data_idx": data_idx
    # }
    inputs = {k: v.to(args.device) for k, v in batch.items()}

    if args.model_type != 'bert':
        inputs['data_idx'] = data_idx
    if args.model_type == 'bert' and 'annotation' in inputs:
        inputs.pop('annotation')

    inputs['input_ids'] = inputs['input_ids'][:, :args.max_seq_length]
    inputs['attention_mask'] = inputs['attention_mask'][:, :args.max_seq_length]
    inputs['token_type_ids'] = inputs['token_type_ids'][:, :args.max_seq_length]
    if inputs.get('annotation') is not None:
        inputs['annotation'] = inputs['annotation'][:, :args.max_seq_length]


    if args.train_rationales and 'annotations' in inputs:
        inputs['mare_att_mask'] = inputs['annotations']
    return inputs


class RoundWrapper:
    def __init__(self, dataloaders):
        self.dataloaders = dataloaders
        self.cur = 0
        self.data_gen = [(data for data in dataloader) for dataloader in dataloaders]
        self.is_end = [False] * len(self.data_gen)
        self._length = sum(len(dataloader) for dataloader in dataloaders)

    def get_next(self, cur):
        data = None
        while data is None:
            if all(self.is_end):
                print('All dataloaders are exhausted')
                break
            try:
                while self.is_end[cur]:
                    cur = (cur - 1) % len(self.dataloaders)
                data = next(self.data_gen[cur])
            except StopIteration:
                self.is_end[cur] = True
                cur = (cur - 1) % len(self.dataloaders)
                # self.data_gen[cur] = iter(self.dataloaders[cur])
                # data = next(self.data_gen[cur])

        return data

    def __iter__(self):
        def gen():
            for _ in range(len(self)):
                cur = self.cur
                self.cur = (self.cur + 1) % len(self.dataloaders)

                data = self.get_next(cur)
                # print(data)

                # one_hot_data_idx = torch.zeros((len(self.dataloaders),), dtype=torch.long, device=data[0].device).scatter_(0, torch.tensor([cur], dtype=torch.long, device=data[0].device), 1)
                # yield (data, one_hot_data_idx)
                yield (data, torch.tensor(cur, dtype=torch.long, device=data['input_ids'].device))

        return gen()

    def __len__(self):
        return self._length


class BalancedRoundWrapper:
    def __init__(self, dataloaders):
        self.dataloaders = dataloaders
        self.cur = 0
        self.data_gen = [(data for data in dataloader) for dataloader in dataloaders]
        self._length = sum(len(dataloader) for dataloader in dataloaders)

    def get_next(self, cur):
        try:
            data = next(self.data_gen[cur])
        except StopIteration:
            self.data_gen[cur] = iter(self.dataloaders[cur])
            data = next(self.data_gen[cur])

        return data

    def __iter__(self):
        def gen():
            for _ in range(len(self)):
                cur = self.cur
                self.cur = (self.cur + 1) % len(self.dataloaders)

                data = self.get_next(cur)

                # one_hot_data_idx = torch.zeros((len(self.dataloaders),), dtype=torch.long, device=data[0].device).scatter_(0, torch.tensor([cur], dtype=torch.long, device=data[0].device), 1)
                # yield (data, one_hot_data_idx)
                yield (data, torch.tensor(cur, dtype=torch.long, device=data['input_ids'].device))

        return gen()

    def __len__(self):
        return self._length


class OversamplingRoundWrapper(BalancedRoundWrapper):
    def __init__(self, dataloaders):
        super().__init__(dataloaders)
        self._length = max(len(dataloader) for dataloader in dataloaders) * len(dataloaders)


class RandomWrapper:
    def __init__(self, dataloaders):
        self.dataloaders = dataloaders
        self.cur = 0
        self.data_gen = [(data for data in dataloader) for dataloader in dataloaders]
        self.is_end = [False] * len(self.data_gen)
        self._length = sum(len(dataloader) for dataloader in dataloaders)

    def get_idx(self):
        if all(self.is_end):
            print('All dataloaders are exhausted')
            return None

        cur = random.randint(0, len(self.dataloaders) - 1)
        while self.is_end[cur]:
            cur = random.randint(0, len(self.dataloaders) - 1)

        return cur

    def get_next(self):
        cur = self.get_idx()
        if cur is None:  # run out of data
            return None

        data = None
        while data is None:
            try:
                data = next(self.data_gen[cur])
            except StopIteration:
                self.is_end[cur] = True
                cur = self.get_idx()
                if cur is None:
                    return None

        return data

    def __iter__(self):
        def gen():
            for _ in range(len(self)):
                cur = self.cur
                self.cur = (self.cur + 1) % len(self.dataloaders)
                data = self.get_next()

                # one_hot_data_idx = torch.zeros((len(self.dataloaders),), dtype=torch.long, device=data[0].device).scatter_(0, torch.tensor([cur], dtype=torch.long, device=data[0].device), 1)
                # yield (data, one_hot_data_idx)
                yield (data, torch.tensor(cur, dtype=torch.long, device=data['input_ids'].device))

        return gen()

    def __len__(self):
        return self._length


class BalancedRandomWrapper:
    def __init__(self, dataloaders):
        self.dataloaders = dataloaders
        self.cur = 0
        self.data_gen = [(data for data in dataloader) for dataloader in dataloaders]
        self._length = sum(len(dataloader) for dataloader in dataloaders)

    def get_idx(self):
        return random.randint(0, len(self.dataloaders) - 1)

    def get_next(self):
        cur = self.get_idx()

        try:
            data = next(self.data_gen[cur])
        except StopIteration:
            self.data_gen[cur] = iter(self.dataloaders[cur])
            data = next(self.data_gen[cur])

        return data

    def __iter__(self):
        def gen():
            for _ in range(len(self)):
                cur = self.cur
                self.cur = (self.cur + 1) % len(self.dataloaders)
                data = self.get_next()

                # one_hot_data_idx = torch.zeros((len(self.dataloaders),), dtype=torch.long, device=data[0].device).scatter_(0, torch.tensor([cur], dtype=torch.long, device=data[0].device), 1)
                # yield (data, one_hot_data_idx)
                yield (data, torch.tensor(cur, dtype=torch.long, device=data['input_ids'].device))

        return gen()

    def __len__(self):
        return self._length


class OversamplingRandomWrapper(BalancedRandomWrapper):
    def __init__(self, dataloaders):
        super().__init__(dataloaders)
        self._length = max(len(dataloader) for dataloader in dataloaders) * len(dataloaders)


class UndersamplingRandomWrapper(BalancedRandomWrapper):
    def __init__(self, dataloaders):
        super().__init__(dataloaders)
        self._length = min(len(dataloader) for dataloader in dataloaders) * len(dataloaders)


wrapper_map = {'round': RoundWrapper,
               'balanced_round': BalancedRoundWrapper,
               'oversampling_round': OversamplingRoundWrapper,
               'random': RandomWrapper,
               'balanced_random': BalancedRandomWrapper,
               'oversampling_random': OversamplingRandomWrapper,
               'undersampling_ramdom': UndersamplingRandomWrapper,
               }


class MyDataCollator:
    def __init__(self,
                 tokenizer=None,
                 model=None,
                 padding=True,
                 max_length=None,
                 pad_to_multiple_of=None,
                 return_tensors: str = "pt",
                 ):
        self.tokenizer: T5TokenizerFast = tokenizer
        self.model = model
        self.padding = padding
        self.max_length = max_length
        self.pad_to_multiple_of=pad_to_multiple_of
        self.return_tensors = return_tensors

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors

        def pad_one(key, pad_to_multiple_of=1):
            if key not in features[0]: return
            tmp = [{'input_ids': each.pop(key)} for each in features]
            tmp = self.tokenizer.pad(
                tmp,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=pad_to_multiple_of,
                return_tensors=return_tensors,
            )

            temp_features[key] = tmp['input_ids']

        temp_features = {}
        padded_keys = [('query_input_ids', 1), ('query_attention_mask', 1),
                       ('annotation', self.pad_to_multiple_of)]

        [pad_one(*k) for k in padded_keys]
        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        features.update(temp_features)

        return features


def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(args.output_dir)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    data_collator = MyDataCollator(
        tokenizer,
        model=model,
        pad_to_multiple_of=1,
    )

    train_dataloader = DataLoader(train_dataset,
                                  sampler=train_sampler,
                                  batch_size=args.train_batch_size,
                                  num_workers=8,
                                  collate_fn=data_collator)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # no_decay = ["bias", "LayerNorm.weight"]
    # optimizer_grouped_parameters = [
    #     {
    #         "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
    #         "weight_decay": args.weight_decay,
    #     },
    #     {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    # ]

    # print(optimizer_grouped_parameters)
    # exit(0)
    # optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    # TODO: seperate learning rate
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon,
                      weight_decay=args.weight_decay)
    if args.warmup_steps == -1:
        args.warmup_steps = int(t_total * 0.1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    # Added here for reproductibility
    set_seed(args)
    best_acc = 0
    # torch.autograd.set_detect_anomaly(True)
    flag = False
    for epoch in train_iterator:
        print(f'Training Epoch: {epoch}, current learning_rate: {optimizer.param_groups[0]["lr"]}')
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            inputs = construct_data(args, batch)

            # inv_keys = []
            # for k, v in inputs.items():
            #     if v is None:
            #         print(k)
            #         inv_keys.append(k)
            # [inputs.pop(e) for e in inv_keys]
            # print({k: v.shape for k, v in inputs.items()})
            # exit(0)
            outputs = model(**inputs)
            loss = outputs[0]
            if torch.isinf(loss):
                logging.info('%s', loss)
                logging.info('%s', outputs['classification_loss'])
                logging.info('%s', outputs['mare_loss'])
                logging.info('%s', '\n'.join(tokenizer.batch_decode(inputs['input_ids'])))
                exit(0)
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            # logging.info('%s', inputs)
            # logging.info('%s', tokenizer.batch_decode(inputs['input_ids']))
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()

                global_step += 1
                flag = False

                # Log metrics
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Only evaluate when single GPU otherwise metrics may not average well
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

                # Save model checkpoint
                if args.local_rank in [-1, 0] and (args.save_steps > 0 and global_step % args.save_steps == 0):
                    update = True
                    if args.evaluate_during_training:
                        results = evaluate(args, model, tokenizer, evaluate_prefix='dev',
                                           aspect=int(args.task_name[-1]) if check(args.task_name) else None)
                        acc = results['acc']
                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                        if acc > best_acc:
                            best_acc = acc
                            print('best acc:', acc)
                        else:
                            update = False
                    if update:
                        output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)

                        model_to_save = model.module if hasattr(model, "module") else model
                        model_to_save.save_pretrained(output_dir)
                        tokenizer.save_pretrained(output_dir)

                        torch.save(args, os.path.join(output_dir, "training_args.bin"))
                        logger.info("Saving model checkpoint to %s", output_dir)

                        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                        logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

            if args.evaluate_during_training and global_step % args.evaluate_steps == 0 and not flag:
                flag = True
                results = evaluate(args, model, tokenizer,
                                   evaluate_prefix='test' if check(args.task_name) else 'dev',
                                   aspect=int(args.task_name[-1]) if check(args.task_name) else None)

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / (global_step + 1e-10)


def train_multiple(args, train_datasets, model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(args.output_dir)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_samplers = [RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset) for
                      train_dataset in train_datasets]
    data_collator = MyDataCollator(
        tokenizer,
        model=model,
        pad_to_multiple_of=1,
    )
    train_dataloaders = [
        DataLoader(train_dataset,
                   sampler=train_samplers[i],
                   batch_size=args.train_batch_size,
                   num_workers=8,
                   collate_fn=data_collator) for
        i, train_dataset in enumerate(train_datasets)]

    lengths = [len(train_dataloader) for train_dataloader in train_dataloaders]

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (sum(lengths) // args.gradient_accumulation_steps) + 1
    else:
        t_total = sum(lengths) // args.gradient_accumulation_steps * args.num_train_epochs
    # print('total train steps: ', t_total)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    if args.warmup_steps == -1:
        args.warmup_steps = int(t_total * 0.1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", sum(lengths))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    # Added here for reproductibility
    set_seed(args)
    best_acc = 0
    torch.autograd.set_detect_anomaly(True)
    flag = False
    Wrapper = wrapper_map[args.dataloader_wrapper]
    print(f'using {args.dataloader_wrapper}.')
    for epoch in train_iterator:
        logger.info(f'Training Epoch: {epoch}, current learning_rate: {optimizer.param_groups[0]["lr"]}')
        epoch_iterator = tqdm(Wrapper(train_dataloaders), desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()

            inputs = construct_data(args, batch[0], batch[1])

            outputs = model(**inputs)
            loss = outputs[0]

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
                flag = False

                # Log metrics
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Only evaluate when single GPU otherwise metrics may not average well
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

                # Save model checkpoint
                if args.local_rank in [-1, 0] and (args.save_steps > 0 and global_step % args.save_steps == 0):
                    update = True
                    if args.evaluate_during_training:
                        results = evaluate_multiple(args, model, tokenizer, evaluate_prefix='dev', )
                        acc = results['acc']
                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                        if acc > best_acc:
                            best_acc = acc
                            print('best acc:', acc)
                        else:
                            update = False
                    if update:
                        output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)

                        model_to_save = model.module if hasattr(model, "module") else model
                        model_to_save.save_pretrained(output_dir)
                        tokenizer.save_pretrained(output_dir)

                        torch.save(args, os.path.join(output_dir, "training_args.bin"))
                        logger.info("Saving model checkpoint to %s", output_dir)

                        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                        logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

            if args.evaluate_during_training and global_step % args.evaluate_steps == 0 and not flag:
                flag = True
                results = evaluate_multiple(args, model, tokenizer, evaluate_prefix='test', )

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / (global_step + 1e-10)



def evaluate(args, model, tokenizer, prefix="", evaluate_prefix='dev',
             aspect=None, task_name=None, data_dir=None, data_idx=None):
    dataset = load_and_cache_examples(args, tokenizer, prefix=evaluate_prefix, aspect=aspect,
                                      evaluate=True, task_name=task_name, data_dir=data_dir)

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset)
    data_collator = MyDataCollator(
        tokenizer,
        model=model,
        pad_to_multiple_of=1,
    )
    eval_dataloader = DataLoader(dataset,
                                 sampler=eval_sampler,
                                 batch_size=args.eval_batch_size,
                                 num_workers=8,
                                 collate_fn=data_collator)

    # multi-gpu evaluate
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    logger.info("  Part of dataset: %s", evaluate_prefix)

    eval_accuracy = eval_examples = 0

    # macs_list = json.load(open('macs_list.json'))
    flops = 0
    start_time = timeit.default_timer()

    # tagging stats
    all_tag_preds = []
    all_tag_gold = []
    total_tags = 0.
    correct_tags = 0.

    # prediction stats
    correct_preds = 0.
    total_preds = 0.0
    all_pred_gold = []
    all_pred_preds = []

    is_t5 = 't5_' in args.model_type
    is_exp = ('mare' in args.model_type or
              'rnp' in args.model_type or
              'fr' in args.model_type or
              'bert_seq' in args.model_type)

    if is_exp:
        all_mare_loss, all_tokens_remained = list(), list()
    all_loss = []
    f1_metric = torch.nn.ModuleList([TokenF1() for _ in range(model.config.num_hidden_layers)]).to(model.device)
    flag = True
    cnt = 0
    data_idx = data_idx or 0
    idx = data_idx + (1 if hasattr(model, 'mare_att') and model.mare_att % 2 else 0)

    idx = (idx + 0) % 3
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        with torch.no_grad():
            inputs = construct_data(args, batch,
                                    data_idx)

            annotation = None
            if 'annotation' in inputs:
                annotation = inputs.pop('annotation')
            outputs = model(**inputs)

            if 'token_type_ids' in inputs and inputs['token_type_ids'][0].sum() > 0:
                tti_mask = inputs['token_type_ids']
            else:
                tti_mask = torch.ones_like(inputs['token_type_ids'])
                if not is_t5: tti_mask[:, 0] = 0
                tti_mask = tti_mask * inputs['attention_mask']

            all_loss.append(outputs.loss)
            if is_exp:
                if outputs.mare_loss.shape:
                    all_mare_loss.append(outputs.mare_loss[idx])
                    all_tokens_remained.append(outputs.tokens_remained[idx])
                    if annotation is not None:
                        if isinstance(outputs.mare_att_mask, (list, tuple)):
                            assert len(f1_metric) == len(outputs.mare_att_mask)
                            for cur_layer, (f1, cur_mask) in enumerate(zip(f1_metric, outputs.mare_att_mask)):
                                if cur_mask.shape == annotation.shape:
                                    f1(cur_mask, annotation, tti_mask)
                                else:
                                    if len(cur_mask.shape) > len(annotation.shape):
                                        cur = cur_mask[:, idx]
                                        cur = torch.cat([cur[:, 0:1], cur[:, 1 - annotation.shape[-1]:]],
                                                        dim=-1)
                                        annot = annotation
                                    else:
                                        cur = cur_mask
                                        annot = annotation[:, idx]

                                    assert cur.shape == annot.shape
                                    cur[..., 0] = 1
                                    f1(cur, annot, tti_mask)
                        else:
                            mare_att_mask = outputs.mare_att_mask
                            assert mare_att_mask.shape == annotation.shape, f'{mare_att_mask.shape} {annotation.shape}'
                            [f1(mare_att_mask, annotation, tti_mask) for f1 in f1_metric]

                        if flag:
                            mare_att_mask = outputs.mare_att_mask[-1] \
                                if isinstance(outputs.mare_att_mask, (list, tuple)) else \
                                outputs.mare_att_mask

                            if len(mare_att_mask.shape) == 3:
                                mare_att_mask = mare_att_mask[:, idx]
                                mare_att_mask = torch.cat([mare_att_mask[:, 0:1],
                                                       mare_att_mask[:, 1 - annotation.shape[-1]:]],
                                                      dim=-1)
                            mare_att_mask[..., 0] = 1
                            flag = False
                            for i in range(3):
                                tokens = []
                                for a, b, c in zip(inputs['input_ids'][i],
                                                   mare_att_mask[i],
                                                   annotation[i]):
                                    if a == tokenizer.pad_token_id:
                                        break
                                    b = b.long() > 0
                                    c = c > 0

                                    prefix = ''
                                    if b and b == c:
                                        prefix = '++'
                                    elif b:
                                        prefix = '--'
                                    elif c:
                                        prefix = '**'
                                    tokens.append(f'{prefix}{"".join(tokenizer.decode(a).split())}{prefix}')

                                logger.info(' '.join(tokens))

                    all_layer_tokens_remained = outputs.layer_tokens_remained[idx]

                else:
                    all_mare_loss.append(outputs.mare_loss)
                    all_tokens_remained.append(outputs.tokens_remained)
                    if annotation is not None:
                        if isinstance(outputs.mare_att_mask, (list, tuple)):
                            assert len(f1_metric) == len(outputs.mare_att_mask)
                            for cur_layer, (f1, cur_mask) in enumerate(zip(f1_metric, outputs.mare_att_mask)):
                                if cur_mask.shape == annotation.shape:
                                    f1(cur_mask, annotation, tti_mask)
                                else:
                                    if len(cur_mask.shape) > len(annotation.shape):
                                        cur = cur_mask[:, idx]
                                        cur = torch.cat([cur[:, 0:1], cur[:, 1 - annotation.shape[-1]:]],
                                                        dim=-1)
                                        annot = annotation
                                    else:
                                        cur = cur_mask
                                        annot = annotation[:, idx]

                                    assert cur.shape == annot.shape
                                    cur[..., 0] = 1
                                    f1(cur, annot, tti_mask)
                        else:
                            mare_att_mask = outputs.mare_att_mask
                            assert mare_att_mask.shape == annotation.shape, f'{mare_att_mask.shape} {annotation.shape}'
                            [f1(mare_att_mask, annotation, tti_mask) for f1 in f1_metric]

                        if flag:
                            mare_att_mask = outputs.mare_att_mask[-1] \
                                if isinstance(outputs.mare_att_mask, (list, tuple)) else \
                                outputs.mare_att_mask

                            if len(mare_att_mask.shape) == 3:
                                mare_att_mask = mare_att_mask[:, idx]
                                mare_att_mask = torch.cat([mare_att_mask[:, 0:1],
                                                       mare_att_mask[:, 1 - annotation.shape[-1]:]],
                                                      dim=-1)
                            mare_att_mask[..., 0] = 1
                            flag = False
                            for i in range(3):
                                tokens = []
                                for a, b, c in zip(inputs['input_ids'][i],
                                                   mare_att_mask[i],
                                                   annotation[i]):
                                    if a == tokenizer.pad_token_id:
                                        break
                                    b = b.long() > 0
                                    c = c > 0

                                    prefix = ''
                                    if b and b == c:
                                        prefix = '++'
                                    elif b:
                                        prefix = '--'
                                    elif c:
                                        prefix = '**'
                                    tokens.append(f'{prefix}{"".join(tokenizer.decode(a).split())}{prefix}')

                                logger.info(' '.join(tokens))
                    all_layer_tokens_remained = outputs.layer_tokens_remained

            # print(outputs[1])
            # print(inputs['labels'])
            # print(outputs[1].shape)
            # print(inputs['labels'].shape)
            # exit(0)

            tensor_logits = outputs[1]
            if not is_t5 and len(tensor_logits.shape) >= 3:
                tensor_logits = tensor_logits[:, idx]
            if is_t5:
                inputs['labels'] = inputs['labels'][:, -2]
                tensor_logits = tensor_logits[:, -2]

            logits = to_list(tensor_logits)
            label_ids = to_list(inputs['labels'])

            preds = to_list(tensor_logits.argmax(-1))
            all_pred_gold.extend(label_ids)
            all_pred_preds.extend(preds)

            if is_exp and annotation is not None:
                mare_att_mask = outputs.mare_att_mask[-1]
                if mare_att_mask.shape[-1] != tti_mask.shape[-1]:
                    mare_att_mask = torch.cat([mare_att_mask[..., 0:1],
                                           mare_att_mask[..., 1 - annotation.shape[-1]:]],
                                          dim=-1)

                if len(mare_att_mask.shape) == 3:
                    mare_att_mask = mare_att_mask[:, idx]
                all_tag_preds.extend(to_list((mare_att_mask * tti_mask).reshape(-1)))
                all_tag_gold.extend(to_list((annotation * tti_mask).reshape(-1)))

            tmp_eval_accuracy = accuracy(logits, label_ids)

            eval_accuracy += tmp_eval_accuracy
            eval_examples += inputs['labels'].size(0)

            assert eval_accuracy <= eval_examples, f'{eval_accuracy}, {eval_examples}'

    evalTime = timeit.default_timer() - start_time
    logger.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / (len(dataset)))

    precision_pred, recall_pred, f1_pred, _ = precision_recall_fscore_support(all_pred_gold, all_pred_preds,
                                                                              average='micro' if is_t5 else 'weighted' if is_eraser(
                                                                                  args) else 'binary')

    acc = eval_accuracy / eval_examples
    eval_token_f1 = f1_metric[-1].compute()
    if is_exp:
        # from AT-BMC
        if all_tag_gold:
            tag_p, tag_r, tag_f1, _ = precision_recall_fscore_support(
                np.array(all_tag_gold), np.array(all_tag_preds), labels=[1])

            logging.info(f'tag_p: {tag_p}, tag_r: {tag_r}, tag_f1: {tag_f1}')

        hidden_size = model.config.hidden_size
        num_hidden_layers = getattr(model.config, 'num_hidden_layers', None) or model.config.num_layers

        logger.info(f"evaluation, loss: {sum(all_loss) / len(all_loss)}")
        logger.info(f"acc: {acc}")
        logger.info(f"precision: {precision_pred}, recall: {recall_pred}, f1: {f1_pred}")
        logger.info(f"tokens_remained: {torch.mean(torch.stack(all_tokens_remained))}")
        logger.info(f"mare_loss: {torch.mean(torch.stack(all_mare_loss))}")
        metrics = [(i, e.compute()) for i, e in enumerate(f1_metric) if e.overall > 0]
        # sparses = all_layer_tokens_remained
        # sparses = [1.0 if 'rnp' not in args.model_type else sparses[-1]] * (num_hidden_layers - len(sparses)) + sparses
        if metrics:
            # assert len(metrics) == len(sparses), f'{len(metrics)}, {len(sparses)}'
            # for m, s in zip(metrics, sparses):
            for m in metrics:
                metric_string = ', '.join([f'{k}: {v:.4f}' for k, v in m[1].items()])
                logger.info(f"layer {m[0]}, {metric_string}")

        # else:
            # flops = 0
            # total_flops = (6 * hidden_size + args.max_seq_length) * num_hidden_layers
            # for s in sparses:
            #     flops += 6 * s * hidden_size + s ** 2 * args.max_seq_length

            # logger.info(f"speed up ratio {total_flops / flops: .4f}")

    else:
        logger.info(
            f"evaluation, acc: {eval_accuracy}, annotation_metrics: {f1_metric[-1].compute()}, precision: {precision_pred}, recall: {recall_pred}, f1: {f1_pred}")

    results = {'acc': acc, 'FLOPS': 2 * flops / len(dataset) / 1000000.0, }
    results.update(eval_token_f1)
    if 'mare' in args.model_type:
        results['tokens_remained'] = torch.mean(torch.stack(all_tokens_remained))
        for i, macs in enumerate(all_layer_tokens_remained):
            results[f'layers_{i}_tokens_remained'] = macs
    return results


def evaluate_multiple(args, model, tokenizer, prefix="", evaluate_prefix='dev'):
    for i, task_name in enumerate(args.task_name.split(',')):
        print('#####################', task_name, '#####################')
        results = evaluate(args, model, tokenizer, prefix=prefix, evaluate_prefix=evaluate_prefix,
                           aspect=int(task_name[-1]),
                           task_name=task_name, data_dir='./data/' + task_name[:-1], data_idx=i)

    return results


def evaluate_grad(args, model, tokenizer, prefix="", evaluate_prefix='train'
                  , aspect=None, task_name=None, data_dir=None, data_idx=None):
    dataset = load_and_cache_examples(args, tokenizer, prefix=evaluate_prefix, aspect=aspect,
                                      evaluate=True, task_name=task_name, data_dir=data_dir)

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=8)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    start_time = timeit.default_timer()
    max_seq_length = args.max_seq_length

    for batch in tqdm(eval_dataloader, desc="Evaluating"):

        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        inputs = {
            "input_ids": batch[0].long(),
            "attention_mask": batch[1].long(),
            "token_type_ids": batch[2].long(),
            "labels": batch[3].long(),
            "data_idx": data_idx,
        }

        if len(batch) > 4:
            inputs['query_length'] = (batch[-1] == 2).long().sum(-1)[0]
            if inputs['query_length'] == 0:
                inputs.pop('query_length')

        if args.train_rationales and len(batch) > 4:
            inputs['mare_att_mask'] = batch[-1]

        outputs = model(**inputs)
        loss = outputs[0] / len(eval_dataloader)

        loss.backward()
    from copy import deepcopy
    grads = {n: deepcopy(p.grad) for n, p in model.named_parameters()}

    evalTime = timeit.default_timer() - start_time
    logger.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(dataset))

    return grads


def evaluate_grad_multiple(args, model, tokenizer, prefix="", evaluate_prefix='dev'):
    import pickle
    from tqdm import tqdm
    grads = []
    length = len(args.task_name.split(','))
    if os.path.exists('grads.pkl'):
        import pickle
        with open('grads.pkl', 'rb') as f:
            grads = pickle.load(f)
    else:
        for i, task_name in enumerate(args.task_name.split(',')):
            print('#####################', task_name, '#####################')
            model.zero_grad()
            grads.append(evaluate_grad(args, model, tokenizer, prefix=prefix, evaluate_prefix=evaluate_prefix,
                                       aspect=int(task_name[-1]), task_name=task_name,
                                       data_dir='./data/' + ('beer' if 'beer' in task_name else 'hotel'),
                                       data_idx=i))
        with open('grads.pkl', 'wb') as f:
            pickle.dump(grads, f)
    cos_sims = []
    cos_sim_fn = torch.nn.CosineSimilarity(dim=0, eps=1e-6)

    keys = [grads[i].keys() for i in range(length)]
    keys = list(set(keys[0]))
    for k in tqdm(keys):
        if 'mare_predictors' in k:
            continue
        g = [grads[i][k] for i in range(length)]
        for i in range(length - 1):
            for j in range(i + 1, length):
                if g[i] is None and g[j] is None:
                    continue
                if g[i] is None:
                    print(k, i)
                if g[j] is None:
                    print(k, j)

                cos_sim = cos_sim_fn(g[i], g[j])
                cos_sims.append((k, i, j, cos_sim))

    with open('grads_sims.pkl', 'wb') as f:
        pickle.dump(cos_sims, f)
    # print(cos_sims)
    return cos_sims


def evaluate_logits(args, model, tokenizer, prefix="", evaluate_prefix='train'
                    , aspect=None, task_name=None, data_dir=None, data_idx=None):
    dataset = load_and_cache_examples(args, tokenizer, prefix=evaluate_prefix, aspect=aspect,
                                      evaluate=True, task_name=task_name, data_dir=data_dir)

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=8)

    # multi-gpu evaluate
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    eval_accuracy = eval_examples = 0

    # macs_list = json.load(open('macs_list.json'))
    flops = 0
    bert_flops = 0
    distilbert_flops = 0
    start_time = timeit.default_timer()
    # token_f1 = OptimizedTokenF1(tokenizer).to(model.device)
    token_f1 = TokenF1().to(model.device)
    is_t5=False
    overall = []
    data_idx = (data_idx + 0) % 3
    idx = data_idx + (1 if hasattr(model, 'mare_att') and model.mare_att % 2 else 0)
    idx = (idx + 0) % 3
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()

        with torch.no_grad():
            inputs = construct_data(args, batch, data_idx)

            outputs = model(**inputs)



            # print(outputs.mare_att_mask[-1][0][0])
            # print(outputs.mare_att_mask[-1][0][1])
            # print(outputs.mare_att_mask[-1][0][2])
            # # for i in range(3):
            # #     tokens = []
            # #     for a, b, c in zip(inputs['input_ids'][i],
            # #                        mare_att_mask[i],
            # #                        annotation[i]):
            # #         if a == tokenizer.pad_token_id:
            # #             break
            # #         b = b.long() > 0
            # #         c = c > 0
            # #
            # #         prefix = ''
            # #         if b and b == c:
            # #             prefix = '++'
            # #         elif b:
            # #             prefix = '--'
            # #         elif c:
            # #             prefix = '**'
            # #         tokens.append(f'{prefix}{"".join(tokenizer.decode(a).split())}{prefix}')
            # #
            # #     logger.info(' '.join(tokens))
            # exit(0)

            if 'token_type_ids' in inputs and inputs['token_type_ids'][0].sum() > 0:
                tti_mask = inputs['token_type_ids']
            else:
                tti_mask = torch.ones_like(inputs['token_type_ids'])
                if not is_t5: tti_mask[:, 0] = 0
                tti_mask = tti_mask * inputs['attention_mask']

            tensor_logits = outputs[1]
            if not is_t5 and len(tensor_logits.shape) >= 3:
                tensor_logits = tensor_logits[:, idx]
            if is_t5:
                inputs['labels'] = inputs['labels'][:, -2]
                tensor_logits = tensor_logits[:, -2]

            eval_accuracy += accuracy(tensor_logits.cpu().numpy(), batch['labels'].cpu().numpy())
            eval_examples += len(batch['labels'])

            mare_att_mask = outputs.mare_att_mask[-1] if isinstance(outputs.mare_att_mask,
                                                            (list, tuple)) else outputs.mare_att_mask

            annotation = batch.get('annotation', torch.zeros_like(inputs['input_ids'])).to(model.device)
            if len(mare_att_mask.shape) == 3:
                mare_att_mask = mare_att_mask[:, idx]
            if mare_att_mask.shape[-1] != tti_mask.shape[-1]:
                mare_att_mask = torch.cat([mare_att_mask[..., 0:1],
                                       mare_att_mask[..., 1 - annotation.shape[-1]:]],
                                      dim=-1)

            token_f1(
                     mare_att_mask,
                     annotation,
                     tti_mask
                     # , inputs['input_ids']
                     )
            # token_f1(mare_att_mask, batch[-1], inputs['attention_mask'] * (1 if inputs['token_type_ids'].sum()<1e-3 else inputs['token_type_ids']))
            for i, (data, mare_att_mask, rationale) in enumerate(
                    zip(batch['input_ids'].long(), mare_att_mask, annotation.long())):
                assert len(data) == len(mare_att_mask) and len(mare_att_mask) == len(rationale), f'{len(data)} {len(mare_att_mask)} {len(rationale)}'
                cur_sample = []
                for a, b, c in zip(data, mare_att_mask, rationale):
                    if a == tokenizer.pad_token_id:
                        break
                    cur_sample.append(
                        (''.join(tokenizer.decode(a).split()), (b.long() > 0).item(), (c.long() > 0).item()))

                predictions = tensor_logits[i].cpu().argmax(-1).item()
                labels = batch['labels'][i].cpu().item()
                overall.append((cur_sample, predictions, labels))
    precision_pred, recall_pred, f1_pred, _ = precision_recall_fscore_support([e[2] for e in overall],
                                                                              [e[1] for e in overall],
                                                                              average='weighted' if is_eraser(
                                                                                  args) else 'micro')

    logger.info(f'metrics: {token_f1.compute()}, acc: {eval_accuracy / eval_examples}')
    logger.info(f'precision {precision_pred}, recall {recall_pred}, f1 {f1_pred}')

    evalTime = timeit.default_timer() - start_time

    return overall


def evaluate_logits_multiple(args, model, tokenizer, prefix="", evaluate_prefix='test'):
    assert evaluate_prefix == 'test'
    import json
    from tqdm import tqdm

    length = len(args.task_name.split(','))
    path = f'decode_{args.task_name}.json'

    overall = []
    for i, task_name in enumerate(args.task_name.split(',')):
        print('#####################', task_name, '#####################')
        model.zero_grad()
        overall.append(evaluate_logits(args, model, tokenizer, prefix=prefix, evaluate_prefix=evaluate_prefix,
                                       aspect=int(task_name[-1]), task_name=task_name, data_dir=args.data_dir,
                                       data_idx=i))
    with open(path, 'w') as f:
        json.dump(overall, f)
        logger.info(f"save ans to {path}")

    return overall


def predict_rationales(args, model, tokenizer, prefix="", evaluate_prefix='train'
                       , aspect=None, task_name=None, data_dir=None, data_idx=None):
    dataset = load_and_cache_examples(args, tokenizer, prefix=evaluate_prefix, aspect=aspect, evaluate=True)

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=8)

    # multi-gpu evaluate
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    eval_accuracy = eval_examples = 0

    is_t5 = 't5_' in args.model_type
    start_time = timeit.default_timer()
    # token_f1 = OptimizedTokenF1(tokenizer).to(model.device)
    token_f1 = TokenF1().to(model.device)
    print(model.device)
    overall = []
    data_idx = data_idx or 0
    idx = data_idx + (1 if model.mare_att % 2 else 0)
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()

        with torch.no_grad():
            inputs = construct_data(args, batch)

            outputs = model(**inputs)

            tensor_logits = outputs[1][:, 0] if is_t5 else outputs[1]

            if isinstance(tensor_logits, (list, tuple)):
                tensor_logits = tensor_logits[idx]

            logits = to_list(tensor_logits)
            label_ids = to_list(inputs['labels'])

            preds = to_list(tensor_logits.argmax(-1))

            acc = accuracy(logits, label_ids)
            eval_accuracy += acc
            eval_examples += len(label_ids)

            mare_att_mask = outputs.mare_att_mask[-1] if isinstance(outputs.mare_att_mask,
                                                            (list, tuple)) else outputs.mare_att_mask

            for i, (data, mare_att_mask) in enumerate(zip(batch['input_ids'].long(), mare_att_mask)):
                assert len(data) == len(mare_att_mask), f'{len(data)} {len(mare_att_mask)}'
                predictions = outputs[1][i].cpu().argmax(-1).item()
                labels = batch['labels'][i].cpu().item()
                overall.append((to_list(data), to_list(mare_att_mask), predictions, labels))

    print(f'metrics: {token_f1.compute()}, acc: {eval_accuracy / eval_examples}')

    evalTime = timeit.default_timer() - start_time

    return overall


def main():
    args = parse_args()

    if (
            os.path.exists(args.output_dir)
            and os.listdir(args.output_dir)
            and args.do_train
            and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    def create_exp_dir(path, scripts_to_save=None):
        if args.output_dir.find("test") != -1:
            return
        if not os.path.exists(path):
            os.mkdir(path)

        print('Experiment dir : {}'.format(path))
        if scripts_to_save is not None:
            if not os.path.exists(os.path.join(path, 'scripts')):
                os.mkdir(os.path.join(path, 'scripts'))
            for script in scripts_to_save:
                dst_file = os.path.join(path, 'scripts', os.path.basename(script))
                shutil.copyfile(script, dst_file)

    # if args.do_train and args.local_rank in [-1, 0] and args.output_dir.find('test')==-1:
    #     create_exp_dir(args.output_dir, scripts_to_save=['run_classification.py', 'transformers/src/transformers/modeling_autobert.py',
    #         'transformers/src/transformers/modeling_distilautobert.py'])

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    args.task_name = args.task_name.lower()

    def get_num_labels(task_name):
        if task_name == '20news':
            num_labels = 20
        elif task_name == 'yelp':
            num_labels = 5
        elif 'legal_small' in task_name:
            mp = [94, 115, 13]
            num_labels = mp[int(task_name[-1])]
        elif 'legal_big' in task_name:
            mp = [118, 129, 13]
            num_labels = mp[int(task_name[-1])]
        elif 'evidence_inference' in task_name:
            num_labels = 3
        else:
            num_labels = 2

        return num_labels

    num_labels = [get_num_labels(e) for e in args.task_name.split(',')]
    num_labels = max(num_labels)

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    if args.do_train:
        config = config_class.from_pretrained(
            args.config_name if args.config_name else args.model_name_or_path,
            num_labels=num_labels,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        tokenizer = tokenizer_class.from_pretrained(
            args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
            do_lower_case=args.do_lower_case,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
    else:
        config = config_class.from_pretrained(
            args.output_dir,
            num_labels=num_labels,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        tokenizer = tokenizer_class.from_pretrained(
            args.output_dir,
            do_lower_case=args.do_lower_case,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )

    print(args)

    if args.do_eval_grad:
        config.output_hidden_states = True

    #######
    if args.do_train:
        get_train_config(args, config)

    #######
    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    if args.do_train and config.mare_att and config.cluster_is_cls == 2:
        tok_id = tokenizer.cls_token_id
        tensor = model.bert.embeddings.word_embeddings.weight[tok_id]
        with torch.no_grad():
            model.bert.clusters.weight.data = tensor.expand_as(model.bert.clusters.weight)


    def copy_weight(module, target):
        if module is None:
            return
        for p1, p2 in zip(module.parameters(), target.parameters()):
            p1.data.copy_(p2.data)
        # logging.info('%s initialized with %s' % (aim, target))

    if args.do_train and 'rnp' == args.model_type:
        if args.use_single_backbone:
            logging.info('FR loaded!!!')
            model.bert_cls = model.bert
        else:
            logging.info('RNP loaded!!!')
            model.bert_cls = deepcopy(model.bert)

    ####### moe initialize in bert #######
    num_tasks = getattr(config, 'num_tasks', 1)

    if args.local_rank == 0:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    model.to(args.device)

    # TODO: fix error
    # import torch._dynamo as dynamo
    # dynamo.config.verbose=True
    # dynamo.config.suppress_errors=True
    # model = torch.compile(model)

    def count_trainable_parameters(model):
        total = 0
        trainable = 0
        for n, p in model.named_parameters():
            trainable += p.numel() if p.requires_grad else 0
            total += p.numel()
        logger.info('trainable: %s', trainable)
        logger.info('total: %s', total)
        logger.info('trainable ratio: %s', trainable / total)

    logger.info("Training/evaluation parameters %s", args)
    count_trainable_parameters(model)

    # Before we do anything with models, we want to ensure that we get fp16 execution of torch.einsum if args.fp16 is set.
    # Otherwise it'll default to "promote" mode, and we'll get fp32 operations. Note that running `--fp16_opt_level="O2"` will
    # remove the need for this code, but it is still valid.
    if args.fp16:
        try:
            import apex

            apex.amp.register_half_function(torch, "einsum")
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
    print(type(model))
    # Training
    if args.do_train:
        if len(args.task_name.split(',')) == 1:
            train_dataset = load_and_cache_examples(args, tokenizer, prefix='train',
                                                    aspect=int(args.task_name[-1]) if check(args.task_name) else None)
            global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        else:
            def get_num_len(x:str):
                i = 1
                while x[-i:].isdigit():
                    i+=1
                return i-1
            train_datasets = [load_and_cache_examples(args, tokenizer, prefix='train',
                                                      aspect=int(task_name[-1]) if check(task_name) else None,
                                                      task_name=task_name, data_dir='./data/' + task_name[:-get_num_len(task_name)]) for
                              task_name in args.task_name.split(',')]
            global_step, tr_loss = train_multiple(args, train_datasets, model, tokenizer)

        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Save the trained model and the tokenizer
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):

        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        # Take care of distributed/parallel training
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir, force_download=True)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        model.to(args.device)

    # Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        ckpts = filter(lambda x: x.startswith('checkpoint-'), os.listdir(args.output_dir))
        checkpoints = [os.path.join(args.output_dir, ckpt) for ckpt in ckpts]
        try:
            checkpoint = list(sorted(checkpoints, key=lambda x: int(x.split('-')[-1])))[-1]  # best checkpoint
            checkpoints = [args.output_dir, checkpoint]
        except:
            checkpoints = [args.output_dir]
        print(args.dataloader_wrapper)
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            # Reload the model
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            try:
                model = model_class.from_pretrained(checkpoint, force_download=True)
            except OSError:
                print('NOT FOUND: ' + checkpoint)
                continue
            model.to(args.device)

            if len(args.task_name.split(',')) == 1:
                # Evaluate
                result = evaluate(args, model, tokenizer, prefix=checkpoint,
                                  evaluate_prefix='test' if check(args.task_name) else 'dev',
                                  aspect=int(args.task_name[-1]) if check(args.task_name) else None)
                evaluate(args, model, tokenizer, prefix=checkpoint, evaluate_prefix='dev',
                         aspect=int(args.task_name[-1]) if check(args.task_name) else None)
            else:
                result = evaluate_multiple(args, model, tokenizer, prefix=checkpoint,
                                           evaluate_prefix='test' if check(args.task_name) else 'dev', )
                # evaluate_multiple(args, model, tokenizer, prefix=global_step,evaluate_prefix='dev',)

            result = dict((k + ("_{}".format(global_step) if global_step else ""), v) for k, v in result.items())
            results.update(result)

    if args.do_eval_grad and args.local_rank in [-1, 0]:

        checkpoints = [args.output_dir]
        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            # Reload the model
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint, force_download=True, config=config)
            model.to(args.device)

            # Evaluate
            result = evaluate_grad_multiple(args, model, tokenizer, prefix=checkpoint, evaluate_prefix='dev')

    if args.do_eval_logits and args.local_rank in [-1, 0]:

        checkpoints = [args.output_dir]
        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            # Reload the model
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint, force_download=True, config=config)
            model.to(args.device)

            # Evaluate
            result = evaluate_logits_multiple(args, model, tokenizer, prefix=checkpoint, evaluate_prefix='test')


    if args.do_predict_rationales and args.local_rank in [-1, 0]:
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            # Reload the model
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            try:
                model = model_class.from_pretrained(checkpoint, force_download=True)
            except OSError:
                print('NOT FOUND: ' + checkpoint)
                continue
            model.to(args.device)

            # Evaluate
            result = predict_rationales(args, model, tokenizer, prefix=checkpoint,
                                        evaluate_prefix='train',
                                        aspect=int(args.task_name[-1]) if check(args.task_name) else None)

            with open(f'rationales_{args.task_name}_train.json', 'w') as f:
                json.dump([result], f)

            result = predict_rationales(args, model, tokenizer, prefix=checkpoint,
                                        evaluate_prefix='dev',
                                        aspect=int(args.task_name[-1]) if check(args.task_name) else None)

            with open(f'rationales_{args.task_name}_dev.json', 'w') as f:
                json.dump([result], f)

            result = predict_rationales(args, model, tokenizer, prefix=checkpoint,
                                        evaluate_prefix='test',
                                        aspect=int(args.task_name[-1]) if check(args.task_name) else None)

            with open(f'rationales_{args.task_name}_test.json', 'w') as f:
                json.dump([result], f)

    logger.info("Results: {}".format(results))

    return results


def get_train_config(args, config):
    def parse_x_to_type(x, type):
        return list(map(type, x.split(',')))

    def get_selection(x, num_layers, decay_num, gamma, sparsity_control, decay_type):
        # cliff decay
        if decay_type == 'cliff':
            selection = [1] * x + [sparsity_control] * (num_layers - x)

        # linear decay
        elif decay_type == 'linear':
            # selection = [1] * x + [1 - (1 - args.sparsity_control) / (config.num_hidden_layers - x) * (i - x) for i in range(x+1, config.num_hidden_layers+1)]
            selection = [1] * x + [gamma - (gamma - sparsity_control) / decay_num * (i - x) for i in
                                   range(x + 1, x + 1 + decay_num)] + [sparsity_control] * (num_layers - x - decay_num)

        # exp decay
        elif decay_type == 'exp':
            # selection = [1] * x + [(args.sparsity_control ** ((i-x)/(config.num_hidden_layers-x))) for i in range(x+1, config.num_hidden_layers+1)]
            # selection = [1] * x + [np.exp((np.log(gamma)*(config.num_hidden_layers-i) + np.log(args.sparsity_control)*(i-x))/(config.num_hidden_layers-x)) for i in range(x+1, config.num_hidden_layers+1)]
            selection = [1] * x + [
                np.exp((np.log(gamma) * (x + decay_num - i) + np.log(sparsity_control) * (i - x)) / decay_num) for i in
                range(x + 1, x + 1 + decay_num)] + [sparsity_control] * (num_layers - x - decay_num)

        # log decay
        elif decay_type == 'log':
            # selection = [1] * x + [(np.log(np.exp(args.sparsity_control)*(i-x) + np.exp(gamma)*(config.num_hidden_layers-i))-np.log(config.num_hidden_layers-x)) for i in range(x+1, config.num_hidden_layers+1)]
            selection = [1] * x + [
                (np.log(np.exp(sparsity_control) * (i - x) + np.exp(gamma) * (x + decay_num - i)) - np.log(decay_num))
                for i in range(x + 1, x + 1 + decay_num)] + [sparsity_control] * (num_layers - x - decay_num)

        # platau decay
        elif decay_type == 'plateau':
            selection = [1] * x + [gamma] * decay_num + [sparsity_control] * (num_layers - decay_num - x)
            selection[-1] = sparsity_control

        # unknown decay
        else:
            raise NotImplementedError

        return selection

    def pad_args(*args, max_len=None):
        if max_len is None:
            max_len = 0
            for arg in args:
                if isinstance(arg, (list, tuple)):
                    max_len = max(max_len, len(arg))
                else:
                    max_len = max(max_len, 1)

        args = list(args)
        for i in range(len(args)):
            if isinstance(args[i], (list, tuple)):
                assert len(args[i]) == 1 or max_len == len(
                    args[i]), f'{args[i]} has length {len(args[i])} and mxn is {max_len}.'
                if len(args[i]) == 1:
                    args[i] = [args[i][0] for _ in range(max_len)]
            else:
                args[i] = [args[i] for _ in range(max_len)]

        return args

    def process(s, config, many=False):
        s = s.strip(';').split(';')
        print(s)
        if len(s) == 1 and s[0].isdigit():
            if bool(int(s[0])):
                if many: return [list(range(config.num_tasks)) for _ in range(config.num_hidden_layers)]
                return list(range(config.num_tasks))
            else:
                if many: return [[0] * config.num_tasks for _ in range(config.num_hidden_layers)]
                return [0] * config.num_tasks
        s = [list(map(int, s.strip().split(','))) for s in s]

        for a in s:
            cur_length = len(set(a))
            for e in a:
                assert e < cur_length, f'{e}, {cur_length}'

        if len(s) == 1:
            s = s[0]

        return s

    config.num_tasks = len(args.task_name.split(','))
    print('num_tasks', config.num_tasks)
    x = parse_x_to_type(args.remain_layers, int)
    decay_num = parse_x_to_type(args.decay_num, int)  # for RG part
    gamma = parse_x_to_type(args.gamma, float)  # for initial sparsity
    decay_type = parse_x_to_type(args.decay_type, str)
    sparsity_control = parse_x_to_type(args.sparsity_control, float)
    pretrain_steps = parse_x_to_type(args.pretrain_steps, int)
    sparse_coefficient = parse_x_to_type(args.sparse_coefficient, float)
    contiguous_coefficient = parse_x_to_type(args.contiguous_coefficient, float)
    multiply_mask = parse_x_to_type(args.multiply_mask, int)[0]
    overall_only = parse_x_to_type(args.overall_only, int)[0]
    # print(config)
    num_layers = getattr(config, 'num_hidden_layers', None) or config.num_layers
    (x,
     decay_num,
     gamma,
     decay_type,
     sparsity_control,
     num_layers,
     pretrain_steps,
     sparse_coefficient,
     contiguous_coefficient) = pad_args(x,
                                        decay_num,
                                        gamma,
                                        decay_type,
                                        sparsity_control,
                                        num_layers,
                                        pretrain_steps,
                                        sparse_coefficient,
                                        contiguous_coefficient,
                                        max_len=max(args.num_clusters, config.num_tasks))  # -> tuples
    selection = [get_selection(x[i], num_layers[i], decay_num[i], gamma[i], sparsity_control[i], decay_type[i])
                 for i in range(len(x))]

    ######
    config.multiply_mask = bool(multiply_mask)
    config.pretrain_steps = pretrain_steps
    config.sparse_coefficient = sparse_coefficient
    config.contiguous_coefficient = contiguous_coefficient
    config.selection = selection
    config.use_single_backbone = bool(args.use_single_backbone)
    config.share_head = bool(args.share_head)
    config.constrain_final = bool(args.constrain_final)
    config.mean_bias = args.mean_bias
    config.mare_att = args.mare_att
    config.num_clusters = args.num_clusters
    config.process_type = args.process_type
    config.overall_only = bool(overall_only)
    config.mean_first = bool(args.mean_first)
    config.mlp_head = bool(args.mlp_head)
    config.refined_clusters = config.num_clusters + (2 if config.mare_att % 2 else 0)
    config.dataloader_wrapper=args.dataloader_wrapper
    config.cluster_is_cls=args.cluster_is_cls

    # print(config.selection)
    # print(config.sparse_coefficient)
    # print(config.contiguous_coefficient)
    # exit(0)
    return config


def parse_args():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .csv files (or other data files) for the task.")

    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        help="task name in [imdb, yelp_p, yelp_f, hyperpartisan]",
    )

    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS),
    )
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=512,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")

    parser.add_argument("--max_chunks",
                        default=1,
                        type=int,
                        help="The maximum total input sequence chunks after WordPiece tokenization. \n"
                        )

    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the test set.")
    parser.add_argument("--train_contrastive",
                        default=False,
                        action='store_true',
                        help=" ")
    parser.add_argument("--train_rl",
                        default=False,
                        action='store_true',
                        help=" ")
    parser.add_argument("--train_both",
                        default=False,
                        action='store_true',
                        help=" ")
    parser.add_argument("--do_eval_grad",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict_rationales",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training",
                        default=False,
                        action='store_true',
                        help="")
    parser.add_argument(
        "--evaluate_steps",
        type=int,
        default=500,
        help="Whether do evaluation after training",
    )
    parser.add_argument("--do_lower_case",
                        default=False,
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
    )

    parser.add_argument("--learning_rate",
                        default=3e-5,
                        type=float,
                        help="The initial learning rate for Adam.")

    parser.add_argument("--alpha",
                        default=1,
                        type=float,
                        help="")

    parser.add_argument("--guide_rate",
                        default=0.5,
                        type=float,
                        help="")

    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=-1, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--logging_steps", type=int, default=5, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=100000, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")

    parser.add_argument("--train_init",
                        default=False,
                        action='store_true',
                        help="option, initizlize the policy network")

    parser.add_argument("--train_teacher",
                        default=False,
                        action='store_true',
                        help="train with KD")

    parser.add_argument("--do_eval_logits",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the dev set.")

    parser.add_argument("--sparse_coefficient", default="1", type=str, help="Sparsity coefficient for mare.")
    parser.add_argument("--contiguous_coefficient", default="1", type=str,
                        help="Contiguous coefficient for mare.")
    parser.add_argument("--remain_layers", default="3", type=str, help="")
    parser.add_argument("--sparsity_control", default="0.13", type=str, help="")
    parser.add_argument("--use_single_backbone", default=0, type=int, help="")
    parser.add_argument("--pretrain_steps", default="0", type=str, help="")
    parser.add_argument("--decay_num", default="0", type=str, help="")
    parser.add_argument("--gamma", default="1", type=str, help="")
    parser.add_argument("--decay_type", default='cliff', type=str, help="")
    parser.add_argument("--share_head", default=0, type=int, help="")
    parser.add_argument("--constrain_final", default=0, type=int, help="")
    parser.add_argument("--dataloader_wrapper", default='round', type=str,
                        choices=list(wrapper_map.keys()))
    parser.add_argument("--train_rationales", action='store_true', default=False, help="")
    parser.add_argument("--num_samples", type=int, default=-1, help="")
    parser.add_argument("--conditioned", default=False, type=bool, help="")
    parser.add_argument("--multiply_mask", default='0', type=str, help="")
    parser.add_argument("--mean_bias", default=0.5, type=float, help="")
    parser.add_argument("--mare_att", default=0, type=int, help="")
    parser.add_argument("--num_clusters", default=0, type=int, help="")
    parser.add_argument("--process_type", default=0, type=int, help="")
    parser.add_argument("--overall_only", default="0", type=str, help="")
    parser.add_argument("--mean_first", default=0, type=int, help="")
    parser.add_argument("--mlp_head", default=1, type=int, help="")
    parser.add_argument("--annotation_ratio", default=None, type=float, help="")
    parser.add_argument("--balance", default=None, type=int, help="")
    parser.add_argument("--cluster_is_cls", default=0, type=int, help="")

    args = parser.parse_args()
    return args


# BERT Perlayer: 24*L*D^2 + 4*L^2*D
# L -> pL : (24*L*D^2 + 4*L^2*D) / (24*p*L*D^2 + 4*p^2*L^2*D)
# = (6*D + L) / (6*p*D + p^2*L)

if __name__ == "__main__":
    main()
