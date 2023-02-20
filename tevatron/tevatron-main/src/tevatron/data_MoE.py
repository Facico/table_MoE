import random
from dataclasses import dataclass
from re import X
from typing import List, Tuple

import datasets
from torch.utils.data import Dataset
from tevatron.driver.transformers_MoE import PreTrainedTokenizer, BatchEncoding, DataCollatorWithPadding
import torch

from .arguments_MoE import DataArguments
from .trainer_MoE import DenseTrainer_MoE
from tevatron.datasets.preprocessor_MoE import seq_to_table
import logging
logger = logging.getLogger(__name__)


class TrainDataset_MoE(Dataset):
    def __init__(
            self,
            data_args: DataArguments,
            dataset: datasets.Dataset,
            tokenizer: PreTrainedTokenizer,
            trainer: DenseTrainer_MoE = None,
    ):
        self.train_data = dataset
        self.tok = tokenizer
        self.trainer = trainer

        self.data_args = data_args
        self.total_len = len(self.train_data)

    def create_one_example(self, text_encoding: List[int], is_query=False):
        item = self.tok.encode_plus(
            text_encoding,
            truncation='only_first',
            max_length=self.data_args.q_max_len if is_query else self.data_args.p_max_len,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return item

    def __len__(self):
        return self.total_len

    def __getitem__(self, item) -> Tuple[BatchEncoding, List[BatchEncoding]]:
        group = self.train_data[item]
        epoch = int(self.trainer.state.epoch)

        _hashed_seed = hash(item + self.trainer.args.seed)

        qry = group['query']
        encoded_query = self.create_one_example(qry, is_query=True)

        encoded_passages = []
        passages_types = []

        group_positives = group['positives']
        group_negatives = group['negatives']
        positives_types = group['positives_types']
        negatives_types = group['negatives_types']

        if self.data_args.positive_passage_no_shuffle:
            pos_psg = group_positives[0]
            pos_type = positives_types[0]
        else:
            pos_psg = group_positives[(_hashed_seed + epoch) % len(group_positives)]
            pos_type = positives_types[(_hashed_seed + epoch) % len(group_positives)]

        encoded_passages.append(self.create_one_example(pos_psg))
        passages_types.append(pos_type)

        negative_size = self.data_args.train_n_passages - 1
        if len(group_negatives) < negative_size:
            negs_index = random.choice([i for i in range(len(group_negatives))], k=negative_size)
            negs = [group_negatives[i] for i in negs_index]
            negs_types = [negatives_types[i] for i in negs_index]

        elif self.data_args.train_n_passages == 1:
            negs = []
            negs_types = []
        elif self.data_args.negative_passage_no_shuffle:
            negs = group_negatives[:negative_size]
            negs_types = negatives_types[:negative_size]
        else:
            negs_index = [i for i in range(len(group_negatives))]
            random.Random(_hashed_seed).shuffle(negs_index)
            _offset = epoch * negative_size % len(group_negatives)

            negs = [group_negatives[i] for i in negs_index]
            negs = negs * 2
            negs_index = negs_index * 2
            negs = negs[_offset: _offset + negative_size]
            negs_index = negs_index[_offset: _offset + negative_size]
            negs_types = [negatives_types[i] for i in negs_index]

        for neg_psg in negs:
            encoded_passages.append(self.create_one_example(neg_psg))
        for neg_type in negs_types:
            passages_types.append(neg_type)

        return encoded_query, encoded_passages, passages_types


class EncodeDataset_MoE(Dataset):
    input_keys = ['text_id', 'text', 'data_type']

    def __init__(self, dataset: datasets.Dataset, tokenizer: PreTrainedTokenizer, max_len=128):
        self.encode_data = dataset
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.encode_data)

    def __getitem__(self, item) -> Tuple[str, BatchEncoding]:
        if 'data_type' not in self.encode_data[item]:
            text_id, text = self.encode_data[item]['text_id'], self.encode_data[item]['text']
            data_type = None
        else:
            text_id, text, data_type = (self.encode_data[item][f] for f in self.input_keys)
        encoded_text = self.tok.encode_plus(
            text,
            max_length=self.max_len,
            truncation='only_first',
            padding=False,
            return_token_type_ids=False,
        )
        return text_id, encoded_text, data_type

class EncodeDataset_tapas_MoE(Dataset):
    input_keys = ['text_id', 'text', 'data_type']

    def __init__(self, dataset: datasets.Dataset, tokenizer_text: PreTrainedTokenizer, tokenizer_table: PreTrainedTokenizer, max_len=128):
        self.encode_data = dataset
        self.tok_text = tokenizer_text
        self.tok_table = tokenizer_table
        self.max_len = max_len

    def __len__(self):
        return len(self.encode_data)

    def __getitem__(self, item) -> Tuple[str, BatchEncoding]:
        if 'data_type' not in self.encode_data[item]:
            text_id, text = self.encode_data[item]['text_id'], self.encode_data[item]['text']
            data_type = None
        else:
            text_id, text, data_type = (self.encode_data[item][f] for f in self.input_keys)
        if data_type is None or self.encode_data[item]['data_type'] == 0:
            encoded_text = self.tok_text.encode_plus(
                text,
                max_length=self.max_len,
                truncation='only_first',
                padding=False,
                return_token_type_ids=False,
            )
        else:
            encoded_text = self.tok_text.encode_plus(
                text,
                max_length=self.max_len,
                truncation='only_first',
                padding=False,
                return_token_type_ids=False,
            )
        return text_id, encoded_text, data_type


@dataclass
class QPCollator_MoE(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    max_q_len: int = 32
    max_p_len: int = 128

    def __call__(self, features):
        qq = [f[0] for f in features]
        dd = [f[1] for f in features]
        MoE_types = [f[2] for f in features]

        if isinstance(qq[0], list):
            qq = sum(qq, [])
        if isinstance(dd[0], list):
            dd = sum(dd, [])
        if isinstance(MoE_types[0], list):
            MoE_types = sum(MoE_types, [])
        q_collated = self.tokenizer.pad(
            qq,
            padding='max_length',
            max_length=self.max_q_len,
            return_tensors="pt",
        )
        d_collated = self.tokenizer.pad(
            dd,
            padding='max_length',
            max_length=self.max_p_len,
            return_tensors="pt",
        )

        MoE_types = {'p_MoE_type_tensor': torch.tensor([[x] for x in MoE_types])}
        return q_collated, d_collated, MoE_types


@dataclass
class EncodeCollator_MoE(DataCollatorWithPadding):
    def __call__(self, features):
        text_ids = [x[0] for x in features]
        text_features = [x[1] for x in features]
        MoE_types = [x[2] for x in features]
        collated_features = super().__call__(text_features)
        if MoE_types[0] is not None:
            MoE_types = torch.tensor(MoE_types, device='cuda:0')
        return text_ids, collated_features, MoE_types