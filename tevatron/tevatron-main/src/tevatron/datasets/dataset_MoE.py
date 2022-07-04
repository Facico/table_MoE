from datasets import load_dataset
from tevatron.driver.transformers_MoE import PreTrainedTokenizer
from .preprocessor_MoE import TrainPreProcessor_MoE, QueryPreProcessor_MoE, CorpusPreProcessor_MoE, CorpusPreProcessor_tapas_MoE
from ..arguments_MoE import DataArguments

DEFAULT_PROCESSORS = [TrainPreProcessor_MoE, QueryPreProcessor_MoE, CorpusPreProcessor_MoE]
PROCESSOR_INFO = {
    'Tevatron/wikipedia-nq': DEFAULT_PROCESSORS,
    'Tevatron/wikipedia-trivia': DEFAULT_PROCESSORS,
    'Tevatron/wikipedia-curated': DEFAULT_PROCESSORS,
    'Tevatron/wikipedia-wq': DEFAULT_PROCESSORS,
    'Tevatron/wikipedia-squad': DEFAULT_PROCESSORS,
    'Tevatron/scifact': DEFAULT_PROCESSORS,
    'Tevatron/msmarco-passage': DEFAULT_PROCESSORS,
    'json': [None, None, None],
    'json_not_encode': DEFAULT_PROCESSORS,
}


class HFTrainDataset_MoE:
    def __init__(self, tokenizer: PreTrainedTokenizer, data_args: DataArguments, cache_dir: str):
        data_files = data_args.train_path
        if data_files:
            data_files = {data_args.dataset_split: data_files}
        self.dataset = load_dataset(data_args.dataset_name,
                                    data_args.dataset_language,
                                    data_files=data_files, cache_dir=cache_dir)[data_args.dataset_split]
        self.preprocessor = PROCESSOR_INFO[data_args.dataset_name][0] if data_args.dataset_name in PROCESSOR_INFO\
            else DEFAULT_PROCESSORS[0]
        self.tokenizer = tokenizer
        self.q_max_len = data_args.q_max_len
        self.p_max_len = data_args.p_max_len
        self.proc_num = data_args.dataset_proc_num
        self.neg_num = data_args.train_n_passages - 1
        self.separator = getattr(self.tokenizer, data_args.passage_field_separator, data_args.passage_field_separator)

    def process(self, shard_num=1, shard_idx=0):
        self.dataset = self.dataset.shard(shard_num, shard_idx)
        if self.preprocessor is not None:
            self.dataset = self.dataset.map(
                self.preprocessor(self.tokenizer, self.q_max_len, self.p_max_len, self.separator),
                batched=False,
                num_proc=self.proc_num,
                remove_columns=self.dataset.column_names,
                desc="Running tokenizer on train dataset",
            )
        return self.dataset


class HFQueryDataset_MoE:
    def __init__(self, tokenizer: PreTrainedTokenizer, data_args: DataArguments, cache_dir: str):
        data_files = data_args.encode_in_path
        if data_files:
            data_files = {data_args.dataset_split: data_files}
        self.dataset = load_dataset(data_args.dataset_name,
                                    data_args.dataset_language,
                                    data_files=data_files, cache_dir=cache_dir)[data_args.dataset_split]
        self.preprocessor = PROCESSOR_INFO[data_args.dataset_name][1] if data_args.dataset_name in PROCESSOR_INFO \
            else DEFAULT_PROCESSORS[1]
        self.tokenizer = tokenizer
        self.q_max_len = data_args.q_max_len
        self.proc_num = data_args.dataset_proc_num

    def process(self, shard_num=1, shard_idx=0):
        self.dataset = self.dataset.shard(shard_num, shard_idx)
        if self.preprocessor is not None:
            self.dataset = self.dataset.map(
                self.preprocessor(self.tokenizer, self.q_max_len),
                batched=False,
                num_proc=self.proc_num,
                remove_columns=self.dataset.column_names,
                desc="Running tokenization",
            )
        return self.dataset


class HFCorpusDataset_MoE:
    def __init__(self, tokenizer: PreTrainedTokenizer, data_args: DataArguments, cache_dir: str):
        data_files = data_args.encode_in_path
        if data_files:
            data_files = {data_args.dataset_split: data_files}
        script_prefix = data_args.dataset_name
        if script_prefix.endswith('-corpus'):
            script_prefix = script_prefix[:-7]
        self.preprocessor = PROCESSOR_INFO[script_prefix][2] \
            if script_prefix in PROCESSOR_INFO else DEFAULT_PROCESSORS[2]
        if data_args.dataset_name == 'json_not_encode':
            data_args.dataset_name = 'json'
        self.dataset = load_dataset(data_args.dataset_name,
                                    data_args.dataset_language,
                                    data_files=data_files, cache_dir=cache_dir)[data_args.dataset_split]
        
        self.tokenizer = tokenizer
        self.p_max_len = data_args.p_max_len
        self.proc_num = data_args.dataset_proc_num
        self.separator = getattr(self.tokenizer, data_args.passage_field_separator, data_args.passage_field_separator)

    def process(self, shard_num=1, shard_idx=0):
        self.dataset = self.dataset.shard(shard_num, shard_idx)
        print(self.preprocessor)
        if self.preprocessor is not None:
            self.dataset = self.dataset.map(
                self.preprocessor(self.tokenizer, self.p_max_len, self.separator),
                batched=False,
                num_proc=self.proc_num,
                remove_columns=self.dataset.column_names,
                desc="Running tokenization",
            )
        return self.dataset

from tevatron.driver.transformers_MoE import TapasTokenizer

class HFCorpusDataset_tapas_MoE:
    def __init__(self, tokenizer_text: PreTrainedTokenizer, tokenizer_table: PreTrainedTokenizer, data_args: DataArguments, cache_dir: str):
        data_files = data_args.encode_in_path
        if data_files:
            data_files = {data_args.dataset_split: data_files}
        script_prefix = data_args.dataset_name
        if script_prefix.endswith('-corpus'):
            script_prefix = script_prefix[:-7]
        self.preprocessor = CorpusPreProcessor_tapas_MoE
        if data_args.dataset_name == 'json_not_encode':
            data_args.dataset_name = 'json'
        self.dataset = load_dataset(data_args.dataset_name,
                                    data_args.dataset_language,
                                    data_files=data_files, cache_dir=cache_dir)[data_args.dataset_split]
        
        self.tokenizer_text = tokenizer_text
        self.tokenizer_table = tokenizer_table
        self.p_max_len = data_args.p_max_len
        self.proc_num = data_args.dataset_proc_num
        self.separator_text = getattr(self.tokenizer_text, data_args.passage_field_separator, data_args.passage_field_separator)
        self.separator_table = getattr(self.tokenizer_table, data_args.passage_field_separator, data_args.passage_field_separator)

    def process(self, shard_num=1, shard_idx=0):
        self.dataset = self.dataset.shard(shard_num, shard_idx)
        print(self.preprocessor)
        if self.preprocessor is not None:
            self.dataset = self.dataset.map(
                self.preprocessor(self.tokenizer_text, self.tokenizer_table, self.p_max_len, self.separator_text, self.separator_table),
                batched=False,
                num_proc=self.proc_num,
                remove_columns=self.dataset.column_names,
                desc="Running tokenization",
            )
        return self.dataset
