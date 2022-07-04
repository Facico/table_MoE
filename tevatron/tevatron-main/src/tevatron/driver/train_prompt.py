import logging
import os
import sys

from tevatron.driver.transformers_MoE import AutoConfig, AutoTokenizer
from tevatron.driver.transformers_MoE import (
    HfArgumentParser,
    set_seed,
)

from tevatron.arguments_MoE import ModelArguments, DataArguments, \
    DenseTrainingArguments as TrainingArguments
from tevatron.data_MoE import TrainDataset_MoE, QPCollator_MoE
from tevatron.modeling_prompt_MoE import DenseModel_MoE, SoftEmbedding
from tevatron.trainer_MoE import DenseTrainer_MoE as Trainer_MoE, GCTrainer_MoE
from tevatron.datasets import HFTrainDataset_MoE

logger = logging.getLogger(__name__)

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        training_args: TrainingArguments

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("MODEL parameters %s", model_args)

    set_seed(training_args.seed)

    num_labels = 1
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False,
    )
    model = DenseModel_MoE.build(
        model_args,
        data_args,
        training_args,
        config=config,
        cache_dir=model_args.cache_dir,
    ) # add soft prompt to model_args

    #copy postion_embedding to MoE type
    """model_dict_copy = {}
    for k, v in model.state_dict().items():
        #print(k)
        if 'position_embedding' in k:
            position_only = ".".join(k.split('.')[:-1])
            if position_only.endswith('_prompt_text'):
                position_only = position_only[:-12] + '.weight'
            elif position_only.endswith('_prompt_table'):
                position_only = position_only[:-13] + '.weight'
            else:
                position_only = position_only + '.weight'
            model_dict_copy[k] = model.state_dict()[position_only]
        else:
            model_dict_copy[k] =  model.state_dict()[k]

    model.load_state_dict(model_dict_copy)"""

    # add position embedding prompt
    trained_parameters = []
    for param in model.parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        #print(name, 'bias' in name)
        if 'learned_embedding_text' in name or 'learned_embedding_table' in name:
            print(name)
            param.requires_grad = True
            trained_parameters.append(param)

    train_dataset = HFTrainDataset_MoE(tokenizer=tokenizer, data_args=data_args,
                                   cache_dir=data_args.data_cache_dir or model_args.cache_dir)
    train_dataset = TrainDataset_MoE(data_args, train_dataset.process(), tokenizer)

    trainer_cls = GCTrainer_MoE if training_args.grad_cache else Trainer_MoE
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=QPCollator_MoE(
            tokenizer,
            max_p_len=data_args.p_max_len,
            max_q_len=data_args.q_max_len
        ),
    )
    train_dataset.trainer = trainer

    trainer.train()  # TODO: resume training
    trainer.save_model()
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
