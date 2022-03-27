#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Command line arguments utils
"""


import logging
import os
import random
import socket
import subprocess
from typing import Tuple
import argparse
import numpy as np
import torch
from omegaconf import DictConfig

logger = logging.getLogger()

def add_cuda_params(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--no_cuda", action="store_true", help="Whether not to use CUDA when available"
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local_rank for distributed training on gpus",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit float precision instead of 32-bit",
    )

    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
def print_args(args):
    logger.info(" **************** CONFIGURATION **************** ")
    for key, val in sorted(vars(args).items()):
        keystr = "{}".format(key) + (" " * (30 - len(key)))
        logger.info("%s -->   %s", keystr, val)
    logger.info(" **************** CONFIGURATION **************** ")
def add_tokenizer_params(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--do_lower_case",
        action="store_true",
        help="Whether to lower case the input text. True for uncased models, False for cased models.",
    )
def get_encoder_params_state(args):
    """
     Selects the param values to be saved in a checkpoint, so that a trained model faile can be used for downstream
     tasks without the need to specify these parameter again
    :return: Dict of params to memorize in a checkpoint
    """
    params_to_save = get_encoder_checkpoint_params_names()

    r = {}
    for param in params_to_save:
        r[param] = getattr(args, param)
    return r
def set_encoder_params_from_state(state, args):
    if not state:
        return
    params_to_save = get_encoder_checkpoint_params_names()

    override_params = [
        (param, state[param])
        for param in params_to_save
        if param in state and state[param]
    ]
    for param, value in override_params:
        if hasattr(args, param):
            logger.warning(
                "Overriding args parameter value from checkpoint state. Param = %s, value = %s",
                param,
                value,
            )
        setattr(args, param, value)
    return args
def add_reader_preprocessing_params(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--gold_passages_src",
        type=str,
        help="File with the original dataset passages (json format). Required for train set",
    )
    parser.add_argument(
        "--gold_passages_src_dev",
        type=str,
        help="File with the original dataset passages (json format). Required for dev set",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=16,
        help="number of parallel processes to binarize reader data",
    )
def add_training_params(parser: argparse.ArgumentParser):
    """
    Common parameters for training
    """
    add_cuda_params(parser)
    parser.add_argument(
        "--train_file", default=None, type=str, help="File pattern for the train set"
    )
    parser.add_argument("--dev_file", default=None, type=str, help="")

    parser.add_argument(
        "--batch_size", default=2, type=int, help="Amount of questions per batch"
    )
    parser.add_argument(
        "--dev_batch_size",
        type=int,
        default=4,
        help="amount of questions per batch for dev set validation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="random seed for initialization and dataset shuffling",
    )

    parser.add_argument(
        "--adam_eps", default=1e-8, type=float, help="Epsilon for Adam optimizer."
    )
    parser.add_argument(
        "--adam_betas",
        default="(0.9, 0.999)",
        type=str,
        help="Betas for Adam optimizer.",
    )

    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument("--log_batch_step", default=100, type=int, help="")
    parser.add_argument("--train_rolling_loss_step", default=100, type=int, help="")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="")
    parser.add_argument(
        "--learning_rate",
        default=1e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )

    parser.add_argument(
        "--warmup_steps", default=100, type=int, help="Linear warmup over warmup_steps."
    )
    parser.add_argument("--dropout", default=0.1, type=float, help="")

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--num_train_epochs",
        default=3.0,
        type=float,
        help="Total number of training epochs to perform.",
    )
def setup_args_gpu(args):
    """
    Setup arguments CUDA, GPU & distributed training
    """

    if args.local_rank == -1 or args.no_cuda:  # single-node multi-gpu (or cpu) mode
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        )
        args.n_gpu = torch.cuda.device_count()
    else:  # distributed mode
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device
    ws = os.environ.get("WORLD_SIZE")

    args.distributed_world_size = int(ws) if ws else 1

    logger.info(
        "Initialized host %s as d.rank %d on device=%s, n_gpu=%d, world size=%d",
        socket.gethostname(),
        args.local_rank,
        device,
        args.n_gpu,
        args.distributed_world_size,
    )
    logger.info("16-bits training: %s ", args.fp16)

def add_encoder_params(parser: argparse.ArgumentParser):
    """
    Common parameters to initialize an encoder-based model
    """
    parser.add_argument(
        "--pretrained_model_cfg",
        default=None,
        type=str,
        help="config name for model initialization",
    )
    parser.add_argument(
        "--encoder_model_type",
        default=None,
        type=str,
        help="model type. One of [hf_bert, pytext_bert, fairseq_roberta]",
    )
    parser.add_argument(
        "--pretrained_file",
        type=str,
        help="Some encoders need to be initialized from a file",
    )
    parser.add_argument(
        "--model_file",
        default=None,
        type=str,
        help="Saved bi-encoder checkpoint file to initialize the model",
    )
    parser.add_argument(
        "--projection_dim",
        default=0,
        type=int,
        help="Extra linear layer on top of standard bert/roberta encoder",
    )
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=512,
        help="Max length of the encoder input sequence",
    )

# TODO: to be merged with conf_utils.py


def set_cfg_params_from_state(state: dict, cfg: DictConfig):
    """
    Overrides some of the encoder config parameters from a give state object
    """
    if not state:
        return

    cfg.do_lower_case = state["do_lower_case"]

    if "encoder" in state:
        saved_encoder_params = state["encoder"]
        # TODO: try to understand why cfg.encoder = state["encoder"] doesn't work

        for k, v in saved_encoder_params.items():

            # TODO: tmp fix
            if k == "q_wav2vec_model_cfg":
                k = "q_encoder_model_cfg"
            if k == "q_wav2vec_cp_file":
                k = "q_encoder_cp_file"
            if k == "q_wav2vec_cp_file":
                k = "q_encoder_cp_file"

            setattr(cfg.encoder, k, v)
    else:  # 'old' checkpoints backward compatibility support
        pass
        # cfg.encoder.pretrained_model_cfg = state["pretrained_model_cfg"]
        # cfg.encoder.encoder_model_type = state["encoder_model_type"]
        # cfg.encoder.pretrained_file = state["pretrained_file"]
        # cfg.encoder.projection_dim = state["projection_dim"]
        # cfg.encoder.sequence_length = state["sequence_length"]


def get_encoder_params_state_from_cfg(cfg: DictConfig):
    """
    Selects the param values to be saved in a checkpoint, so that a trained model can be used for downstream
    tasks without the need to specify these parameter again
    :return: Dict of params to memorize in a checkpoint
    """
    return {
        "do_lower_case": cfg.do_lower_case,
        "encoder": cfg.encoder,
    }


def set_seed(args):
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def setup_cfg_gpu(cfg):
    """
    Setup params for CUDA, GPU & distributed training
    """
    logger.info("CFG's local_rank=%s", cfg.local_rank)
    ws = os.environ.get("WORLD_SIZE")
    cfg.distributed_world_size = int(ws) if ws else 1
    logger.info("Env WORLD_SIZE=%s", ws)

    if cfg.distributed_port and cfg.distributed_port > 0:
        logger.info("distributed_port is specified, trying to init distributed mode from SLURM params ...")
        init_method, local_rank, world_size, device = _infer_slurm_init(cfg)

        logger.info(
            "Inferred params from SLURM: init_method=%s | local_rank=%s | world_size=%s",
            init_method,
            local_rank,
            world_size,
        )

        cfg.local_rank = local_rank
        cfg.distributed_world_size = world_size
        cfg.n_gpu = 1

        torch.cuda.set_device(device)
        device = str(torch.device("cuda", device))

        torch.distributed.init_process_group(
            backend="nccl", init_method=init_method, world_size=world_size, rank=local_rank
        )

    elif cfg.local_rank == -1 or cfg.no_cuda:  # single-node multi-gpu (or cpu) mode
        device = str(torch.device("cuda" if torch.cuda.is_available() and not cfg.no_cuda else "cpu"))
        cfg.n_gpu = torch.cuda.device_count()
    else:  # distributed mode
        torch.cuda.set_device(cfg.local_rank)
        device = str(torch.device("cuda", cfg.local_rank))
        torch.distributed.init_process_group(backend="nccl")
        cfg.n_gpu = 1

    cfg.device = device

    logger.info(
        "Initialized host %s as d.rank %d on device=%s, n_gpu=%d, world size=%d",
        socket.gethostname(),
        cfg.local_rank,
        cfg.device,
        cfg.n_gpu,
        cfg.distributed_world_size,
    )
    logger.info("16-bits training: %s ", cfg.fp16)
    return cfg


def _infer_slurm_init(cfg) -> Tuple[str, int, int, int]:

    node_list = os.environ.get("SLURM_STEP_NODELIST")
    if node_list is None:
        node_list = os.environ.get("SLURM_JOB_NODELIST")
    logger.info("SLURM_JOB_NODELIST: %s", node_list)

    if node_list is None:
        raise RuntimeError("Can't find SLURM node_list from env parameters")

    local_rank = None
    world_size = None
    distributed_init_method = None
    device_id = None
    try:
        hostnames = subprocess.check_output(["scontrol", "show", "hostnames", node_list])
        distributed_init_method = "tcp://{host}:{port}".format(
            host=hostnames.split()[0].decode("utf-8"),
            port=cfg.distributed_port,
        )
        nnodes = int(os.environ.get("SLURM_NNODES"))
        logger.info("SLURM_NNODES: %s", nnodes)
        ntasks_per_node = os.environ.get("SLURM_NTASKS_PER_NODE")
        if ntasks_per_node is not None:
            ntasks_per_node = int(ntasks_per_node)
            logger.info("SLURM_NTASKS_PER_NODE: %s", ntasks_per_node)
        else:
            ntasks = int(os.environ.get("SLURM_NTASKS"))
            logger.info("SLURM_NTASKS: %s", ntasks)
            assert ntasks % nnodes == 0
            ntasks_per_node = int(ntasks / nnodes)

        if ntasks_per_node == 1:
            gpus_per_node = torch.cuda.device_count()
            node_id = int(os.environ.get("SLURM_NODEID"))
            local_rank = node_id * gpus_per_node
            world_size = nnodes * gpus_per_node
            logger.info("node_id: %s", node_id)
        else:
            world_size = ntasks_per_node * nnodes
            proc_id = os.environ.get("SLURM_PROCID")
            local_id = os.environ.get("SLURM_LOCALID")
            logger.info("SLURM_PROCID %s", proc_id)
            logger.info("SLURM_LOCALID %s", local_id)
            local_rank = int(proc_id)
            device_id = int(local_id)

    except subprocess.CalledProcessError as e:  # scontrol failed
        raise e
    except FileNotFoundError:  # Slurm is not installed
        pass
    return distributed_init_method, local_rank, world_size, device_id


def setup_logger(logger):
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    log_formatter = logging.Formatter("[%(thread)s] %(asctime)s [%(levelname)s] %(name)s: %(message)s")
    console = logging.StreamHandler()
    console.setFormatter(log_formatter)
    logger.addHandler(console)