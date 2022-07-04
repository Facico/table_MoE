import json
import os
import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor
import torch.distributed as dist

from transformers_MoE import AutoModel, PreTrainedModel, MoE_BertModel
from transformers_MoE.modeling_outputs import ModelOutput
from transformers_MoE.moe_layer import Bias_hard_MoE_Linear

if __name__ == '__main__':
    ans = 1