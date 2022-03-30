import torch
import torch.nn as nn
from torch import Tensor, Size
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from typing import Union, List, Tuple

# MoE_type_tensor: 0 --> text  1 --> table
class Bias_hard_MoE_Linear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Bias_hard_MoE_Linear, self).__init__(in_features, out_features, bias)
        if self.bias is not None:
            
            self.bias_table = Parameter(torch.empty(out_features, **factory_kwargs))
            self.bias_text = Parameter(torch.empty(out_features, **factory_kwargs))

            b = self.bias.clone().detach()
            self.bias_table.requires_grad = False
            self.bias_table.copy_(b.contiguous())
            self.bias_table.requires_grad = True

            self.bias_text.requires_grad = False
            self.bias_text.copy_(b.contiguous())
            self.bias_text.requires_grad = True
    
    def forward(self, input: Tensor, MoE_type_tensor=None) -> Tensor:
        if MoE_type_tensor is None:
            return F.linear(input, self.weight, self.bias_text)
        
        out_text =  F.linear(input, self.weight, self.bias_text)
        out_table =  F.linear(input, self.weight, self.bias_table)
        size_index = [out_text.shape[0]] + [1] * (len(out_text.shape) - 1)

        one_index = MoE_type_tensor.view(*size_index)

        output = out_text * (1 - one_index) + out_table * one_index
        """output = torch.mm(input, self.weight.t())
        batch_size = input.shape[0]

        table_bias_w = torch.diag_embed(MoE_type_tensor)
        text_bias_w = torch.diag_embed(1 - MoE_type_tensor)

        table_bias = torch.mm(table_bias_w, self.bias_table.expand(batch_size, -1))
        text_bias = torch.mm(text_bias_w, self.bias_table.expand(batch_size, -1))

        output = output + table_bias + text_bias"""
        # [batch size, out_features]
        """if MoE_type == 'text':

            return F.linear(input, self.weight, self.bias_text)
        elif MoE_type == 'table':
            return F.linear(input, self.weight, self.bias_table)
        else:
            raise ValueError(f"unknown type {MoE_type}")"""
        
        return output

def MoE_prune_linear_layer(layer: nn.Linear, index: torch.LongTensor, dim: int = 0) -> nn.Linear:
    """
    Prune a linear layer to keep only entries in index.

    Used to remove heads.

    Args:
        layer (`torch.nn.Linear`): The layer to prune.
        index (`torch.LongTensor`): The indices to keep in the layer.
        dim (`int`, *optional*, defaults to 0): The dimension on which to keep the indices.

    Returns:
        `torch.nn.Linear`: The pruned layer as a new layer with `requires_grad=True`.
    """
    index = index.to(layer.weight.device)
    W = layer.weight.index_select(dim, index).clone().detach()
    if layer.bias is not None:
        if dim == 1:
            b = layer.bias.clone().detach()
        else:
            b = layer.bias[index].clone().detach()
    new_size = list(layer.weight.size())
    new_size[dim] = len(index)
    new_layer = Bias_hard_MoE_Linear(new_size[1], new_size[0], bias=layer.bias is not None).to(layer.weight.device)
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(W.contiguous())
    new_layer.weight.requires_grad = True
    if layer.bias is not None:
        new_layer.bias_table.requires_grad = False
        new_layer.bias_table.copy_(b.contiguous())
        new_layer.bias_table.requires_grad = True

        new_layer.bias_text.requires_grad = False
        new_layer.bias_text.copy_(b.contiguous())
        new_layer.bias_text.requires_grad = True
    return new_layer

_shape_t = Union[int, List[int], Size]

class LN_hard_MoE(nn.LayerNorm):
    def __init__(self, normalized_shape: _shape_t, eps: float = 1e-5, elementwise_affine: bool = True,
                 device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(LN_hard_MoE, self).__init__(normalized_shape, eps, elementwise_affine)
        if self.bias is not None:
            if self.elementwise_affine:
                self.bias_table = Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
                self.bias_text = Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
            else:
                self.register_parameter('bias_table', None)
                self.register_parameter('bias_text', None)
            
            b = self.bias.clone().detach()
            self.bias_table.requires_grad = False
            self.bias_table.copy_(b.contiguous())
            self.bias_table.requires_grad = True
            self.bias_text.requires_grad = False
            self.bias_text.copy_(b.contiguous())
            self.bias_text.requires_grad = True
            

    def forward(self, input: Tensor, MoE_type_tensor=None) -> Tensor:
        if MoE_type_tensor is None:
            return F.layer_norm(
            input, self.normalized_shape, self.weight, self.bias_text, self.eps)
        out_text = F.layer_norm(
            input, self.normalized_shape, self.weight, self.bias_text, self.eps)
        out_table = F.layer_norm(
            input, self.normalized_shape, self.weight, self.bias_table, self.eps)
        size_index = [out_text.shape[0]] + [1] * (len(out_text.shape) - 1)

        one_index = MoE_type_tensor.view(*size_index)

        output = out_text * (1 - one_index) + out_table * one_index

        return output
        """if MoE_type == 'text':
            return F.layer_norm(
            input, self.normalized_shape, self.weight, self.bias_text, self.eps)
        elif MoE_type == 'table':
            return F.layer_norm(
            input, self.normalized_shape, self.weight, self.bias_table, self.eps)
        else:
            raise ValueError(f"unknown type {MoE_type}")"""
if __name__ == '__main__':
    m = Bias_hard_MoE_Linear(10, 2)
    print(m.weight.shape)
    testx = torch.randn(5,10)
    MoE_type_list = torch.tensor([0.0,1.0,1.0,0.0,0.0])
    out = m(testx, MoE_type_tensor=MoE_type_list)

    """m = LN_hard_MoE(10)
    testx = torch.randn(5,10)
    MoE_type_list = torch.tensor([0.0,1.0,1.0,0.0,0.0])
    out = m(testx, MoE_type_tensor=MoE_type_list)"""