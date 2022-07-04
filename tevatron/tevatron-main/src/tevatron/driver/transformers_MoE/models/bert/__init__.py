# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import TYPE_CHECKING

from ...utils import _LazyModule, is_flax_available, is_tf_available, is_tokenizers_available, is_torch_available


_import_structure = {
    "configuration_bert": ["BERT_PRETRAINED_CONFIG_ARCHIVE_MAP", "BertConfig", "BertOnnxConfig"],
    "tokenization_bert": ["BasicTokenizer", "BertTokenizer", "WordpieceTokenizer"],
}

if is_tokenizers_available():
    _import_structure["tokenization_bert_fast"] = ["BertTokenizerFast"]

if is_torch_available():
    _import_structure["modeling_bert"] = [
        "BERT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "BertForMaskedLM",
        "BertForMultipleChoice",
        "BertForNextSentencePrediction",
        "BertForPreTraining",
        "BertForQuestionAnswering",
        "BertForSequenceClassification",
        "BertForTokenClassification",
        "BertLayer",
        "BertLMHeadModel",
        "BertModel",
        "BertPreTrainedModel",
        "load_tf_weights_in_bert",
    ]

# add MoE
if is_torch_available():
    _import_structure["modeling_bert_MoE"] = [
        "MoE_BERT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "MoE_BertForMaskedLM",
        "MoE_BertForMultipleChoice",
        "MoE_BertForNextSentencePrediction",
        "MoE_BertForPreTraining",
        "MoE_BertForQuestionAnswering",
        "MoE_BertForSequenceClassification",
        "MoE_BertForTokenClassification",
        "MoE_BertLayer",
        "MoE_BertLMHeadModel",
        "MoE_BertModel",
        "MoE_BertPreTrainedModel",
        "MoE_load_tf_weights_in_bert",
    ]

    _import_structure["modeling_bert_position_MoE"] = [
        "MoE_position_BERT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "MoE_position_BertForMaskedLM",
        "MoE_position_BertForMultipleChoice",
        "MoE_position_BertForNextSentencePrediction",
        "MoE_position_BertForPreTraining",
        "MoE_position_BertForQuestionAnswering",
        "MoE_position_BertForSequenceClassification",
        "MoE_position_BertForTokenClassification",
        "MoE_position_BertLayer",
        "MoE_position_BertLMHeadModel",
        "MoE_position_BertModel",
        "MoE_position_BertPreTrainedModel",
        "MoE_position_load_tf_weights_in_bert",
    ]

    _import_structure["modeling_bert_prompt_MoE"] = [
        "MoE_prompt_BERT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "MoE_prompt_BertForMaskedLM",
        "MoE_prompt_BertForMultipleChoice",
        "MoE_prompt_BertForNextSentencePrediction",
        "MoE_prompt_BertForPreTraining",
        "MoE_prompt_BertForQuestionAnswering",
        "MoE_prompt_BertForSequenceClassification",
        "MoE_prompt_BertForTokenClassification",
        "MoE_prompt_BertLayer",
        "MoE_prompt_BertLMHeadModel",
        "MoE_prompt_BertModel",
        "MoE_prompt_BertPreTrainedModel",
        "MoE_prompt_load_tf_weights_in_bert",
    ]
    

if is_tf_available():
    _import_structure["modeling_tf_bert"] = [
        "TF_BERT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFBertEmbeddings",
        "TFBertForMaskedLM",
        "TFBertForMultipleChoice",
        "TFBertForNextSentencePrediction",
        "TFBertForPreTraining",
        "TFBertForQuestionAnswering",
        "TFBertForSequenceClassification",
        "TFBertForTokenClassification",
        "TFBertLMHeadModel",
        "TFBertMainLayer",
        "TFBertModel",
        "TFBertPreTrainedModel",
    ]

if is_flax_available():
    _import_structure["modeling_flax_bert"] = [
        "FlaxBertForMaskedLM",
        "FlaxBertForMultipleChoice",
        "FlaxBertForNextSentencePrediction",
        "FlaxBertForPreTraining",
        "FlaxBertForQuestionAnswering",
        "FlaxBertForSequenceClassification",
        "FlaxBertForTokenClassification",
        "FlaxBertModel",
        "FlaxBertPreTrainedModel",
    ]

if TYPE_CHECKING:
    from .configuration_bert import BERT_PRETRAINED_CONFIG_ARCHIVE_MAP, BertConfig, BertOnnxConfig
    from .tokenization_bert import BasicTokenizer, BertTokenizer, WordpieceTokenizer

    if is_tokenizers_available():
        from .tokenization_bert_fast import BertTokenizerFast

    if is_torch_available():
        from .modeling_bert import (
            BERT_PRETRAINED_MODEL_ARCHIVE_LIST,
            BertForMaskedLM,
            BertForMultipleChoice,
            BertForNextSentencePrediction,
            BertForPreTraining,
            BertForQuestionAnswering,
            BertForSequenceClassification,
            BertForTokenClassification,
            BertLayer,
            BertLMHeadModel,
            BertModel,
            BertPreTrainedModel,
            load_tf_weights_in_bert,
        )

    if is_tf_available():
        from .modeling_tf_bert import (
            TF_BERT_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFBertEmbeddings,
            TFBertForMaskedLM,
            TFBertForMultipleChoice,
            TFBertForNextSentencePrediction,
            TFBertForPreTraining,
            TFBertForQuestionAnswering,
            TFBertForSequenceClassification,
            TFBertForTokenClassification,
            TFBertLMHeadModel,
            TFBertMainLayer,
            TFBertModel,
            TFBertPreTrainedModel,
        )

    if is_flax_available():
        from .modeling_flax_bert import (
            FlaxBertForMaskedLM,
            FlaxBertForMultipleChoice,
            FlaxBertForNextSentencePrediction,
            FlaxBertForPreTraining,
            FlaxBertForQuestionAnswering,
            FlaxBertForSequenceClassification,
            FlaxBertForTokenClassification,
            FlaxBertModel,
            FlaxBertPreTrainedModel,
        )

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
