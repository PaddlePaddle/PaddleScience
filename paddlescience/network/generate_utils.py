# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
""" 
This code is refer from:
https://github.com/zabaras/transformer-physx/blob/main/trphysx/transformer/generate_utils.py
"""

from typing import Tuple, Dict

import paddle
from paddle import Tensor

Tensor = paddle.Tensor
LongTensor = paddle.int64


class GenerationMixin:
    """Class containing generative functions for transformers
    """

    def prepare_inputs_for_generation(self,
                                      inputs_embeds: Tensor,
                                      position_ids: Tensor=None,
                                      prop_embeds: Tensor=None,
                                      **kwargs) -> Dict[str, Tensor]:
        """Prepares input features for prediction

        Args:
            inputs_features (Dict[str, Tensor]): Input feature tensors
            that are being generated.

        Returns:
            Dict[str, Tensor]: Dictionary of model inputs
        """
        inputs_features = {
            "inputs_embeds": inputs_embeds,
            "position_ids": position_ids,
            "prop_embeds": prop_embeds
        }
        inputs = {}

        for k, v in inputs_features.items():
            if isinstance(v, Tensor):
                # Make sure all embeddings are of equal and proper length
                inputs[k] = v[:, -self.n_ctx:]

        if "past" in kwargs.keys():
            for k, v in inputs.items():
                if isinstance(v, Tensor):
                    inputs[k] = v[:, -1].unsqueeze(1)

        return { ** inputs, ** kwargs}

    def _use_cache(self, outputs, use_cache):
        """During generation, decide whether to pass the `past` variable to the next forward pass."""
        if len(outputs) <= 1 or use_cache is False:
            return False
        if hasattr(self, "mem_len") and self.mem_len == 0:
            return False
        return True

    @paddle.no_grad()
    def generate(self,
                 inputs_embeds: Tensor,
                 position_ids: Tensor=None,
                 prop_embeds: Tensor=None,
                 max_length: int=None,
                 attention_mask: LongTensor=None,
                 use_cache: bool=False,
                 **model_specific_kwargs) -> Tuple[Tensor]:
        """Generated a predicted sequence of features

        Args:
            inputs_embeds (Tensor): [batch, seq, n_embed] Input feature tensor
            position_ids (Tensor, optional): [seq, n_embed] Position tensor. Defaults to None.
            prop_embeds (Tensor, optional): [batch, seq, n_embed] Property tensor. Defaults to None.
            max_length (int, optional): Length of time series to predict. Defaults to None.
            attention_mask (LongTensor, optional): Manual attention mask. Defaults to None.
            use_cache (bool, optional): Cache past transformer states for faster generation. Defaults to False.

        Returns:
            Tuple[Tensor]: [batch, max_length, n_embed] Predicted feature tensor, additional optional transformer outputs.
        """
        max_length = max_length if max_length is not None else 0
        use_cache = use_cache if use_cache is not None else True

        assert isinstance(
            max_length, int
        ) and max_length > 0, "`max_length` should be a strictly positive integer."
        assert isinstance(use_cache, bool), "`use_cache` should be a boolean."

        # create attention mask if necessary
        # if attention_mask is None:
        #     attention_mask = torch.ones(inputs_embeds.shape).to(inputs_embeds.device)

        output = self._generate_time_series(
            inputs_embeds,
            position_ids,
            prop_embeds,
            max_length=max_length,
            attention_mask=attention_mask,
            use_cache=use_cache,
            **model_specific_kwargs, )

        return output

    def _generate_time_series(self,
                              inputs_embeds: Tensor,
                              position_ids: Tensor,
                              prop_embeds: Tensor,
                              max_length: int,
                              use_cache: bool=None,
                              **model_specific_kwargs) -> Tuple[Tensor]:
        """Function that calls model forward to predict 

        Args:
            inputs_embeds (Tensor): [batch, seq, n_embed] Input feature tensor
            position_ids (Tensor, optional): [seq, n_embed] Position tensor. Defaults to None.
            prop_embeds (Tensor, optional): [batch, seq, n_embed] Property tensor. Defaults to None.
            max_length (int, optional): Length of time series to predict. Defaults to None.
            attention_mask (LongTensor, optional): Manual attention mask. Defaults to None.
            use_cache (bool, optional): [description]. Defaults to None.

        Returns:
            Tuple[Tensor]: [batch, max_length, n_embed] Predicted feature tensor, additional optional transformer outputs.
        """
        past = None

        cur_len = inputs_embeds.shape[1]
        assert (
            cur_len < max_length
        ), f"The input context is {cur_len}, but `max_length` is only {max_length}. Please make sure that `max_length` larger than the input"

        while cur_len < max_length:
            # Prepare inputs for transformer
            model_inputs = self.prepare_inputs_for_generation(
                inputs_embeds,
                position_ids,
                prop_embeds,
                use_cache=use_cache,
                past=past,
                **model_specific_kwargs, )

            outputs = self.forward(**model_inputs)

            next_output = outputs[0][:, -1:]

            if self._use_cache(outputs, use_cache):
                past = [
                    output[:, :, :, -(self.n_ctx - 1):]
                    for output in outputs[1]
                ]

            # add past output embedding and increase length by one
            inputs_embeds = paddle.concat([inputs_embeds, next_output], axis=1)
            cur_len = cur_len + 1

        return (inputs_embeds, ) + outputs[1:]

    @staticmethod
    def _reorder_cache(past: Tuple, beam_idx: Tensor) -> Tuple[Tensor]:
        return tuple(
            layer_past.index_select(1, beam_idx) for layer_past in past)
