import math
from typing import List
from typing import Optional

import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class LoRALinear(nn.Linear):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        merge_weights: bool = True,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        if not isinstance(r, int) or r <= 0:
            raise ValueError("Lora rank r should be a positive integer")
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights

        # Actual trainable parameters
        self.lora_A = self.create_parameter(
            shape=[in_features, r],
            dtype=self._dtype,
            is_bias=False,
            default_initializer=nn.initializer.KaimingUniform(
                negative_slope=math.sqrt(5), nonlinearity="leaky_relu"
            ),
        )
        self.lora_B = self.create_parameter(
            shape=[r, out_features],
            dtype=self._dtype,
            is_bias=False,
            default_initializer=nn.initializer.Constant(value=0.0),
        )
        self.scaling = self.lora_alpha / self.r

        # Freezing the pre-trained weight matrix
        self.weight.stop_gradient = True

    def train(self):
        super().train()
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            new_weight = self.weight - self.lora_A @ self.lora_B * self.scaling
            self.weight.set_value(new_weight)
            self.merged = False

    def eval(self):
        super().eval()
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            new_weight = self.weight + self.lora_A @ self.lora_B * self.scaling
            self.weight.set_value(new_weight)
            self.merged = True

    def forward(self, input: paddle.Tensor):
        result = F.linear(x=input, weight=self.weight, bias=self.bias, name=self.name)
        if not self.merged:
            result += (
                self.lora_dropout(input) @ self.lora_A @ self.lora_B
            ) * self.scaling
        return result

    def extra_repr(self):
        name = f", name={self.name}" if self.name else ""
        return f"in_features={self.weight.shape[0]}, out_features={self.weight.shape[1]}, rank={self.r}{name}"


def convert_linear_layer_to_lora(
    model,
    part_module_name=None,
    r=0,
    lora_alpha=1,
    lora_dropout=0,
    merge_weights=True,
):
    for i in model.named_sublayers():
        module_name = i[0]

        parent_module = model
        attribute_chain = module_name.split(".")
        for name in attribute_chain[:-1]:
            parent_module = getattr(parent_module, name)
        module = getattr(parent_module, attribute_chain[-1])

        if isinstance(module, nn.Linear):
            lora_module = LoRALinear(
                in_features=module.weight.shape[0],
                out_features=module.weight.shape[1],
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                merge_weights=merge_weights,
            )

            lora_module.weight = module.weight
            if module.bias is not None:
                lora_module.bias = module.bias
            setattr(parent_module, attribute_chain[-1], lora_module)
        else:
            continue

    return model

    #     for target_module, enable_lora in zip(target_modules, enable_lora_list):
    #         for i in model.named_sublayers():
    #             module_name = i[0]
    #             if re.fullmatch(target_module, module_name):
    #                 self._find_and_replace_module(model, module_name, lora_config, enable_lora)
    #     return model

    # def _find_and_replace_module(self, model, module_name, lora_config, enable_lora):
    #     parent_module = model
    #     attribute_chain = module_name.split(".")
    #     for name in attribute_chain[:-1]:
    #         parent_module = getattr(parent_module, name)
    #     module = getattr(parent_module, attribute_chain[-1])
    #     lora_module = None
    #     if isinstance(module, nn.Linear):
    #         lora_module = LoRALinear(
    #             in_features=module.weight.shape[0],
    #             out_features=module.weight.shape[1],
    #             r=lora_config.r,
    #             lora_alpha=lora_config.lora_alpha,
    #             lora_dropout=lora_config.lora_dropout,
    #             merge_weights=lora_config.merge_weights,
    #         )

    #     lora_module.weight = module.weight
    #     if module.bias is not None:
    #         lora_module.bias = module.bias
    #     setattr(parent_module, attribute_chain[-1], lora_module)

    #     for target_module, enable_lora in zip(target_modules, enable_lora_list):
    #         for i in model.named_sublayers():
    #             module_name = i[0]
    #             if re.fullmatch(target_module, module_name):
    #                 self._find_and_replace_module(model, module_name, lora_config, enable_lora)
    #     return model
