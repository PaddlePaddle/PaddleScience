import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from typing import List, Optional
from paddle import Tensor

from paddle_geometric.nn.nlp.llm import BOS, LLM, MAX_NEW_TOKENS
from paddle_geometric.utils import scatter


class GRetriever(nn.Layer):
    r"""The G-Retriever model from the `"G-Retriever: Retrieval-Augmented
    Generation for Textual Graph Understanding and Question Answering"
    <https://arxiv.org/abs/2402.07630>`_ paper.

    Args:
        llm (LLM): The LLM to use.
        gnn (paddle.nn.Layer): The GNN to use.
        use_lora (bool, optional): If set to :obj:`True`, will use LORA from
            :obj:`peft` for training the LLM, see
            `here <https://huggingface.co/docs/peft/en/index>`_ for details.
            (default: :obj:`False`)
        mlp_out_channels (int, optional): The size of each graph embedding
            after projection. (default: :obj:`4096`)
    """

    def __init__(
        self,
        llm: LLM,
        gnn: nn.Layer,
        use_lora: bool = False,
        mlp_out_channels: int = 4096,
    ) -> None:
        super().__init__()

        self.llm = llm
        self.gnn = gnn.to(self.llm.device)

        self.word_embedding = self.llm.word_embedding
        self.llm_generator = self.llm.llm
        if use_lora:
            from peft import (
                LoraConfig,
                get_peft_model,
                prepare_model_for_kbit_training,
            )
            self.llm_generator = prepare_model_for_kbit_training(
                self.llm_generator)
            lora_r: int = 8
            lora_alpha: int = 16
            lora_dropout: float = 0.05
            lora_target_modules = ['q_proj', 'v_proj']
            config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
                bias='none',
                task_type='CAUSAL_LM',
            )
            self.llm_generator = get_peft_model(self.llm_generator, config)

        mlp_hidden_channels = self.gnn.out_channels
        self.projector = nn.Sequential(
            nn.Linear(mlp_hidden_channels, mlp_hidden_channels),
            nn.Sigmoid(),
            nn.Linear(mlp_hidden_channels, mlp_out_channels),
        ).to(self.llm.device)

    def encode(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor,
        edge_attr: Optional[Tensor],
    ) -> Tensor:
        x = x.to(self.llm.device)
        edge_index = edge_index.to(self.llm.device)
        if edge_attr is not None:
            edge_attr = edge_attr.to(self.llm.device)
        batch = batch.to(self.llm.device)

        out = self.gnn(x, edge_index, edge_attr=edge_attr)
        return scatter(out, batch, dim=0, reduce='mean')

    def forward(
        self,
        question: List[str],
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor,
        label: List[str],
        edge_attr: Optional[Tensor] = None,
        additional_text_context: Optional[List[str]] = None,
    ):
        x = self.encode(x, edge_index, batch, edge_attr)
        x = self.projector(x)
        xs = paddle.split(x, 1, axis=0)

        # Handle questions without node features:
        batch_unique = paddle.unique(batch)
        batch_size = len(question)
        if len(batch_unique) < batch_size:
            xs = [
                xs[i] if i in batch_unique else None for i in range(batch_size)
            ]

        inputs_embeds, attention_mask, label_input_ids = self.llm._get_embeds(
            question, additional_text_context, xs, label)

        with self.llm.autocast_context:
            outputs = self.llm_generator(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=label_input_ids,
            )

        return outputs.loss

    @paddle.no_grad()
    def inference(
        self,
        question: List[str],
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor,
        edge_attr: Optional[Tensor] = None,
        additional_text_context: Optional[List[str]] = None,
        max_out_tokens: Optional[int] = MAX_NEW_TOKENS,
    ):
        x = self.encode(x, edge_index, batch, edge_attr)
        x = self.projector(x)
        xs = paddle.split(x, 1, axis=0)

        # Handle questions without node features:
        batch_unique = paddle.unique(batch)
        batch_size = len(question)
        if len(batch_unique) < batch_size:
            xs = [
                xs[i] if i in batch_unique else None for i in range(batch_size)
            ]

        inputs_embeds, attention_mask, _ = self.llm._get_embeds(
            question, additional_text_context, xs)

        bos_token = self.llm.tokenizer(
            BOS,
            add_special_tokens=False,
        ).input_ids[0]

        with self.llm.autocast_context:
            outputs = self.llm_generator.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=max_out_tokens,
                attention_mask=attention_mask,
                bos_token_id=bos_token,
                use_cache=True  # Important to set!
            )

        return self.llm.tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True,
        )

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(\n'
                f'  llm={self.llm},\n'
                f'  gnn={self.gnn},\n'
                f')')
