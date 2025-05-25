import os.path as osp
from pathlib import Path
from typing import Any, Dict, Optional, Union

import paddle

from paddle_geometric.io import fs

try:
    from paddle_hub import ModelHubMixin, hf_hub_download
except ImportError:
    ModelHubMixin = object
    hf_hub_download = None

CONFIG_NAME = 'config.json'
MODEL_HUB_ORGANIZATION = "paddle_geometric"
MODEL_WEIGHTS_NAME = 'model.pdparams'
TAGS = ['graph-machine-learning']


class PyGModelHubMixin(ModelHubMixin):
    r"""A mixin for saving and loading models to the
    `Paddle Hub <https://www.paddlepaddle.org.cn/hub>`_.

    Methods to interact with Paddle Hub for model saving and loading.
    """

    def __init__(self, model_name: str, dataset_name: str, model_kwargs: Dict):
        ModelHubMixin.__init__(self)

        # Paddle Hub API requires the model config to be serialized in a compatible format
        self.model_config = {
            k: v
            for k, v in model_kwargs.items() if isinstance(v, (str, int, float))
        }
        self.model_name = model_name
        self.dataset_name = dataset_name

    def construct_model_card(self, model_name: str, dataset_name: str) -> Any:
        from paddle_hub import ModelCard, ModelCardData
        card_data = ModelCardData(
            language='en',
            license='mit',
            library_name=MODEL_HUB_ORGANIZATION,
            tags=TAGS,
            datasets=dataset_name,
            model_name=model_name,
        )
        card = ModelCard.from_template(card_data)
        return card

    def _save_pretrained(self, save_directory: Union[Path, str]):
        path = osp.join(save_directory, MODEL_WEIGHTS_NAME)
        model_to_save = self.module if hasattr(self, 'module') else self
        paddle.save(model_to_save.state_dict(), path)

    def save_pretrained(self, save_directory: Union[str, Path],
                        push_to_hub: bool = False,
                        repo_id: Optional[str] = None, **kwargs):
        r"""Save a trained model to a local directory or to Paddle Hub."""

        config = self.model_config
        kwargs.pop('config', None)  # remove config to prevent duplication

        super().save_pretrained(
            save_directory=save_directory,
            config=config,
            push_to_hub=push_to_hub,
            repo_id=repo_id,
            **kwargs,
        )
        model_card = self.construct_model_card(self.model_name, self.dataset_name)
        if push_to_hub:
            model_card.push_to_hub(repo_id)

    @classmethod
    def _from_pretrained(
        cls,
        model_id,
        revision,
        cache_dir,
        force_download,
        proxies,
        resume_download,
        local_files_only,
        token,
        dataset_name='',
        model_name='',
        map_location='cpu',
        strict=False,
        **model_kwargs,
    ):
        map_location = paddle.set_device(map_location)

        if osp.isdir(model_id):
            model_file = osp.join(model_id, MODEL_WEIGHTS_NAME)
        else:
            model_file = hf_hub_download(
                repo_id=model_id,
                filename=MODEL_WEIGHTS_NAME,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                token=token,
                local_files_only=local_files_only,
            )

        config = model_kwargs.pop('config', None)
        if config is not None:
            model_kwargs = {**model_kwargs, **config}

        model = cls(dataset_name, model_name, model_kwargs)

        state_dict = fs.paddle_load(model_file, map_location=map_location)
        model.set_state_dict(state_dict)

        model.eval()

        return model

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        force_download: bool = False,
        resume_download: bool = False,
        proxies: Optional[Dict] = None,
        token: Optional[Union[str, bool]] = None,
        cache_dir: Optional[str] = None,
        local_files_only: bool = False,
        **model_kwargs,
    ) -> Any:
        r"""Downloads and instantiates a model from Paddle Hub."""

        return super().from_pretrained(
            pretrained_model_name_or_path,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            use_auth_token=token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            **model_kwargs,
        )
