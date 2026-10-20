"""Dictionary interface for PaddleScience supervised training and evaluation."""

import paddle

from ppsci.arch.aurora.batch import Batch
from ppsci.arch.aurora.batch import Metadata
from ppsci.arch.aurora.model import aurora as models
from ppsci.arch.base import Arch


class AuroraArch(Arch):
    """Adapt an Aurora model to PaddleScience's dictionary input/output API.

    Dynamic keys are ``surf_<variable>`` and ``atmos_<variable>``. Supply static
    fields once in the constructor, or as ``static_<variable>`` inputs. Metadata
    can be provided once or replaced with ``inputs['metadata']`` on each call.
    Surface and atmospheric predictions retain their history/level dimensions.
    """

    def __init__(
        self,
        model_type="AuroraSmallPretrained",
        model_kwargs=None,
        metadata: Metadata = None,
        static_vars=None,
        output_keys=None,
    ):
        super().__init__()
        if model_type not in models.__all__:
            raise ValueError(f"Unknown Aurora model type: {model_type}")
        self.model = getattr(models, model_type)(**(model_kwargs or {}))
        self.metadata = metadata
        self._static_names = tuple((static_vars or {}).keys())
        for name, value in (static_vars or {}).items():
            self.register_buffer(
                f"_static_{name}", paddle.to_tensor(value), persistable=False
            )
        self.input_keys = tuple(
            f"surf_{v}"
            for v in self.model.surf_vars
            if v not in self.model.output_only_surf_vars
        )
        self.input_keys += tuple(f"atmos_{v}" for v in self.model.atmos_vars)
        self.output_keys = (
            tuple(output_keys)
            if output_keys is not None
            else (
                tuple(f"surf_{v}" for v in self.model.surf_vars)
                + tuple(f"atmos_{v}" for v in self.model.atmos_vars)
            )
        )

    def forward(self, inputs):
        if self._input_transform is not None:
            inputs = self._input_transform(inputs)
        metadata = inputs.get("metadata", self.metadata)
        if metadata is None:
            raise ValueError(
                "Aurora requires Metadata with coordinates, valid times, and pressure levels"
            )
        static_vars = {
            name: getattr(self, f"_static_{name}") for name in self._static_names
        }
        static_vars.update(
            {
                name[7:]: value
                for name, value in inputs.items()
                if name.startswith("static_")
            }
        )
        batch = Batch(
            surf_vars={
                name[5:]: value
                for name, value in inputs.items()
                if name.startswith("surf_")
            },
            atmos_vars={
                name[6:]: value
                for name, value in inputs.items()
                if name.startswith("atmos_")
            },
            static_vars=static_vars,
            metadata=metadata,
        )
        prediction = self.model(batch, lead_times=inputs.get("lead_times"))
        outputs = {f"surf_{k}": value for k, value in prediction.surf_vars.items()}
        outputs.update(
            {f"atmos_{k}": value for k, value in prediction.atmos_vars.items()}
        )
        outputs = {key: outputs[key] for key in self.output_keys}
        if self._output_transform is not None:
            outputs = self._output_transform(inputs, outputs)
        return outputs
