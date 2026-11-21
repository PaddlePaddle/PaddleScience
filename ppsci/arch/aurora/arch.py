"""Dictionary interface for PaddleScience supervised training and evaluation."""

from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple

import paddle

from ppsci.arch import base
from ppsci.arch.aurora.batch import Batch
from ppsci.arch.aurora.batch import Metadata
from ppsci.arch.aurora.model import aurora as models


class AuroraArch(base.Arch):
    """Adapt an Aurora model to PaddleScience's dictionary input/output API.

    Dynamic keys are ``surf_<variable>`` and ``atmos_<variable>``. Supply static
    fields once in the constructor, or as ``static_<variable>`` inputs. Metadata
    can be provided once or replaced with ``inputs['metadata']`` on each call.
    Surface and atmospheric predictions retain their history/level dimensions.

    Args:
        model_type (str): Aurora variant to construct.
        model_kwargs (Optional[Dict[str, Any]]): Variant constructor arguments.
        metadata (Optional[Metadata]): Coordinates, pressure levels and UTC times.
        static_vars (Optional[Dict[str, paddle.Tensor]]): Static fields shared by batches.
        output_keys (Optional[Tuple[str, ...]]): Output fields; defaults to all variables.
    """

    def __init__(
        self,
        model_type: str = "AuroraSmallPretrained",
        model_kwargs: Optional[Dict[str, Any]] = None,
        metadata: Optional[Metadata] = None,
        static_vars: Optional[Dict[str, paddle.Tensor]] = None,
        output_keys: Optional[Tuple[str, ...]] = None,
    ):
        super().__init__()
        if model_type not in models.__all__:
            raise ValueError(f"Unknown Aurora model type: {model_type}")
        self.model = getattr(models, model_type)(**(model_kwargs or {}))
        self.metadata = metadata
        self._static_names = tuple((static_vars or {}).keys())
        unknown_static = set(self._static_names).difference(self.model.static_vars)
        if unknown_static:
            raise ValueError(f"Unknown static variables: {sorted(unknown_static)}")
        for name, value in (static_vars or {}).items():
            tensor = (
                value.detach().clone()
                if isinstance(value, paddle.Tensor)
                else paddle.to_tensor(value)
            )
            self.register_buffer(f"_static_{name}", tensor, persistable=False)
        self.input_keys = tuple(
            f"surf_{v}"
            for v in self.model.surf_vars
            if v not in self.model.output_only_surf_vars
        )
        self.input_keys += tuple(f"atmos_{v}" for v in self.model.atmos_vars)
        available_outputs = tuple(f"surf_{v}" for v in self.model.surf_vars) + tuple(
            f"atmos_{v}" for v in self.model.atmos_vars
        )
        self.output_keys = (
            tuple(output_keys) if output_keys is not None else available_outputs
        )
        if not self.output_keys or len(set(self.output_keys)) != len(self.output_keys):
            raise ValueError("output_keys must be nonempty and unique")
        unknown_outputs = set(self.output_keys).difference(available_outputs)
        if unknown_outputs:
            raise ValueError(f"Unknown output fields: {sorted(unknown_outputs)}")

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, paddle.Tensor]:
        if self._input_transform is not None:
            inputs = self._input_transform(inputs)
        missing = set(self.input_keys).difference(inputs)
        if missing:
            raise ValueError(f"Missing dynamic fields: {sorted(missing)}")
        dynamic_fields = {key for key in inputs if key.startswith(("surf_", "atmos_"))}
        unknown = dynamic_fields.difference(self.input_keys)
        if unknown:
            raise ValueError(f"Unknown dynamic fields: {sorted(unknown)}")
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
        missing_static = set(self.model.static_vars).difference(static_vars)
        if missing_static:
            raise ValueError(f"Missing static variables: {sorted(missing_static)}")
        unknown_static = set(static_vars).difference(self.model.static_vars)
        if unknown_static:
            raise ValueError(f"Unknown static variables: {sorted(unknown_static)}")
        batch = Batch(
            surf_vars={
                name: inputs[f"surf_{name}"]
                for name in self.model.surf_vars
                if name not in self.model.output_only_surf_vars
            },
            atmos_vars={
                name: inputs[f"atmos_{name}"] for name in self.model.atmos_vars
            },
            static_vars={name: static_vars[name] for name in self.model.static_vars},
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
