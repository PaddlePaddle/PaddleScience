r"""Mix-in classes for paddle.nn.Layer subclasses."""

from typing import Dict
from typing import Optional

import paddle


class DeviceProperty(object):
    r"""Mixin for paddle.nn.Layer to provide 'device' property."""

    @property
    def device(self) -> (paddle.CPUPlace, paddle.CUDAPlace, str):
        r"""Device of first found module parameter or buffer."""
        for param in self.parameters():
            if param is not None:
                return param.place
        for buffer in self.buffers():
            if buffer is not None:
                return buffer.place
        return paddle.CPUPlace()


class ReprWithCrossReferences(object):
    r"""Mixin of __repr__ for paddle.nn.Layer subclasses to include cross-references to reused modules."""

    def __repr__(self) -> str:
        return self._repr_impl()

    def _repr_impl(
        self,
        prefix: str = "",
        module: Optional[paddle.nn.Layer] = None,
        memo: Optional[Dict[paddle.nn.Layer, str]] = None,
    ) -> str:
        if module is None:
            module = self
        if memo is None:
            memo = {}
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = module.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split("\n")
        child_lines = []
        for key, child in module._sub_layers.items():
            mod_str = self._repr_impl(prefix=prefix + key + ".", module=child, memo=memo)
            mod_str = _addindent(mod_str, 2)
            prev_key = memo.get(child)
            if prev_key:
                mod_str = f"{prev_key}(\n  {mod_str}\n)"
                mod_str = _addindent(mod_str, 2)
            child_lines.append(f"({key}): {mod_str}")
            memo[child] = prefix + key
        lines = extra_lines + child_lines

        main_str = module.full_name() + "("
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += "\n  " + "\n  ".join(lines) + "\n"
        main_str += ")"
        return main_str


def _addindent(s_: str, numSpaces: int) -> str:
    r"""Add indentation to multi-line string."""
    # Copied from https://github.com/pytorch/pytorch/blob/992d251c39b5eb45e0b898feac46b18a0a8c8e8f/torch/nn/modules/module.py#L29-L38
    s = s_.split("\n")
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(numSpaces * " " + line) for line in s]
    s = "\n".join(s)
    s = first + "\n" + s
    return s
