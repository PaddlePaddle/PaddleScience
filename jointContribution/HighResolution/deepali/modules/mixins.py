from typing import Dict
from typing import Optional

import paddle


class DeviceProperty(object):
    """Mixin for paddle.nn.Layer to provide 'device' property."""

    @property
    def device(self) -> str:
        """Device of first found module parameter or buffer."""
        for param in self.parameters():
            if param is not None:
                return param.place
        for buffer in self.buffers():
            if buffer is not None:
                return buffer.place
        return str("cpu").replace("cuda", "gpu")


class ReprWithCrossReferences(object):
    """Mixin of __repr__ for paddle.nn.Layer subclasses to include cross-references to reused modules."""

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
        extra_lines = []
        extra_repr = module.extra_repr()
        if extra_repr:
            extra_lines = extra_repr.split("\n")
        child_lines = []
        for key, child in module._modules.items():
            mod_str = self._repr_impl(
                prefix=prefix + key + ".", module=child, memo=memo
            )
            mod_str = _addindent(mod_str, 2)
            prev_key = memo.get(child)
            if prev_key:
                mod_str = f"{prev_key}(\n  {mod_str}\n)"
                mod_str = _addindent(mod_str, 2)
            child_lines.append(f"({key}): {mod_str}")
            memo[child] = prefix + key
        lines = extra_lines + child_lines
        main_str = module._get_name() + "("
        if lines:
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += "\n  " + "\n  ".join(lines) + "\n"
        main_str += ")"
        return main_str


def _addindent(s_: str, numSpaces: int) -> str:
    """Add indentation to multi-line string."""
    s = s_.split("\n")
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(numSpaces * " " + line) for line in s]
    s = "\n".join(s)
    s = first + "\n" + s
    return s
