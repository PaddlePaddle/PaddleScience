# Installation

## Installation via PyPI

This Python package can be installed with [pip]:

```
pip install hf-deepali
```

The latest development version can be installed directly from the GitHub repository, i.e.,

```
pip install git+https://github.com/BioMedIA/deepali.git
```

Alternatively, it can be installed from a previously cloned local Git repository using

```
git clone https://github.com/BioMedIA/deepali.git && pip install ./deepali
```

This will install missing dependencies in the current Python environment from [PyPI].
To use [conda] for installing the required dependencies (recommended), create a conda environment
with pre-installed dependencies **before** running `pip install`. For further information on how
to create and manage project dependencies using conda, see [conda/README.md](https://github.com/BioMedIA/deepali/tree/main/conda/README.md).

Additional optional dependencies of the {mod}`deepali.utils` library can be installed with the command:

```
pip install hf-deepali[utils]
# or pip install git+https://github.com/BioMedIA/deepali.git#egg=deepali[utils]
```


[conda]: https://docs.conda.io/en/latest/
[pip]: https://pip.pypa.io/en/stable/
[PyPI]: https://pypi.org/
[Miniconda]: https://docs.conda.io/en/latest/miniconda.html
