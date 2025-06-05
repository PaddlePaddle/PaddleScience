# MatterSim for Paddle

> [!IMPORTANT]
> This repo experimentally integrates [Paddle backend](https://www.paddlepaddle.org.cn/en/install/quick?docurl=/documentation/docs/en/develop/install/pip/linux-pip_en.html) to MatterSim.
>
> It was developed base v1.1.2rc3 of MatterSim. It is recommended to install **nightly-build(develop)** Paddle before running any code in this branch.
>

## **Install**

To install the package, run the following command under the root of the folder:

```bash
# install dependices
conda env create -f environment.yaml
conda activate mattersim

# install paddle on CUDA 12.6
python -m pip install --pre paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/nightly/cu126/

# install paddle on CUDA 11.8
python -m pip install --pre paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/nightly/cu118/

# install paddle-ema
pip install git+https://github.com/BeingGod/paddle_ema.git

# install paddle-geometric
pip install git+https://github.com/BeingGod/paddle_geometric.git

# install mattersim
pip install . -v
```

## **Pretained-Model Convert**

Make sure you have already installed torch before runing below command:

```bash
python src/mattersim/utils/convert_pretrained_model_utils.py
```

## **Inference**

A Minimal inference test.

```python
import os

import paddle
from ase.build import bulk
from ase.units import GPa

from mattersim.forcefield import MatterSimCalculator

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)

device = "gpu" if paddle.device.cuda.device_count() > 0 else "cpu"
paddle.device.set_device(device)
print(f"Running MatterSim on {device}")

si = bulk("Si", "diamond", a=5.43)
si.calc = MatterSimCalculator()
print(f"Energy (eV)                 = {si.get_potential_energy()}")
print(f"Energy per atom (eV/atom)   = {si.get_potential_energy()/len(si)}")
print(f"Forces of first atom (eV/A) = {si.get_forces()[0]}")
print(f"Stress[0][0] (eV/A^3)       = {si.get_stress(voigt=False)[0][0]}")
print(f"Stress[0][0] (GPa)          = {si.get_stress(voigt=False)[0][0] / GPa}")
```

## **Finetune**

A Minimal finetune test.

```bash
# 1MB
python src/mattersim/training/finetune_mattersim.py --load_model_path mattersim-v1.0.0-1M.pdparams --train_data_path tests/data/high_level_water.xyz
# 5MB
python src/mattersim/training/finetune_mattersim.py --load_model_path mattersim-v1.0.0-5M.pdparams --train_data_path tests/data/high_level_water.xyz
```

## **Known Issues**

* Not support distributed finetune.


## Reference

We kindly request that users of MatterSim version 1.0.0 cite our preprint available on arXiv:
```
@article{yang2024mattersim,
      title={MatterSim: A Deep Learning Atomistic Model Across Elements, Temperatures and Pressures},
      author={Han Yang and Chenxi Hu and Yichi Zhou and Xixian Liu and Yu Shi and Jielan Li and Guanzhi Li and Zekun Chen and Shuizhou Chen and Claudio Zeni and Matthew Horton and Robert Pinsler and Andrew Fowler and Daniel Zügner and Tian Xie and Jake Smith and Lixin Sun and Qian Wang and Lingyu Kong and Chang Liu and Hongxia Hao and Ziheng Lu},
      year={2024},
      eprint={2405.04967},
      archivePrefix={arXiv},
      primaryClass={cond-mat.mtrl-sci},
      url={https://arxiv.org/abs/2405.04967},
      journal={arXiv preprint arXiv:2405.04967}
}
```

> [!IMPORTANT]
> We kindly ask users to **explicitly** specify the exact model version and checkpoint (e.g., **MatterSim-v1.0.0-1M**) when reporting results in academic papers or technical reports, rather than referring to the model generically as **MatterSim**. Precise versioning is crucial for ensuring reproducibility. For instance, the statement "_This study was conducted using MatterSim-v1.0.0-1M_" serves as a good example.

## Limitations

**MatterSim-v1** is designed specifically for atomistic simulations of bulk materials. Applications or interpretations beyond this scope should be approached with caution. For instance, when using the model for simulations involving surfaces, interfaces, or properties influenced by long-range interactions, the results may be qualitatively accurate but are not suitable for quantitative analysis. In such cases, we recommend fine-tuning the model to better align with the specific application.

## Trademarks

This project may contain trademarks or logos for projects, products, or services.
Authorized use of Microsoft trademarks or logos is subject to and must follow [Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.

## Responsible AI Transparency Documentation

The responsible AI transparency documentation can be found [here](MODEL_CARD.md).


## Researcher and Developers
MatterSim is actively under development, and we welcome community engagement. If you have research interests related to this model, ideas you’d like to contribute, or issues to report, we encourage you to reach out to us at [ai4s-materials@microsoft.com](mailto:ai4s-materials@microsoft.com).
