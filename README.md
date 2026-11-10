# SimpleICL

[![arXiv](https://img.shields.io/badge/arXiv-2609.38173-b31b1b.svg)](https://arxiv.org/pdf/2609.38173)
[![Project Page](https://img.shields.io/badge/Project_Page-SimpleICL-2ea44f.svg)](https://simpleicl.github.io/simpleicl/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97_Hugging_Face-SimpleICL-FFD21E.svg)](https://huggingface.co/SimpleICL)

Code for **In-context Robot Learning Made Simple: A Democratized Recipe for
Manipulation Tasks**. This release combines real-robot and simulation workflows.

| Workflow | Start here | Environment |
| --- | --- | --- |
| Real-robot training, Franka FR3 deployment, paired data collection | [English](real/README.md) · [中文](real/README_zh.md) | `simpleicl-real` |
| Simulation collection, training, Native/OOD evaluation | [Simulation guide](sim/README.md) | `simpleicl-sim` |

Both workflows install a Python package named `simpleicl`. Use separate
environments and run commands from the directory named in each guide. There is
no root-level package or combined dependency installation.


```text
real/          Real-robot resources
sim/           Simulation resources
```

Resources are separate from code: follow the selected guide for datasets,
simulation assets, checkpoints, and normalization statistics.

If you find this paper useful for your work, please cite:
```
@article{li2026context,
  title={In-context Robot Learning Made Simple: A Democratized Recipe for Manipulation Tasks},
  author={Li, Minxing and Han, Minghao and Zhao, Weizhi and Wang, Hanwen and Liu, Xiangshuo and Shang, Shuyao and Zhou, Jingxiang and Sun, Mingchao and Pan, Hongyu and Xu, Mu and others},
  journal={arXiv preprint arXiv:2609.38173},
  year={2026}
}
```

If you have any questions, feel free to contact [liminxing2024@ia.ac.cn](mailto:liminxing2024@ia.ac.cn) or [minghao.hmh@gmail.com](mailto:minghao.hmh@gmail.com).
