# HDMI: Learning Interactive Humanoid Whole-Body Control from Human Videos

<div align="center">
<a href="https://hdmi-humanoid.github.io/">
  <img alt="Website" src="https://img.shields.io/badge/Website-Visit-blue?style=flat&logo=google-chrome"/>
</a>

<a href="https://www.youtube.com/watch?v=GvIBzM7ieaA&list=PL0WMh2z6WXob0roqIb-AG6w7nQpCHyR0Z&index=12">
  <img alt="Video" src="https://img.shields.io/badge/Video-YouTube-red?style=flat&logo=youtube"/>
</a>

<a href="https://arxiv.org/pdf/2509.16757">
  <img alt="Arxiv" src="https://img.shields.io/badge/Paper-Arxiv-b31b1b?style=flat&logo=arxiv"/>
</a>

<a href="https://github.com/LeCAR-Lab/HDMI/stargazers">
    <img alt="GitHub stars" src="https://img.shields.io/github/stars/LeCAR-Lab/HDMI?style=social"/>
</a>


</div>

HDMI is a novel framework that enables humanoid robots to acquire diverse whole-body interaction skills directly from monocular RGB videos of human demonstrations.

This repository contains the official training code of **HDMI: Learning Interactive Humanoid Whole-Body Control from Human Videos**.


## 🚀 Quick Start

Setup virtual environment with `uv sync` and apply mjlab patch (venv files)

```bash
patch --forward -p0 < patches/mjlab_local.patch
```

### Prepare Data

AMASS data: refer to https://github.com/Axellwppr/gentle-humanoid-training. use `scripts/data_process/generate_amass_dataset.py` to convert to HDMI format.

Lafan data: refer to https://github.com/EGalahad/lafan-process.

### Train and Evaluate

Teacher policy 
```bash
uv run scripts/train.py algo=ppo_roa_train task=G1/tracking/amass
uv run scripts/train.py algo=ppo_roa_train task=G1/tracking/lafan
```

Student policy 
```bash
uv run scripts/train.py algo=ppo_roa_finetune task=G1/tracking/lafan
```

Play policy
```bash
uv run scripts/play.py algo=ppo_roa_play task=G1/tracking/lafan checkpoint_path=run:<wandb_run_path>
```

add `export_policy=true` to export onnx model.

## Sim2Real

Please see [github.com/EGalahad/sim2real](https://github.com/EGalahad/sim2real) for details.
