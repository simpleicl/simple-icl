"""Hydra training entrypoint shared by local and distributed launchers."""

import hydra
import torch.distributed as dist
from omegaconf import DictConfig

from simpleicl.runtime import run_training
from simpleicl.utils.config_resolvers import register_default_resolvers

register_default_resolvers()


@hydra.main(config_path="configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    try:
        run_training(cfg)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
