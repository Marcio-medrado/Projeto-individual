import argparse
from pathlib import Path

import yaml

from taia_lab.pipelines.transfer_pipeline import parse_cfg, run_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Caminho para YAML do pipeline de Transfer Learning")
    args = parser.parse_args()

    config_path = Path(args.config)
    y = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cfg = parse_cfg(y)
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
