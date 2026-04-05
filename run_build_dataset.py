from argparse import ArgumentParser
from pathlib import Path

from src.config import load_config
from src.pipeline import build_dataset


def main() -> None:
    parser = ArgumentParser(description="生成带局部动态 lag 的 LiquidSugar 数据集")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/liquidsugar_lag.yaml",
        help="YAML 配置文件路径",
    )
    args = parser.parse_args()

    config = load_config(Path(args.config))
    build_dataset(config)


if __name__ == "__main__":
    main()
