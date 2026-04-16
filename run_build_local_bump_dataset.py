from argparse import ArgumentParser
from pathlib import Path

from src.config import load_config
from src.local_bump import build_local_bump_dataset


def main() -> None:
    parser = ArgumentParser(description="Generate a local-bump lag dataset for LiquidSugar.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/liquidsugar_local_bump.yaml",
        help="YAML config path.",
    )
    args = parser.parse_args()

    config = load_config(Path(args.config))
    build_local_bump_dataset(config)


if __name__ == "__main__":
    main()
