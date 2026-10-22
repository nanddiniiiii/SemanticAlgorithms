"""mdjcm: run with --config; unsupported trainers fail explicitly."""
import argparse
from semantic_lossless.config import load_config
from semantic_lossless.cli import main
if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",required=True)
    args=parser.parse_args()
    if load_config(args.config).algorithm != "mdjcm":
        parser.error("Configuration belongs to a different algorithm")
    main(["train","--config",args.config])
