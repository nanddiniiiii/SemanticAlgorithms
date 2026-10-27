"""sqgan: run with --config; unsupported trainers fail explicitly."""
import argparse
from src.config import load_config
from src.cli import main
if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",required=True)
    args=parser.parse_args()
    if load_config(args.config).algorithm != "sqgan":
        parser.error("Configuration belongs to a different algorithm")
    main(["train","--config",args.config])
