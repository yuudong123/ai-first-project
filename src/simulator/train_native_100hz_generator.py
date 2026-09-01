"""Train the separate seven-sensor 100Hz native residual generator."""

import argparse
import json

from native_rate_utils import train_rate_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()
    result = train_rate_model(100, args.epochs, args.batch_size, args.patience)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
