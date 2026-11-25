"""Train the regression and time-series models from scratchpad_General.ipynb.

Usage:
    python train_signals.py --task regression
    python train_signals.py --task timeseries [--epochs 100]
Then run export_web.py to publish the results to docs/mnist/.
"""
import argparse

from signal_models import regression, timeseries


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=("regression", "timeseries"), required=True)
    parser.add_argument("--epochs", type=int, default=timeseries.EPOCHS, help="Max epochs for the time-series models.")
    args = parser.parse_args()
    if args.task == "regression":
        for task in regression.TASKS:
            regression.train(task)
    else:
        if args.epochs < 1:
            parser.error("epochs must be positive")
        timeseries.train_all(args.epochs)


if __name__ == "__main__":
    main()
