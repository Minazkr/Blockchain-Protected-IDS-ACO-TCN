"""
Script to run experiments on the NSL-KDD dataset using the revision framework.
"""

import argparse
import os
import sys

# Add the parent directory to the system path to allow importing run_revision
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import run_revision as R


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for NSL-KDD experiments."""
    parser = argparse.ArgumentParser(description="Run NSL-KDD dataset training and evaluation pipeline.")
    
    parser.add_argument(
        "--nsl_train", 
        required=True, 
        help="Path to the NSL-KDD training data file."
    )
    parser.add_argument(
        "--nsl_test", 
        required=True, 
        help="Path to the NSL-KDD test data file."
    )
    parser.add_argument(
        "--out_dir", 
        default="results/nsl", 
        help="Directory to save experimental results and logs."
    )
    parser.add_argument(
        "--runs", 
        type=int, 
        default=3, 
        help="Number of independent runs to average results."
    )
    parser.add_argument(
        "--seq_lens", 
        default="1,16", 
        help="Comma-separated list of sequence lengths to evaluate."
    )
    parser.add_argument(
        "--steps", 
        default="data,main", 
        help="Execution steps to run (e.g., data preparation, main training)."
    )
    
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    # Forward the parsed arguments to the core revision runner
    R.main([
        "--dataset", "nsl-kdd",
        "--nsl_train", args.nsl_train,
        "--nsl_test", args.nsl_test,
        "--out_dir", args.out_dir,
        "--steps", args.steps,
        "--runs", str(args.runs),
        "--seq_lens", args.seq_lens,
    ])


if __name__ == "__main__":
    main()
