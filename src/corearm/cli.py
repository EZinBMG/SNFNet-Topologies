import argparse
import os
from pathlib import Path
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description="Reproduce random and core-arm network simulations.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="Run simulations and summarize their numerical outputs.")
    run_parser.add_argument("--profile", choices=("representative", "manuscript"), default="representative")
    run_parser.add_argument("--out", type=Path)
    summary_parser = subparsers.add_parser("summarize", help="Summarize the archived original simulations.")
    summary_parser.add_argument("--reference", action="store_true", required=True)
    summary_parser.add_argument("--out", type=Path, default=Path("results/reference"))
    verify_parser = subparsers.add_parser("verify", help="Check saved simulation outputs and reference differences.")
    verify_parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    try:
        if args.command == "run":
            from .workflow import run
            destination = args.out or Path("results") / args.profile
            output = run(args.profile, destination)
            from .validation import verify
            report = verify(output)
            print(f"Output: {output}")
            print(f"Numerical consistency checks: {'passed' if report['checks_passed'] else 'failed'}")
            print("Original-reference differences: reference_comparison.csv")
            return 0 if report["checks_passed"] else 1
        if args.command == "summarize":
            from .workflow import summarize_reference
            print(f"Output: {summarize_reference(args.out)}")
            return 0
        from .validation import verify
        report = verify(args.output.resolve())
        print(f"Numerical consistency checks: {'passed' if report['checks_passed'] else 'failed'}")
        print(f"Report: {args.output / 'validation.json'}")
        return 0 if report["checks_passed"] else 1
    except (ValueError, OSError, KeyError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
