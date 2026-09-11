#!/usr/bin/env python3
"""
Contract Tester - Core CLI Tool
"""

import argparse
import sys


def run(input_val: str | None = None) -> str:
    """Core task execution."""
    print(f"Executing contract-tester with input: {input_val}")
    return "Success"


def main():
    parser = argparse.ArgumentParser(description="Contract Tester execution script.")
    parser.add_argument("input_pos", nargs="?", default=None, help="Positional input argument")
    parser.add_argument("--input", "-i", dest="input_opt", type=str, default=None, help="Input")
    args = parser.parse_args()

    input_val = args.input_opt or args.input_pos
    run(input_val)
    return 0


if __name__ == "__main__":
    sys.exit(main())
