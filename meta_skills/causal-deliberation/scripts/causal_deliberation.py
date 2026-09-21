"""Demonstrate suspended causal reasoning without operating a game."""
import argparse
import json
from acr_agi3.agent.deliberation import ThoughtState

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", default="Which condition opens the blocked route?")
    args = parser.parse_args()
    state = ThoughtState()
    state.ask(args.question)
    print(json.dumps(state.snapshot()))

if __name__ == "__main__":
    main()
