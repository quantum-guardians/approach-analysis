#!/usr/bin/env python3
"""Entry point for the n-hop approach analysis.

Usage examples
--------------
Analyse a single random graph (correlation plots)::

    python main.py analyse

Run with custom parameters::

    python main.py analyse --vertices 5 --connectivity 0.7 --seed 42 --output out.png

Compare n-hop counts and SC ratio across multiple graphs::

    python main.py nhop-connectivity --vertices 5 --num-graphs 20 --num-orientations 300 --seed 0
"""

import argparse

from src.commands import analyse, nhop_connectivity


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyse n-hop approach on random graph orientations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True

    analyse.register_parser(subparsers)
    nhop_connectivity.register_parser(subparsers)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

