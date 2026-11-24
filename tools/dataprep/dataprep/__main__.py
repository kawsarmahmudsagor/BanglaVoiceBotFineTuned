"""Entry point: python -m dataprep <command> ..."""
from __future__ import annotations

import argparse
import sys

from . import export, listen, roundtrip_cosyvoice, select
from .sources import bengaliai
from .steps import audio_stats, brouhaha, dnsmos, speakers, text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dataprep", description="Speech dataset curation CLI")
    commands = parser.add_subparsers(dest="command", required=True)
    roundtrip_cosyvoice.add_parser(commands)
    text.add_parser(commands)
    audio_stats.add_parser(commands)
    dnsmos.add_parser(commands)
    brouhaha.add_parser(commands)
    speakers.add_parser(commands)
    listen.add_parser(commands)
    select.add_parser(commands)
    export.add_parser(commands)
    imp = commands.add_parser("import", help="import a source corpus into the work dir")
    bengaliai.add_parser(imp.add_subparsers(dest="source", required=True))
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
