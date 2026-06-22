#!/usr/bin/env python3
"""Extract H (entropy) changes from experiment log and write to CSV.

Usage:
    python3 tools/extract_entropy.py path/to/experiment.log
    python3 tools/extract_entropy.py path/to/experiment.log -o output.csv
"""

import argparse
import re
import sys
from pathlib import Path


def extract_h_values(log_path: str) -> list[tuple[int, float, float, float, int]]:
    lines = []
    with open(log_path) as f:
        for line in f:
            m = re.search(
                r"Belief update: H=([\d.-]+)→([\d.-]+)\s+IG=([\d.-]+)\s+candidates=(\d+)",
                line,
            )
            if m:
                lines.append((
                    len(lines) + 1,
                    float(m.group(1)),
                    float(m.group(2)),
                    float(m.group(3)),
                    int(m.group(4)),
                ))
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract entropy H from log")
    parser.add_argument("log", help="Path to experiment log file")
    parser.add_argument("-o", "--output", help="Output CSV file")
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        print(f"Error: {log_path} not found", file=sys.stderr)
        sys.exit(1)

    entries = extract_h_values(str(log_path))

    csv = "n,index,H_before,H_after,IG,candidates\n"
    csv += "\n".join(
        f"{len(entries)},{i},{h0:.4f},{h1:.4f},{ig:+.4f},{cand}"
        for i, (_, h0, h1, ig, cand) in enumerate(entries, 1)
    )
    csv += "\n"

    if args.output:
        Path(args.output).write_text(csv)
        print(f"Written {len(entries)} lines to {args.output}")
    else:
        sys.stdout.write(csv)


if __name__ == "__main__":
    main()
