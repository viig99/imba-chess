"""Mirror a KLENT run's metrics.jsonl into TensorBoard (x-axis = positions played).

Runs beside the training process and never touches it: re-reading the file
from the start on launch, then following appended lines.
"""

import argparse
import json
from pathlib import Path
import time

from torch.utils.tensorboard import SummaryWriter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="KLENT output directory")
    parser.add_argument("--follow", action="store_true")
    args = parser.parse_args()
    metrics = args.run / "metrics.jsonl"
    writer = SummaryWriter(args.run / "tb", purge_step=0)
    offset = 0
    while True:
        if metrics.exists():
            with metrics.open() as stream:
                stream.seek(offset)
                for line in iter(stream.readline, ""):
                    if not line.endswith("\n"):
                        break  # partially written line; retry next poll
                    offset += len(line.encode())
                    row = json.loads(line)
                    step = row["positions"]
                    for key, value in row.items():
                        if isinstance(value, (int, float)) and not isinstance(value, bool):
                            writer.add_scalar(key if "/" in key else f"run/{key}", value, step)
            writer.flush()
        if not args.follow:
            break
        time.sleep(30)


if __name__ == "__main__":
    main()
