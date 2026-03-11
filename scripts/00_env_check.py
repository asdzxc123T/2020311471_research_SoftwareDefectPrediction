from __future__ import annotations

import shutil
import sys
from pathlib import Path


def main() -> int:
    print("Python:", sys.version.replace("\n", " "))
    print("CWD:", Path.cwd())

    tools = ["git", "python"]
    for t in tools:
        print(f"{t}:", shutil.which(t) or "NOT_FOUND")

    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

