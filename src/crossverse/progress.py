"""Dependency-free progress bar for long pipeline steps.

In a terminal it redraws one line; redirected to a log file (followed with `Get-Content -Wait` or
`tail -f`) it prints a new line every 5%.
"""

from __future__ import annotations

import sys
import time

CR = chr(13)


class Progress:
    def __init__(self, total: int, unit: str = "items", label: str = "", width: int = 30):
        self.total, self.unit, self.label, self.width = max(total, 1), unit, label, width
        self.done, self.t0, self.last_step = 0, time.time(), -1
        self.tty = sys.stdout.isatty()

    def update(self, n: int = 1) -> None:
        self.done += n
        frac = min(self.done / self.total, 1.0)
        step = int(frac * 20)  # 5% steps in log files
        if not self.tty and step == self.last_step and self.done < self.total:
            return
        self.last_step = step
        elapsed = time.time() - self.t0
        left = elapsed / frac * (1 - frac) if frac else 0
        filled = int(self.width * frac)
        line = (f"{self.label}[{'#' * filled}{'-' * (self.width - filled)}] {frac:6.1%}  {self.done}/{self.total} "
                f"{self.unit}  elapsed {elapsed // 60:.0f}m{elapsed % 60:02.0f}s  ~{left // 60:.0f}m{left % 60:02.0f}s left")
        if self.tty:
            print(CR + line, end="", flush=True)
            if self.done >= self.total:
                print()
        else:
            print(line, flush=True)
