""" """

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class LoopLevel:
    """Class representing a loop level in the schedule."""

    # Child loop levels nested within this loop level.
    children: List[LoopLevel] = []

    # Flag indicating if this is the root loop level.
    is_root: bool = False

    # Flag indicating if this loop level represents a for-loop.
    is_for_loop: bool = False
    # The bound of the for loop, if applicable. None if the bound is not derived by the compiler.
    loop_bound: int | None = None

    # The name of the function being computed or stored at this loop level.
    function: str | None = None
    # Flag indicating if this loop level represents a store operation.
    is_store: bool = False


def parse_schedule(schedule_file: Path) -> LoopLevel:
    """Parse the schedule file and return the root loop level.

    :param schedule_file: the path to the schedule file.
    :return: root loop level of the schedule.
    """
    return LoopLevel()
