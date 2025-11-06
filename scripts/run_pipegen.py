#!/usr/bin/env python3
"""
scripts/run_pipegen.py

Script to run pipeline generation.

Usage:
  python3 scripts/run_pipegen.py
"""

import subprocess
from pathlib import Path
import logging


logger = logging.getLogger(__name__)


def build_pipegen():
    """Build the pipegen executable using cmake."""
    logger.info("Building pipegen executable...")
    build_dir = Path("build")
    build_dir.mkdir(exist_ok=True)
    subprocess.run(["cmake", ".."], cwd=build_dir, check=True)
    subprocess.run(["make"], cwd=build_dir, check=True)
    logger.info("Build completed.")


def main():
    # Build the pipegen executable
    build_pipegen()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
