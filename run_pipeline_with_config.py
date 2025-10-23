#!/usr/bin/env python
"""Wrapper script to run pipeline with custom configuration."""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ground_truth_pipeline import main

if __name__ == "__main__":
    # Parse arguments
    persona_config = None
    ground_truth_path = None

    if len(sys.argv) > 1:
        config_json = sys.argv[1]
        try:
            persona_config = json.loads(config_json)
        except json.JSONDecodeError as e:
            print(f"Error parsing persona config: {e}")
            sys.exit(1)

    if len(sys.argv) > 2:
        ground_truth_path = sys.argv[2]

    # Ensure experiments directory exists
    Path("experiments").mkdir(parents=True, exist_ok=True)

    # Run pipeline with config
    main(persona_config=persona_config, ground_truth_path=ground_truth_path)
