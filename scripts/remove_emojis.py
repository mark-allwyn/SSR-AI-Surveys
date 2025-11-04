#!/usr/bin/env python3
"""
Remove emojis and em-dashes from all files in the repository.

This script:
- Removes Unicode emoji characters
- Removes em-dashes (—)
- Preserves format codes (%s, %d, %f, etc.)
- Processes Python, Markdown, and YAML files
- Creates a backup before making changes
"""

import re
import sys
from pathlib import Path
from typing import List, Tuple
import shutil

# Define emoji patterns
# Unicode ranges for emojis
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F700-\U0001F77F"  # alchemical symbols
    "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
    "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
    "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    "\U0001FA00-\U0001FA6F"  # Chess Symbols
    "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
    "\U00002702-\U000027B0"  # Dingbats
    "\U000024C2-\U0001F251"  # Enclosed characters
    "\u2705"                  # white heavy check mark
    "\u2714"                  # heavy check mark
    "\u274C"                  # cross mark
    "\u2611"                  # ballot box with check
    "\u2613"                  # saltire (X)
    "\u26A0"                  # warning sign
    "\u26A1"                  # high voltage
    "\u2B50"                  # star
    "\u2728"                  # sparkles
    "\u25B6"                  # play button
    "\u23F8"                  # pause button
    "\u23F9"                  # stop button
    "\u23FA"                  # record button
    "\u25C0"                  # reverse button
    "\u23CF"                  # eject button
    "\u23ED"                  # next track
    "\u23EE"                  # previous track
    "\u23EF"                  # play/pause button
    "\u25FB"                  # white square
    "\u25FC"                  # black square
    "\u25FD"                  # white small square
    "\u25FE"                  # black small square
    "\U0001F3C6"              # trophy
    "\U0001F4CA"              # bar chart
    "\U0001F4CB"              # clipboard
    "\U0001F4CC"              # pushpin
    "\U0001F4CD"              # round pushpin
    "\U0001F4CE"              # paperclip
    "\U0001F4CF"              # straight ruler
    "\U0001F4D0"              # triangular ruler
    "\U0001F4D1"              # bookmark tabs
    "\U0001F4D2"              # ledger
    "\U0001F4D3"              # notebook
    "\U0001F4D4"              # notebook with decorative cover
    "\U0001F4D5"              # closed book
    "\U0001F4D6"              # open book
    "\U0001F4D7"              # green book
    "\U0001F4D8"              # blue book
    "\U0001F4D9"              # orange book
    "\U0001F4DA"              # books
    "\U0001F4DB"              # name badge
    "\U0001F4DC"              # scroll
    "\U0001F4DD"              # memo
    "\U0001F50D"              # magnifying glass
    "\U0001F4A1"              # light bulb
    "\U0001F527"              # wrench
    "\U0001F528"              # hammer
    "\u2699"                  # gear
    "\u2692"                  # hammer and pick
    "\u26CF"                  # pick
    "\U0001F6E0"              # hammer and wrench
    "\U0001F4E6"              # package
    "\U0001F4E7"              # e-mail
    "\U0001F4E8"              # incoming envelope
    "\U0001F4E9"              # envelope with arrow
    "\U0001F4EA"              # closed mailbox with lowered flag
    "\U0001F4EB"              # closed mailbox with raised flag
    "\U0001F4EC"              # open mailbox with raised flag
    "\U0001F4ED"              # open mailbox with lowered flag
    "\U0001F4EE"              # postbox
    "\U0001F4EF"              # postal horn
    "\U0001F3C3"              # runner
    "\U0001F6B6"              # pedestrian
    "\U0001F6B4"              # bicyclist
    "\U0001F6B5"              # mountain bicyclist
    "\U0001F3C7"              # horse racing
    "\U0001F3C2"              # snowboarder
    "\U0001F3C4"              # surfer
    "\u26F7"                  # skier
    "\U0001F3CA"              # swimmer
    "\u26F9"                  # person with ball
    "\U0001F3CB"              # weight lifter
    "\U0001F6A7"              # construction
    "\U0001F6A8"              # police car light
    "\U0001F6A9"              # triangular flag
    "\U0001F6AA"              # door
    "\U0001F6AB"              # no entry sign
    "\U0001F6AC"              # smoking symbol
    "\U0001F6AD"              # no smoking symbol
    "\U0001F6AE"              # litter in bin sign
    "\U0001F6AF"              # do not litter symbol
    "\U0001F6B0"              # potable water symbol
    "\U0001F6B1"              # non-potable water symbol
    "\U0001F6B2"              # bicycle
    "\U0001F6B3"              # no bicycles
    "]",
    flags=re.UNICODE
)

# Em-dash pattern
EM_DASH_PATTERN = re.compile(r"—")

# Format code patterns to preserve (not emojis)
FORMAT_CODE_PATTERN = re.compile(r"%[sdifeFgGcoxXb]")


def is_format_code_context(text: str, pos: int, window: int = 5) -> bool:
    """Check if position is within a format code context."""
    start = max(0, pos - window)
    end = min(len(text), pos + window)
    context = text[start:end]
    return bool(FORMAT_CODE_PATTERN.search(context))


def remove_emojis_from_text(text: str) -> Tuple[str, int]:
    """
    Remove emojis and em-dashes from text.

    Returns:
        Tuple of (cleaned_text, number_of_removals)
    """
    original_length = len(text)

    # Remove emojis
    cleaned = EMOJI_PATTERN.sub('', text)

    # Remove em-dashes
    cleaned = EM_DASH_PATTERN.sub('-', cleaned)

    # Calculate removals (approximate)
    removals = original_length - len(cleaned)

    return cleaned, removals


def process_file(file_path: Path, dry_run: bool = False) -> Tuple[bool, int]:
    """
    Process a single file to remove emojis.

    Returns:
        Tuple of (was_modified, number_of_removals)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            original_content = f.read()

        cleaned_content, removals = remove_emojis_from_text(original_content)

        if cleaned_content != original_content:
            if not dry_run:
                # Create backup
                backup_path = file_path.with_suffix(file_path.suffix + '.emoji_backup')
                shutil.copy2(file_path, backup_path)

                # Write cleaned content
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(cleaned_content)

            return True, removals

        return False, 0

    except Exception as e:
        print(f"Error processing {file_path}: {e}", file=sys.stderr)
        return False, 0


def find_files_to_process(root_dir: Path) -> List[Path]:
    """Find all files to process (Python, Markdown, YAML)."""
    files = []

    # Python files
    files.extend(root_dir.glob("**/*.py"))

    # Markdown files
    files.extend(root_dir.glob("**/*.md"))

    # YAML files
    files.extend(root_dir.glob("**/*.yaml"))
    files.extend(root_dir.glob("**/*.yml"))

    # Filter out virtual environments, cache, etc.
    excluded_dirs = {'.venv', 'venv', '__pycache__', '.git', 'node_modules', '.pytest_cache'}

    filtered_files = []
    for file in files:
        if not any(excluded in file.parts for excluded in excluded_dirs):
            filtered_files.append(file)

    return sorted(filtered_files)


def main():
    """Main execution function."""
    root_dir = Path.cwd()

    print("Emoji Removal Tool")
    print("=" * 60)
    print(f"Root directory: {root_dir}")
    print()

    # Find files
    print("Scanning for files...")
    files = find_files_to_process(root_dir)
    print(f"Found {len(files)} files to process")
    print()

    # Process files
    modified_files = []
    total_removals = 0

    for file_path in files:
        relative_path = file_path.relative_to(root_dir)
        was_modified, removals = process_file(file_path, dry_run=False)

        if was_modified:
            modified_files.append((relative_path, removals))
            total_removals += removals
            print(f"✓ Modified: {relative_path} ({removals} characters removed)")
        else:
            print(f"  Skipped:  {relative_path} (no emojis found)")

    print()
    print("=" * 60)
    print("Summary:")
    print(f"  Total files scanned: {len(files)}")
    print(f"  Files modified: {len(modified_files)}")
    print(f"  Total characters removed: {total_removals}")
    print()

    if modified_files:
        print("Modified files:")
        for file_path, removals in modified_files:
            print(f"  - {file_path} ({removals} removals)")
        print()
        print("Backup files created with .emoji_backup extension")
        print("Review changes and delete backups if satisfied")
    else:
        print("No files were modified")

    return 0


if __name__ == "__main__":
    sys.exit(main())
