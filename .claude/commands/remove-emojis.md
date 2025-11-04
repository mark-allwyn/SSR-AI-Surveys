---
description: Remove all emojis and em-dashes from all files in the repository
---

Run the emoji removal script to clean up all Python, Markdown, and YAML files in the repository.

Execute: `python3 .claude/remove_emojis.py`

This will:
- Scan all .py, .md, .yaml, and .yml files
- Remove Unicode emoji characters
- Replace em-dashes (—) with regular hyphens (-)
- Create .emoji_backup files for safety
- Show a summary of changes

After completion, review the changes and report the results to the user.
