import re

with open('N:/Civitas/converters/dravidiancodemix.py', 'r', encoding='utf-8') as f:
    text = f.read()

text = text.replace('"malayalam": ("ml", "mal_full_offensive"),', '"malayalam": ("ml", "mal_full_offensive"),  # Note: The raw data might lack Malayalam files; dravidianlt.py (Phase 2) will cover it if so.')

with open('N:/Civitas/converters/dravidiancodemix.py', 'w', encoding='utf-8') as f:
    f.write(text)
