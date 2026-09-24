"""
Repo-wide pytest setup. Runs before any test module is imported, so settings
read from the environment at import time (config.py) see these values.
"""

import os
import tempfile

# Never let tests write into the real history/feedback database.
os.environ.setdefault("DB_PATH", os.path.join(tempfile.mkdtemp(prefix="civitas-test-"), "civitas.db"))
