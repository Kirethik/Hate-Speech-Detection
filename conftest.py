"""
Repo-wide pytest setup. Runs before any test module is imported, so settings
read from the environment at import time (config.py) see these values.
"""

import os
import tempfile

# Never let tests write into the real history/feedback database.
os.environ.setdefault("DB_PATH", os.path.join(tempfile.mkdtemp(prefix="civitas-test-"), "civitas.db"))
os.environ.setdefault("CIVITAS_DATA_DIR", tempfile.mkdtemp(prefix="civitas-voice-test-"))
# Legacy tests exercise the original Model A/B (demo) path; voice tests inject fake models.
os.environ.setdefault("CIVITAS_ENGINE", "legacy")
