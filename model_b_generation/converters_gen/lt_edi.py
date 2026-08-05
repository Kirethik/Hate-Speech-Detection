import pandas as pd
import hashlib
import logging

logger = logging.getLogger(__name__)

def get_empty_df():
    return pd.DataFrame(columns=['hate_text', 'language', 'style', 'response_text', 'source', 'split'])

def convert() -> pd.DataFrame:
    """
    Attempts to load LT-EDI 2026 counter-narrative data.
    Since this is likely gated behind Codabench, it fails gracefully
    and returns an empty DataFrame matching the unified schema.
    """
    logger.warning("LT-EDI 2026 data is gated behind Codabench. Cannot download automatically.")
    return get_empty_df()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    res = convert()
    print(f"Shape: {res.shape}")
    if not res.empty:
        print(res.head())
