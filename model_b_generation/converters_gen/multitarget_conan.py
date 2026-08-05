import pandas as pd
import hashlib
import logging
import traceback
from datasets import load_dataset
import requests
import io

logger = logging.getLogger(__name__)

def get_empty_df():
    return pd.DataFrame(columns=['hate_text', 'language', 'style', 'response_text', 'source', 'split'])

def assign_split(text):
    if not isinstance(text, str):
        text = str(text)
    hash_val = int(hashlib.md5(text.encode('utf-8')).hexdigest(), 16)
    return 'val' if hash_val % 1000 < 100 else 'train'

def convert() -> pd.DataFrame:
    """
    Loads Multitarget-CONAN dataset from HuggingFace or fallback GitHub CSV.
    Maps columns: HATE_SPEECH -> hate_text, COUNTER_NARRATIVE -> response_text.
    Sets language='en', style='empathetic', source='multitarget_conan'.
    """
    df = None
    try:
        try:
            ds = load_dataset('Rhma/Multitarget-CONAN', split='train')
            df = ds.to_pandas()
        except Exception as e:
            logger.warning(f"Failed to load Rhma/Multitarget-CONAN: {e}")
            url = "https://raw.githubusercontent.com/marcoguerini/Multitarget-CONAN/master/Multitarget-CONAN.csv"
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            df = pd.read_csv(io.StringIO(resp.text))
                
        if df is None or df.empty:
            return get_empty_df()

        col_mapping = {c: c.upper() for c in df.columns}
        df.rename(columns=col_mapping, inplace=True)
        
        if 'HATE_SPEECH' not in df.columns or 'COUNTER_NARRATIVE' not in df.columns:
            logger.warning("Required columns missing in Multitarget-CONAN")
            return get_empty_df()
            
        df = df.rename(columns={'HATE_SPEECH': 'hate_text', 'COUNTER_NARRATIVE': 'response_text'})
        
        df['language'] = 'en'
        df['style'] = 'empathetic'
        df['source'] = 'multitarget_conan'
        
        df = df.dropna(subset=['hate_text', 'response_text'])
        df['split'] = df['hate_text'].apply(assign_split)
        
        return df[['hate_text', 'language', 'style', 'response_text', 'source', 'split']]
        
    except Exception as e:
        logger.error(f"Error processing Multitarget-CONAN: {e}")
        logger.debug(traceback.format_exc())
        return get_empty_df()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    res = convert()
    print(f"Shape: {res.shape}")
    if not res.empty:
        print(res.head())
