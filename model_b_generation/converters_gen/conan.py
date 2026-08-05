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
    Loads the CONAN dataset from HuggingFace or fallback GitHub CSV.
    Filters to English subset only.
    Maps columns: HATE_SPEECH -> hate_text, COUNTER_NARRATIVE -> response_text
    Sets language='en', style='empathetic', source='conan'.
    Returns an empty DataFrame if loading fails.
    """
    df = None
    try:
        try:
            ds = load_dataset('LanguageTreatmentFoundation/CONAN', split='train')
            df = ds.to_pandas()
        except Exception as e:
            logger.warning(f"Failed to load LanguageTreatmentFoundation/CONAN: {e}")
            try:
                ds = load_dataset('marcoguerini/CONAN', split='train')
                df = ds.to_pandas()
            except Exception as e2:
                logger.warning(f"Failed to load marcoguerini/CONAN: {e2}")
                url = "https://raw.githubusercontent.com/marcoguerini/CONAN/master/CONAN/CONAN.csv"
                resp = requests.get(url, timeout=10)
                resp.raise_for_status()
                df = pd.read_csv(io.StringIO(resp.text))
                
        if df is None or df.empty:
            return get_empty_df()

        col_mapping = {c: c.upper() for c in df.columns}
        df.rename(columns=col_mapping, inplace=True)
        
        if 'HATE_SPEECH' not in df.columns or 'COUNTER_NARRATIVE' not in df.columns:
            logger.warning("Required columns missing in CONAN")
            return get_empty_df()
            
        df = df.rename(columns={'HATE_SPEECH': 'hate_text', 'COUNTER_NARRATIVE': 'response_text'})
        
        if 'LANGUAGE' in df.columns:
            df = df[df['LANGUAGE'].str.lower().str.startswith('en')]
            
        df['language'] = 'en'
        df['style'] = 'empathetic'
        df['source'] = 'conan'
        
        df = df.dropna(subset=['hate_text', 'response_text'])
        df['split'] = df['hate_text'].apply(assign_split)
        
        return df[['hate_text', 'language', 'style', 'response_text', 'source', 'split']]
        
    except Exception as e:
        logger.error(f"Error processing CONAN: {e}")
        logger.debug(traceback.format_exc())
        return get_empty_df()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    res = convert()
    print(f"Shape: {res.shape}")
    if not res.empty:
        print(res.head())
