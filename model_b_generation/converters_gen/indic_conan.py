import pandas as pd
import hashlib
import logging
import traceback
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
    Loads IndicCONAN dataset from GitHub.
    It contains Hindi and English pairs.
    Fails gracefully returning an empty DataFrame if unavailable.
    """
    try:
        url = "https://raw.githubusercontent.com/sahoonihar/IndicCONAN/main/IndicCONAN.csv"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
                
        if df is None or df.empty:
            return get_empty_df()
            
        col_mapping = {c: c.lower() for c in df.columns}
        df.rename(columns=col_mapping, inplace=True)
        
        hs_col = next((c for c in df.columns if 'hate' in c), None)
        cn_col = next((c for c in df.columns if 'counter' in c or 'response' in c), None)
        lang_col = next((c for c in df.columns if 'lang' in c), None)
        
        if not hs_col or not cn_col:
            logger.warning("Could not identify hate_speech or counter_narrative columns in IndicCONAN")
            return get_empty_df()
            
        df = df.rename(columns={hs_col: 'hate_text', cn_col: 'response_text'})
        
        if lang_col:
            df['language'] = df[lang_col].apply(lambda x: 'hi' if 'hi' in str(x).lower() else 'en')
        else:
            df['language'] = 'hi'
            
        df['style'] = 'empathetic'
        df['source'] = 'indic_conan'
        
        df = df.dropna(subset=['hate_text', 'response_text'])
        df['split'] = df['hate_text'].apply(assign_split)
        
        return df[['hate_text', 'language', 'style', 'response_text', 'source', 'split']]
        
    except Exception as e:
        logger.warning(f"Error processing IndicCONAN (expected if repo is down/missing): {e}")
        return get_empty_df()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    res = convert()
    print(f"Shape: {res.shape}")
    if not res.empty:
        print(res.head())
