"""
Silver (machine-translated) Model B pairs: English pairs -> hi / ta / te / ml,
plus Roman Urdu via Hindi. Used by notebook 03a on Colab; the pure helpers
here are unit-tested locally.

Why Roman Urdu goes through Hindi: spoken Urdu and Hindi are largely the same
language and chat Roman Urdu is written phonetically, so romanising a Hindi
translation (e.g. "log", "sab", "insaan") is closer to how people type than
romanising Urdu script, which drops short vowels. It is still APPROXIMATE:
review_silver.py exists so you can check and fix a sample.

Each silver row keeps the English row's group_id and split, so a translation
can never leak into a different split from its original.
"""

import hashlib

import pandas as pd

from model_b_generation.converters_gen import PAIR_COLUMNS

# our language code -> (IndicTrans2 target code, post-processing)
INDICTRANS_TARGETS = {
    "hi": "hin_Deva",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "ml": "mal_Mlym",
    "ur_roman": "hin_Deva",   # translated to Hindi, then romanised
}
INDICTRANS_MODEL = "ai4bharat/indictrans2-en-indic-dist-200M"
SILVER_SOURCE = "silver_mt"


def select_english(pairs: pd.DataFrame, per_task: int = 2500, seed: int = 42) -> pd.DataFrame:
    """English gold rows to translate: up to per_task per task, every split kept."""
    en = pairs[(pairs["language"] == "en") & (pairs["source"] != SILVER_SOURCE)]
    parts = [g.sample(min(len(g), per_task), random_state=seed) for _, g in en.groupby("task")]
    return pd.concat(parts).reset_index(drop=True) if parts else en.iloc[:0]


def romanize_hindi(text: str) -> str:
    from script_augment import ScriptAugmenter
    return ScriptAugmenter(p_augment=1.0)._augment_indic(text, "hi", "native").text


def silver_id(group_id: str, language: str, task: str, target_text_en: str) -> str:
    raw = f"{group_id}|{language}|{task}|{target_text_en}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]


def make_silver_rows(english: pd.DataFrame, language: str, src_translations: list[str],
                     tgt_translations: list[str]) -> pd.DataFrame:
    if language == "ur_roman":
        src_translations = [romanize_hindi(t) for t in src_translations]
        tgt_translations = [romanize_hindi(t) for t in tgt_translations]
    rows = []
    for r, s, t in zip(english.itertuples(index=False), src_translations, tgt_translations):
        s, t = str(s).strip(), str(t).strip()
        if not s or not t:
            continue
        rows.append({
            "task": r.task, "source_text": s, "target_text": t, "language": language,
            "target": r.target, "source": SILVER_SOURCE, "split": r.split, "group_id": r.group_id,
            "silver_id": silver_id(r.group_id, language, r.task, r.target_text),
            "origin_source": r.source, "en_source_text": r.source_text,
            "en_target_text": r.target_text,
        })
    cols = PAIR_COLUMNS + ["silver_id", "origin_source", "en_source_text", "en_target_text"]
    return pd.DataFrame(rows, columns=cols)


class IndicTrans2:
    """Colab-only wrapper (needs a GPU, `IndicTransToolkit` and HF access to the model)."""

    def __init__(self, model_name: str = INDICTRANS_MODEL, device: str = "cuda"):
        import torch
        from IndicTransToolkit.processor import IndicProcessor
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        self.torch = torch
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, trust_remote_code=True, torch_dtype=torch.float16).to(device).eval()
        self.ip = IndicProcessor(inference=True)

    def translate(self, texts: list[str], tgt: str, batch_size: int = 64,
                  num_beams: int = 4) -> list[str]:
        out = []
        for i in range(0, len(texts), batch_size):
            chunk = [str(t) for t in texts[i:i + batch_size]]
            batch = self.ip.preprocess_batch(chunk, src_lang="eng_Latn", tgt_lang=tgt)
            enc = self.tok(batch, truncation=True, padding="longest", max_length=256,
                           return_tensors="pt", return_attention_mask=True).to(self.device)
            with self.torch.no_grad():
                gen = self.model.generate(**enc, use_cache=True, min_length=0, max_length=256,
                                          num_beams=num_beams, num_return_sequences=1)
            dec = self.tok.batch_decode(gen, skip_special_tokens=True,
                                        clean_up_tokenization_spaces=True)
            out += self.ip.postprocess_batch(dec, lang=tgt)
        return out
