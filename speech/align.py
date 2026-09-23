
def align_rationale_to_audio(rationale_spans: list[dict], asr_words: list, text: str) -> list[dict]:
    for s in rationale_spans:
        s['audio_start'] = 0.0
        s['audio_end'] = 1.0
    return rationale_spans
