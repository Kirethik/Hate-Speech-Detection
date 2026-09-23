
def normalize_asr_output(asr_result, language: str) -> dict:
    return {
        "native_text": asr_result.text,
        "roman_text": asr_result.text,
        "chosen_text": asr_result.text,
        "chosen_script": 'native'
    }
