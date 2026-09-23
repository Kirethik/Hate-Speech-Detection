
def build_prompt(task, lang, target_group, text) -> str:
    return f"{task} {lang} {target_group}: {text}"
