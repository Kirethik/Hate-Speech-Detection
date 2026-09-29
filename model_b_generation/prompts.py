"""
The one prompt format Model B is trained and run with. dataset_gen.py
(training) and infer_gen.py / generate.py (inference) both build prompts here,
so the two can never drift apart.

    "{task} | {language} | {target}: {text}"
    e.g. "rewrite | Tamil | religion: <the flagged sentence>"

task:     rewrite = say the same thing without the hate (for the speaker / live call)
          respond = a calm counter-narrative reply (for a bystander / moderator)
language: the language the OUTPUT must be in (mt0 was instruction-tuned with
          natural-language names, so names beat ISO codes)
target:   Model A's target-group label, or "unknown"
"""

TASKS = ("rewrite", "respond")

LANGUAGE_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "ur": "Urdu",
    "ur_roman": "Roman Urdu",
    "ta": "Tamil",
    "te": "Telugu",
    "ml": "Malayalam",
    "kn": "Kannada",
}


def build_prompt(task: str, language: str, text: str, target: str | None = None) -> str:
    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}, got {task!r}")
    lang = LANGUAGE_NAMES.get(language, "English")
    tgt = (target or "unknown").strip() or "unknown"
    return f"{task} | {lang} | {tgt}: {str(text).strip()}"
