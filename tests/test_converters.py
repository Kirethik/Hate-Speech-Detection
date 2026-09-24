"""
Converter + build pipeline tests on tiny synthetic raw files.

The fixtures follow each converter's DOCUMENTED expected layout. They prove the
code paths work and the label mappings are right; they cannot prove the real
downloads match that layout. That is what the per-converter summaries printed
by build_dataset.py are for.
"""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from label_maps import SEVERITY_CLASSES, TARGET_CLASSES

REPO = Path(__file__).resolve().parent.parent
SCHEMA = {"text", "language", "hate_label", "target_label", "severity_label",
          "rationale_spans", "source", "split"}
SEV = {name: i for i, name in enumerate(SEVERITY_CLASSES)}


def _check_schema(df: pd.DataFrame):
    assert SCHEMA <= set(df.columns)
    assert set(df["hate_label"].unique()) <= {0, 1}
    assert df["target_label"].between(-1, len(TARGET_CLASSES) - 1).all()
    assert df["severity_label"].between(-1, len(SEVERITY_CLASSES) - 1).all()
    assert set(df["split"].unique()) <= {"train", "val"}
    assert (df["text"].str.len() > 0).all()


# --------------------------------------------------------------------------- #
# fixture writers (n distinct rows each, so dedup/splitting has material)
# --------------------------------------------------------------------------- #
def write_ruhsold(raw: Path, n=20):
    d = raw / "RUHSOLD"; d.mkdir(parents=True)
    labels = ["Abusive/Offensive", "Normal", "Religious Hate", "Sexism", "Profane/Untargeted"]
    rows = [{"tweet": f"ruhsold sample {i} yaar", "label": i % 5} for i in range(n * 5)]
    pd.DataFrame(rows).to_csv(d / "train.csv", index=False)
    return labels


def write_constraint(raw: Path):
    d = raw / "constraint2021"; d.mkdir(parents=True)
    labels = ["non-hostile", "hate,offensive", "fake", "defamation", "Hate", "fake,non-hostile", "weird"]
    for split in ("train", "val"):
        rows = [{"Unique ID": i, "Post": f"हिंदी पोस्ट {split} {i}", "Labels Set": labels[i % len(labels)]}
                for i in range(35)]
        pd.DataFrame(rows).to_csv(d / f"{split}.csv", index=False)


def write_dravidianlt(raw: Path):
    d = raw / "dravidianlt" / "telugu"; d.mkdir(parents=True)
    for split in ("train", "dev"):
        lines = ["text\tlabel"]
        for i in range(30):
            label = ["Not_offensive", "Offensive_Untargetede", "not-telugu"][i % 3]
            lines.append(f"telugu text {split} {i}\t{label}")
        (d / f"telugu_hate_{split}.tsv").write_text("\n".join(lines), encoding="utf-8")


def write_dravidiancodemix(raw: Path):
    d = raw / "DravidianCodeMix-Dataset" / "DravidianCodeMix"; d.mkdir(parents=True)
    for split in ("train", "dev"):
        lines = []
        for i in range(30):
            label = ["Not_offensive", "Offensive_Targeted_Insult_Group", "not-Tamil"][i % 3]
            lines.append(f"tamil code mix {split} {i}\t{label}")
        (d / f"tamil_offensive_full_{split}.csv").write_text("\n".join(lines), encoding="utf-8")


def write_macd(raw: Path):
    d = raw / "MACD" / "dataset_80_10_10"; d.mkdir(parents=True)
    for split in ("train", "val"):
        pd.DataFrame([{"text": f"ಕನ್ನಡ ಪಠ್ಯ {split} {i}", "label": i % 2} for i in range(30)]) \
            .to_csv(d / f"kannada_{split}.csv", index=False)


def write_hatexplain(raw: Path):
    d = raw / "HateXplain" / "Data"; d.mkdir(parents=True)
    data, div = {}, {"train": [], "val": [], "test": []}
    for i in range(30):
        pid = f"p{i}"
        hateful = i % 2 == 0
        data[pid] = {
            "post_tokens": ["these", "people", f"word{i}", "are", "vermin"],
            "annotators": [{"label": "hatespeech" if hateful else "normal",
                            "target": ["Islam"] if hateful else ["None"]}] * 3,
            "rationales": [[0, 0, 0, 0, 1]] * 2 if hateful else [],
        }
        div[["train", "val", "test"][i % 3]].append(pid)
    (d / "dataset.json").write_text(json.dumps(data), encoding="utf-8")
    (d / "post_id_divisions.json").write_text(json.dumps(div), encoding="utf-8")


def write_ieee(raw: Path):
    d = raw / "ieee_razi"; d.mkdir(parents=True)
    rows = [{"Tweet": t, "Label": l} for t, l in
            [("tum bohat bure ho 1", "hate"), ("acha din hai 2", "0"), ("تم برے ہو", "1"),
             ("kuch bhi 4", "maybe")] * 10]
    for i, r in enumerate(rows):
        r["Tweet"] = f"{r['Tweet']} #{i}"
    pd.DataFrame(rows).to_csv(d / "razi.csv", index=False)


# --------------------------------------------------------------------------- #
# per-converter tests
# --------------------------------------------------------------------------- #
def test_ruhsold_all_five_labels_map(tmp_path):
    from converters import ruhsold
    write_ruhsold(tmp_path)
    df = ruhsold.convert(str(tmp_path))
    _check_schema(df)
    assert len(df) == 100  # no row dropped by a KeyError anymore
    assert (df["severity_label"] >= 0).all()
    assert df.loc[df["hate_label"] == 0, "severity_label"].eq(SEV["normal"]).all()
    assert (df["severity_label"] == SEV["hate"]).sum() == 40  # Religious Hate + Sexism


def test_constraint_label_sets(tmp_path):
    from converters import constraint2021
    assert constraint2021.resolve_labels("hate,offensive") == (1, "hate")
    assert constraint2021.resolve_labels("Hate") == (1, "hate")
    assert constraint2021.resolve_labels("defamation") == (1, "offensive_profanity")
    assert constraint2021.resolve_labels("fake,defamation") == (1, "offensive_profanity")
    assert constraint2021.resolve_labels("non-hostile") == (0, "normal")
    assert constraint2021.resolve_labels("fake") is None          # misinformation, not abuse
    assert constraint2021.resolve_labels("fake,non-hostile") is None
    assert constraint2021.resolve_labels("weird") is None         # unknown -> dropped, not hate
    write_constraint(tmp_path)
    df = constraint2021.convert(str(tmp_path))
    _check_schema(df)
    assert (df["language"] == "hi").all()
    assert len(df) == 40  # 4 of 7 label patterns kept, x 5 each, x 2 files


def test_dravidianlt_drops_not_language_rows(tmp_path):
    from converters import dravidianlt
    write_dravidianlt(tmp_path)
    df = dravidianlt.convert(str(tmp_path))
    _check_schema(df)
    assert not df["text"].str.contains("not-").any()
    assert len(df) == 40  # 2/3 of 60
    assert set(df["hate_label"]) == {0, 1}


def test_dravidiancodemix(tmp_path):
    from converters import dravidiancodemix
    write_dravidiancodemix(tmp_path)
    df = dravidiancodemix.convert(str(tmp_path))
    _check_schema(df)
    assert len(df) == 40 and (df["language"] == "ta").all()


def test_macd_label_is_inverted(tmp_path):
    from converters import macd
    write_macd(tmp_path)
    df = macd.convert(str(tmp_path))
    _check_schema(df)
    raw0 = df[df["text"].str.endswith(" 0")]  # raw label 0 = abusive in MACD
    assert (raw0["hate_label"] == 1).all()


def test_hatexplain_rationale_and_target(tmp_path):
    from converters import hatexplain
    write_hatexplain(tmp_path)
    df = hatexplain.convert(str(tmp_path))
    _check_schema(df)
    hateful = df[df["hate_label"] == 1].iloc[0]
    spans = json.loads(hateful["rationale_spans"])
    assert [hateful["text"][s:e] for s, e in spans] == ["vermin"]
    assert hateful["target_label"] == TARGET_CLASSES.index("religion")


def test_ieee_razi_unknown_labels_dropped_and_urdu_script(tmp_path):
    from converters import ieee_razi
    write_ieee(tmp_path)
    df = ieee_razi.convert(str(tmp_path))
    _check_schema(df)
    assert len(df) == 30  # "maybe" rows dropped
    assert set(df.loc[df["text"].str.startswith("تم"), "language"]) == {"ur"}
    assert set(df.loc[~df["text"].str.startswith("تم"), "language"]) == {"ur_roman"}


def test_sbic_and_toxigen_maps_cover_real_values():
    from label_maps import SBIC_TARGET_MAP, TOXIGEN_TARGET_MAP
    for cat in ("race", "gender", "culture", "disabled", "social", "body", "victim"):
        assert SBIC_TARGET_MAP[cat] in TARGET_CLASSES
    assert SBIC_TARGET_MAP["social"] == "political"
    for grp in ("black", "asian", "chinese", "mexican", "latino", "jewish", "muslim", "women",
                "lgbtq", "trans", "bisexual", "native_american", "middle_east",
                "mental_dis", "physical_dis", "immigrant"):
        assert TOXIGEN_TARGET_MAP[grp] in TARGET_CLASSES


# --------------------------------------------------------------------------- #
# end-to-end: build_dataset.py -> prepare_splits.py on the fixtures
# --------------------------------------------------------------------------- #
def test_build_and_prepare_splits_end_to_end(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "data"
    for writer in (write_ruhsold, write_constraint, write_dravidianlt,
                   write_dravidiancodemix, write_macd, write_hatexplain, write_ieee):
        writer(raw)

    env = {"PYTHONIOENCODING": "utf-8"}
    import os
    env = {**os.environ, **env}
    r = subprocess.run([sys.executable, "build_dataset.py", "--raw_dir", str(raw),
                        "--output_dir", str(out), "--no_implicit"],
                       cwd=REPO, capture_output=True, text=True, encoding="utf-8", env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    built = pd.read_csv(out / "train.csv")
    assert "script" in built.columns
    assert {"native", "latin"} <= set(built["script"])
    assert {"kn", "ur", "ur_roman", "hi", "te", "ta", "en"} <= set(built["language"])

    r = subprocess.run([sys.executable, "prepare_splits.py", "--data_dir", str(out)],
                       cwd=REPO, capture_output=True, text=True, encoding="utf-8", env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    splits = {n: pd.read_csv(out / f"{n}.csv") for n in ("train", "val", "test")}
    for df in splits.values():
        assert "script" in df.columns
    keys = {n: set(d["text"].str.casefold()) for n, d in splits.items()}
    assert not (keys["train"] & keys["val"]) and not (keys["train"] & keys["test"])
    report = (out / "DATA_REPORT.md").read_text(encoding="utf-8")
    assert "Rows per split x language" in report and "## Warnings" in report
