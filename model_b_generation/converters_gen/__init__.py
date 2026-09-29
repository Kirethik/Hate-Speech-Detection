"""Model B (alternate speech) converters. Each module exposes convert(raw_dir) -> DataFrame[PAIR_COLUMNS]."""

from . import conan, indic_conan, multitarget_conan, paradetox, qian
from ._common import PAIR_COLUMNS

SOURCES = {
    "paradetox": paradetox,          # rewrite, en + hi
    "multitarget_conan": multitarget_conan,  # respond, en
    "conan": conan,                  # respond, en
    "qian": qian,                    # respond, en
    "indic_conan": indic_conan,      # respond, hi + en
}
