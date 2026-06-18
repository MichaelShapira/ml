#!/usr/bin/env python3
"""
build_yiddish_dataset.py
=========================

Builds an instruction-tuning dataset for Yiddish <-> English translation by
combining several freely available public parallel corpora, cleaning and
deduplicating them, and writing the result out as JSONL files ready to feed
into a fine-tuning pipeline.

SOURCES COMBINED INTO TRAIN / VAL
----------------------------------
  1. sentence-transformers/parallel-sentences-ccmatrix  (config "en-yi")
     ~275K mined sentence pairs (web-crawled, aligned via LASER embeddings).
     Largest source, but noisier since it's automatically mined.
  2. Helsinki-NLP/opus-100                              (config "en-yi")
     ~19K sentence pairs from the OPUS collection (subtitles, software UI,
     mixed domains).
  3. Helsinki-NLP/tatoeba_mt                            (yid<->eng pair)
     Small, crowd-translated, mostly short/simple sentences. Released under
     CC-BY 2.0 (one of the more clearly-licensed sources here).

HELD OUT FOR EVALUATION (never mixed into train/val)
------------------------------------------------------
  4. openlanguagedata/flores_plus  (ydd_Hebr <-> eng_Latn)
     ~2,000 professionally human-translated sentences. Small, but high
     quality and totally independent of the mined web data above, so it's
     a much more trustworthy signal of real translation quality than a
     random held-out slice of the noisy training data would be.

OUTPUT
------
  <out-dir>/train.jsonl   (opus100 + tatoeba, the mined/crowd training data)
  <out-dir>/val.jsonl     (FLORES+ 'dev' by default: a clean, human-translated
                           evaluation set, independent of the training data)

Amazon Bedrock fine-tuning only accepts train + validation files, so no test
file is produced. FLORES+ is an evaluation benchmark (its 'dev' split is the
conventional validation set), which makes it a much better validation signal
than a random slice of the noisy mined corpora.

Each line is one training example. Default format ("alpaca"):
  {"instruction": "Translate the following Yiddish text to English.",
   "input": "<yiddish sentence>",
   "output": "<english sentence>",
   "source": "ccmatrix",
   "direction": "yi2en"}

Pass --format chatml to instead get OpenAI/HF-chat-style records:
  {"messages": [{"role": "user", "content": "..."},
                {"role": "assistant", "content": "..."}],
   "source": "ccmatrix", "direction": "yi2en"}

Pass --format bedrock to get Amazon Bedrock / Nova (e.g. Nova Lite) supervised
fine-tuning records ("bedrock-conversation-2024" schema):
  {"schemaVersion": "bedrock-conversation-2024",
   "system": [{"text": "<instruction>"}],
   "messages": [
     {"role": "user", "content": [{"text": "<source text>"}]},
     {"role": "assistant", "content": [{"text": "<translation>"}]}]}

REQUIREMENTS
------------
  pip install datasets huggingface_hub tqdm

USAGE
-----
  python build_yiddish_dataset.py --out-dir ./data --direction yi2en
  python build_yiddish_dataset.py --out-dir ./data --format bedrock
  python build_yiddish_dataset.py --out-dir ./data --direction both --format chatml
  python build_yiddish_dataset.py --out-dir ./data --sources opus100 tatoeba
  # CCMatrix is noisy and OFF by default; opt in explicitly if you want it:
  python build_yiddish_dataset.py --out-dir ./data --sources ccmatrix opus100 tatoeba

NOTE ON NETWORK ACCESS
-----------------------
This script needs normal internet access to huggingface.co to download the
source datasets. Run it on your own machine, a Colab notebook, or wherever
you have unrestricted internet -- it will not work in network-sandboxed
environments that block huggingface.co.
"""

import argparse
import json
import random
import re
import sys
import unicodedata
from pathlib import Path


# ---------------------------------------------------------------------------
# Loading individual sources -> normalized list of (yiddish, english, tag)
# ---------------------------------------------------------------------------

def load_ccmatrix():
    """sentence-transformers/parallel-sentences-ccmatrix, config 'en-yi'."""
    from datasets import load_dataset

    pairs = []
    ds = load_dataset(
        "sentence-transformers/parallel-sentences-ccmatrix", "en-yi", split="train"
    )
    for row in ds:
        en = (row.get("english") or "").strip()
        yi = (row.get("non_english") or "").strip()
        if en and yi:
            pairs.append((yi, en, "ccmatrix"))
    return pairs


def load_opus100():
    """Helsinki-NLP/opus-100, config 'en-yi'."""
    from datasets import load_dataset

    pairs = []
    ds = load_dataset("Helsinki-NLP/opus-100", "en-yi", split="train")
    for row in ds:
        t = row["translation"]
        en = (t.get("en") or "").strip()
        yi = (t.get("yi") or "").strip()
        if en and yi:
            pairs.append((yi, en, "opus100"))
    return pairs


def load_tatoeba():
    """
    Helsinki-NLP/tatoeba_mt, Yiddish<->English pair.

    Config names use sorted ISO-639-3 codes, so the Yiddish<->English subset
    lives under 'eng-yid' (eng < yid). The dataset ships a Python loading
    script, so recent versions of `datasets` require trust_remote_code=True;
    without it load_dataset() raises and this source silently contributes
    nothing -- which is exactly what was happening before. We pass it
    explicitly and fall back to a couple of legacy config spellings just in
    case the release on disk is older.

    The data is symmetric and published as a 'test' (and sometimes 'validation'
    / 'train') split; we collect everything available.
    """
    from datasets import load_dataset

    candidates = ["eng-yid", "yid-eng", "yi-en", "en-yi"]
    last_error = None
    for cfg in candidates:
        ds_dict = None
        for kwargs in ({"trust_remote_code": True}, {}):
            try:
                ds_dict = load_dataset("Helsinki-NLP/tatoeba_mt", cfg, **kwargs)
                break
            except TypeError:
                # older `datasets` without the trust_remote_code kwarg
                continue
            except Exception as e:  # config didn't exist / other load error
                last_error = e
                break
        if ds_dict is None:
            continue

        pairs = []
        for split_name in ds_dict.keys():
            for row in ds_dict[split_name]:
                src = (row.get("sourceString") or "").strip()
                tgt = (row.get("targetString") or "").strip()
                src_lang = str(row.get("sourceLang") or "")
                if not src or not tgt:
                    continue
                if src_lang.startswith("yi"):
                    pairs.append((src, tgt, "tatoeba"))
                else:
                    pairs.append((tgt, src, "tatoeba"))
        print(f"[info]   tatoeba config '{cfg}' worked "
              f"(splits: {list(ds_dict.keys())}) -> {len(pairs)} pairs")
        return pairs

    print(f"[warn] could not load Helsinki-NLP/tatoeba_mt under any of "
          f"{candidates} ({last_error}); skipping this source")
    return []


SOURCES = {
    "ccmatrix": load_ccmatrix,
    "opus100": load_opus100,
    "tatoeba": load_tatoeba,
}

# CCMatrix is automatically web-mined and ~70% of its Yiddish side is
# elongation spam / misaligned Modern Hebrew, so it is intentionally left out
# of the defaults. Pass `--sources ccmatrix opus100 tatoeba` to opt back in.
DEFAULT_SOURCES = ["opus100", "tatoeba"]


def load_flores(splits=("dev",)):
    """
    openlanguagedata/flores_plus, Eastern Yiddish (ydd_Hebr) vs English
    (eng_Latn), for the requested split(s). FLORES sentences are aligned by an
    integer id field shared across all language configs, so we load both
    configs and zip them together by id.

    FLORES+ is a professionally human-translated *evaluation* benchmark (the
    'dev' split is the conventional validation set; 'devtest' is the
    conventional test set). It is independent of the mined web training data,
    which makes it a much more trustworthy validation signal than a random
    slice of the noisy training corpora.

    Returns a list of (yiddish, english, "flores") tuples, or an empty list if
    anything about the schema doesn't match what's expected (HF dataset schemas
    do change between releases).
    """
    from datasets import load_dataset

    try:
        ds_yi = load_dataset("openlanguagedata/flores_plus", "ydd_Hebr")
        ds_en = load_dataset("openlanguagedata/flores_plus", "eng_Latn")
    except Exception as e:
        print(f"[warn] could not load FLORES+ (ydd_Hebr/eng_Latn): {e}")
        return []

    def text_field(row):
        for key in ("text", "sentence", "translation"):
            if key in row and isinstance(row[key], str):
                return row[key]
        raise KeyError(f"no text-like field found in row: {list(row.keys())}")

    pairs = []
    for split in splits:
        try:
            yi_split, en_split = ds_yi[split], ds_en[split]
        except KeyError:
            continue
        en_by_id = {row["id"]: text_field(row) for row in en_split}
        for row in yi_split:
            rid = row["id"]
            if rid in en_by_id:
                yi_text = text_field(row).strip()
                en_text = en_by_id[rid].strip()
                if yi_text and en_text:
                    pairs.append((yi_text, en_text, "flores"))
    return pairs


# ---------------------------------------------------------------------------
# Cleaning / dedup
# ---------------------------------------------------------------------------

HEBREW_SCRIPT_RE = re.compile(r"[\u0590-\u05FF\uFB1D-\uFB4F]")

# A single character repeated 4+ times in a row ("ויייייי...", "aaaa...").
# Mined web corpora (esp. CCMatrix yi) are full of this elongation spam.
REPEAT_RUN_RE = re.compile(r"(.)\1{3,}")

# Tunable thresholds for the spam / sanity filters.
MIN_CHAR_DIVERSITY = 0.25   # unique/total chars, for strings >= 20 chars
MAX_LEN_RATIO = 6.0         # gross length mismatch between the two sides


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def looks_spammy(text: str) -> bool:
    """
    Detect the junk that dominates the mined CCMatrix Yiddish side:
      * elongated character runs (same char 4+ times in a row), and
      * pathologically low character diversity in longer strings.
    """
    if REPEAT_RUN_RE.search(text):
        return True
    compact = text.replace(" ", "")
    if len(compact) >= 20 and len(set(compact)) / len(compact) < MIN_CHAR_DIVERSITY:
        return True
    return False


def is_valid_pair(yi: str, en: str, min_chars: int, max_chars: int) -> bool:
    if not (min_chars <= len(yi) <= max_chars):
        return False
    if not (min_chars <= len(en) <= max_chars):
        return False
    # The Yiddish side should actually contain Hebrew-script characters --
    # cheap filter against rows where alignment/columns got mixed up.
    if not HEBREW_SCRIPT_RE.search(yi):
        return False
    # ... and the English side should mostly NOT be Hebrew-script.
    if HEBREW_SCRIPT_RE.search(en):
        return False
    # Reject character-elongation spam / near-zero-diversity garbage on
    # either side -- this is the main source of the corrupted examples.
    if looks_spammy(yi) or looks_spammy(en):
        return False
    # Reject grossly mismatched lengths, a cheap signal of bad alignment.
    lo, hi = sorted((len(yi), len(en)))
    if lo > 0 and hi / lo > MAX_LEN_RATIO:
        return False
    return True


def clean_pairs(raw_pairs, min_chars: int, max_chars: int):
    cleaned = []
    for yi, en, src in raw_pairs:
        yi, en = normalize(yi), normalize(en)
        if is_valid_pair(yi, en, min_chars, max_chars):
            cleaned.append((yi, en, src))
    return cleaned


def dedupe(pairs):
    seen = set()
    out = []
    for yi, en, src in pairs:
        key = (yi.lower(), en.lower())
        if key in seen:
            continue
        seen.add(key)
        out.append((yi, en, src))
    return out


# ---------------------------------------------------------------------------
# Instruction formatting
# ---------------------------------------------------------------------------

YI2EN_INSTRUCTIONS = [
    "Translate the following Yiddish text to English.",
    "Translate this from Yiddish into English.",
    "Provide an accurate English translation of the given Yiddish sentence.",
]

EN2YI_INSTRUCTIONS = [
    "Translate the following English text to Yiddish.",
    "Translate this from English into Yiddish.",
    "Provide an accurate Yiddish translation of the given English sentence.",
]


def to_examples(pairs, direction, rng):
    """direction: 'yi2en', 'en2yi', or 'both'."""
    examples = []
    for yi, en, src in pairs:
        if direction in ("yi2en", "both"):
            examples.append({
                "instruction": rng.choice(YI2EN_INSTRUCTIONS),
                "input": yi,
                "output": en,
                "source": src,
                "direction": "yi2en",
            })
        if direction in ("en2yi", "both"):
            examples.append({
                "instruction": rng.choice(EN2YI_INSTRUCTIONS),
                "input": en,
                "output": yi,
                "source": src,
                "direction": "en2yi",
            })
    return examples


def to_chatml(example):
    user_content = f"{example['instruction']}\n\n{example['input']}"
    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": example["output"]},
        ],
        "source": example["source"],
        "direction": example["direction"],
    }


def to_bedrock(example):
    """
    Amazon Bedrock / Nova supervised fine-tuning record
    ("bedrock-conversation-2024" schema, used by Nova Lite text-to-text SFT).

    Each line is one conversation: an optional `system` prompt, then a
    user/assistant turn pair. `content` is a list of typed blocks; for plain
    text translation we emit a single {"text": ...} block per turn. The schema
    only allows schemaVersion/system/messages, so the source/direction tags
    used by the other formats are intentionally dropped here.

    See: https://docs.aws.amazon.com/bedrock/latest/userguide/preparing-text-data.html
    """
    return {
        "schemaVersion": "bedrock-conversation-2024",
        "system": [{"text": example["instruction"]}],
        "messages": [
            {"role": "user", "content": [{"text": example["input"]}]},
            {"role": "assistant", "content": [{"text": example["output"]}]},
        ],
    }


def format_record(example, fmt: str):
    if fmt == "chatml":
        return to_chatml(example)
    if fmt == "bedrock":
        return to_bedrock(example)
    return example  # alpaca (raw)


def write_jsonl(path: Path, examples, fmt: str):
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(format_record(ex, fmt), ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build(args, sources=SOURCES, flores_loader=load_flores):
    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- load + merge training sources -------------------------------------
    all_pairs = []
    for name in args.sources:
        loader = sources.get(name)
        if loader is None:
            print(f"[warn] unknown source '{name}', skipping")
            continue
        print(f"[info] loading {name} ...")
        try:
            pairs = loader()
        except Exception as e:
            print(f"[warn] failed to load '{name}': {e}")
            pairs = []
        print(f"[info]   -> {len(pairs)} raw pairs")
        all_pairs.extend(pairs)

    print(f"[info] total raw pairs before cleaning: {len(all_pairs)}")
    cleaned = clean_pairs(all_pairs, args.min_chars, args.max_chars)
    print(f"[info] pairs after length/script filtering: {len(cleaned)}")
    cleaned = dedupe(cleaned)
    print(f"[info] pairs after dedup: {len(cleaned)}")

    if not cleaned:
        print("[error] no usable training pairs were loaded -- check your "
              "internet access to huggingface.co and the source list.",
              file=sys.stderr)
    rng.shuffle(cleaned)

    # --- pick the validation set -------------------------------------------
    # Preferred: FLORES+ 'dev', a professionally human-translated eval set that
    # is independent of the mined training corpora -- a far cleaner validation
    # signal than a random slice of noisy training data. Falls back to carving
    # a --val-frac holdout from training if FLORES can't be loaded (Bedrock
    # fine-tuning always requires a validation file).
    val_pairs = []
    train_pairs = cleaned
    if args.val_source == "flores":
        print("[info] loading FLORES+ 'dev' as the validation set "
              "(ydd_Hebr/eng_Latn) ...")
        flores_raw = flores_loader(splits=("dev",))
        val_pairs = dedupe(clean_pairs(flores_raw, args.min_chars, args.max_chars))
        if not val_pairs:
            print("[warn] FLORES+ unavailable; falling back to a held-out "
                  f"{args.val_frac:.0%} slice of the training data")

    if not val_pairs:  # holdout fallback (or --val-source holdout)
        n_val = int(len(cleaned) * args.val_frac)
        val_pairs = cleaned[:n_val]
        train_pairs = cleaned[n_val:]
    else:
        # Leakage guard: drop any training pair that also appears in the
        # FLORES validation set (by normalized text key).
        val_keys = {(yi.lower(), en.lower()) for yi, en, _ in val_pairs}
        before = len(train_pairs)
        train_pairs = [p for p in train_pairs
                       if (p[0].lower(), p[1].lower()) not in val_keys]
        removed = before - len(train_pairs)
        if removed:
            print(f"[info] removed {removed} training pairs that overlapped "
                  f"the validation set")

    train_examples = to_examples(train_pairs, args.direction, rng)
    val_examples = to_examples(val_pairs, args.direction, rng)
    rng.shuffle(train_examples)
    rng.shuffle(val_examples)

    write_jsonl(out_dir / "train.jsonl", train_examples, args.format)
    write_jsonl(out_dir / "val.jsonl", val_examples, args.format)
    print(f"[info] wrote {len(train_examples)} examples -> {out_dir/'train.jsonl'}")
    print(f"[info] wrote {len(val_examples)} examples -> {out_dir/'val.jsonl'}")

    # --- summary -------------------------------------------------------------
    print("\nSource breakdown in train split:")
    counts = {}
    for yi, en, src in train_pairs:
        counts[src] = counts.get(src, 0) + 1
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v}")
    val_src = val_pairs[0][2] if val_pairs else "(none)"
    print(f"Validation set: {len(val_pairs)} pairs from '{val_src}'")

    return {"train": train_examples, "val": val_examples}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="./data")
    ap.add_argument("--direction", choices=["yi2en", "en2yi", "both"],
                     default="yi2en",
                     help="Which translation direction(s) to generate examples for")
    ap.add_argument("--format", choices=["alpaca", "chatml", "bedrock"],
                     default="alpaca",
                     help="Output record format. 'bedrock' emits the "
                          "bedrock-conversation-2024 schema for Amazon Nova "
                          "(e.g. Nova Lite) supervised fine-tuning.")
    ap.add_argument("--val-source", choices=["flores", "holdout"],
                     default="flores",
                     help="Where the validation set comes from. 'flores' "
                          "(default) uses FLORES+ 'dev', a clean human-"
                          "translated eval set; 'holdout' carves a --val-frac "
                          "slice out of the training data instead.")
    ap.add_argument("--val-frac", type=float, default=0.02,
                     help="Validation fraction, used only for "
                          "--val-source holdout (or as a fallback if FLORES+ "
                          "can't be loaded).")
    ap.add_argument("--min-chars", type=int, default=2)
    ap.add_argument("--max-chars", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sources", nargs="+", default=DEFAULT_SOURCES,
                     choices=list(SOURCES.keys()),
                     help="Which training sources to include. CCMatrix is "
                          "excluded by default because it is mostly corrupt; "
                          "add it explicitly if you want it.")
    return ap.parse_args(argv)


if __name__ == "__main__":
    build(parse_args())
