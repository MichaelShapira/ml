"""whisper-translate-subtitles

Triggered by whisper/word-output/<uuid>.out. Reads the word-level transcript,
groups it into sentence cues, translates each cue with Amazon Bedrock and
writes <base>.<lang>.srt + .en/.<lang>.json under transcripts-en/.

Per-job parameters are read from the sibling <uuid>.cfg.json sidecar (written
by the transcribe Lambda from jobs/<base>.json):
  - modelId         Bedrock model / inference-profile id (default: env MODEL_ID)
  - targetLanguage  language to translate INTO (default: English)
  - extraPrompt     extra text appended to the system prompt (default: none)
  - base            original job base name, used to name the output objects
No sidecar => defaults (English, env MODEL_ID, output named after the uuid),
which keeps the original notebook behaviour intact.
"""
import os
import re
import json
import urllib.parse
import urllib.request
import boto3

s3 = boto3.client("s3")
bedrock = boto3.client("bedrock-runtime",
                       region_name=os.environ.get("BEDROCK_REGION", "us-east-1"))
_secrets = boto3.client("secretsmanager",
                        region_name=os.environ.get("BEDROCK_REGION", "us-east-1"))

MODEL_ID = os.environ.get("MODEL_ID", "us.anthropic.claude-opus-4-8")

# OpenAI GPT-5.x on Bedrock uses the OpenAI Responses API on the bedrock-mantle
# endpoint, authenticated with a Bedrock long-term API key (NOT Converse / the
# Lambda's SigV4 role). The key lives in Secrets Manager.
MANTLE_BASE_URL = os.environ.get(
    "MANTLE_BASE_URL", "https://bedrock-mantle.us-east-1.api.aws/openai/v1")
OPENAI_SECRET_ID = os.environ.get("OPENAI_SECRET_ID", "/agent-core/bedrock-long-term-key")
OPENAI_SECRET_KEY = os.environ.get("OPENAI_SECRET_KEY", "bedrock-long-term")
_api_key_cache = {"key": None}


def is_responses_model(model_id):
    return model_id.startswith("openai.gpt-5")


def _bedrock_api_key():
    if _api_key_cache["key"] is None:
        raw = _secrets.get_secret_value(SecretId=OPENAI_SECRET_ID)["SecretString"]
        try:
            _api_key_cache["key"] = json.loads(raw)[OPENAI_SECRET_KEY]
        except (ValueError, KeyError):
            _api_key_cache["key"] = raw  # plain-string secret fallback
    return _api_key_cache["key"]


def responses_text(model_id, system_prompt, user_text, max_tokens):
    payload = {
        "model": model_id,
        "instructions": system_prompt,
        "input": user_text,
        "max_output_tokens": max(max_tokens * 2, 4000),
        "reasoning": {"effort": "low"},
    }
    req = urllib.request.Request(
        MANTLE_BASE_URL.rstrip("/") + "/responses",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": "Bearer " + _bedrock_api_key(),
                 "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
    for item in data.get("output", []):
        for c in item.get("content", []) or []:
            if isinstance(c, dict) and c.get("type") == "output_text":
                return c.get("text", "")
    return data.get("output_text", "")
OUT_BUCKET = os.environ["OUT_BUCKET"]
OUT_PREFIX = os.environ.get("OUT_PREFIX", "transcripts-en/")
MAX_DUR = float(os.environ.get("MAX_CUE_SECONDS", "6"))
MAX_CHARS = int(os.environ.get("MAX_CUE_CHARS", "84"))
PAUSE_GAP = float(os.environ.get("PAUSE_GAP", "0.7"))
BATCH = int(os.environ.get("TRANSLATE_BATCH", "30"))
LINE_LEN = int(os.environ.get("SRT_LINE_LEN", "42"))

SENTENCE_END = set(".?!\u2026:;")


def _with_extra(base, extra):
    extra = (extra or "").strip()
    return base + "\n\nAdditional instructions:\n" + extra if extra else base


def build_story_system(target_language, extra):
    """Pass 1 prompt: turn the whole transcript into one flowing narrative."""
    base = (
        "You are an expert Yiddish-to-{tgt} translator and storyteller. You will "
        "receive the FULL text of an automatically transcribed Yiddish talk (Hebrew "
        "script), which may contain transcription glitches and lacks punctuation. "
        "Translate and retell the ENTIRE talk as one coherent, natural, well-spoken "
        "{tgt} narrative that reads as connected, flowing prose. Use context to "
        "resolve transcription errors and keep the through-line of the story, but "
        "stay faithful to what is actually said — do not invent events or add "
        "commentary. Write ONLY the narrative, wrapped in <story> and </story> tags."
    ).format(tgt=target_language)
    return _with_extra(base, extra)


def build_subtitle_prompt(target_language, extra, story):
    """Pass 2 prompt: per-line subtitles, consulting the story for context."""
    base = (
        "You are an expert Yiddish-to-{tgt} subtitle translator. You receive "
        "consecutive subtitle lines (Yiddish in Hebrew script) from an automatic "
        "transcription. A reference translation of the WHOLE talk is provided below "
        "inside <story> tags: consult it to understand the context and keep each "
        "subtitle meaningful, consistent, and connected to the overall narrative — "
        "not as isolated fragments. Translate each input line into clear, natural, "
        "idiomatic {tgt} suitable as a subtitle, faithfully and in context. Do NOT "
        "merge, split, reorder, summarize, add, or omit lines: produce exactly one "
        "translation per input line. Render religious, Talmudic, and cultural terms "
        "accurately and keep names in standard spelling. When a term would be unclear "
        "to a general audience AND you are certain of its meaning, add a brief "
        "clarification in square brackets right after it, e.g. 'Eretz [Israel]' or "
        "'the Rambam [Maimonides]'. Leave text already in {tgt} unchanged. Return "
        "ONLY a JSON array of strings: exactly one translation per input line, in the "
        "same order and the same count."
    ).format(tgt=target_language)
    base = _with_extra(base, extra)
    story = (story or "").strip()
    if story:
        base = base + "\n\n<story>\n" + story + "\n</story>"
    return base


def generate_story(full_text, model_id, target_language, extra):
    """Pass 1: produce a flowing narrative used as context for the subtitles."""
    full_text = (full_text or "").strip()
    if not full_text:
        return ""
    sysp = build_story_system(target_language, extra)
    try:
        out = converse_text(model_id, sysp, full_text, 8000)
    except Exception as e:  # noqa: BLE001 - story is best-effort context
        print("Story pass failed (%s); continuing without story context"
              % type(e).__name__)
        return ""
    m = re.search(r"<story>(.*?)</story>", out, re.S | re.I)
    return (m.group(1).strip() if m else out.strip())


def lang_slug(target_language):
    t = (target_language or "English").strip().lower()
    if t in ("english", "en", "eng"):
        return "en"
    slug = re.sub(r"[^a-z0-9]+", "-", t).strip("-")
    return slug or "xx"


def load_config(bucket, out_key):
    """Read the sibling <uuid>.cfg.json next to the .out, if present."""
    cfg_key = (out_key[:-4] if out_key.endswith(".out") else out_key) + ".cfg.json"
    try:
        body = s3.get_object(Bucket=bucket, Key=cfg_key)["Body"].read()
        return json.loads(body)
    except Exception as e:  # noqa: BLE001 - missing config is normal
        print("No job config sidecar (%s); using defaults" % type(e).__name__)
        return {}


def parse_transcript(data):
    candidates = data if isinstance(data, list) else [data]
    for c in candidates:
        if isinstance(c, str):
            try:
                c = json.loads(c)
            except (ValueError, TypeError):
                continue
        if isinstance(c, dict) and ("chunks" in c or "text" in c):
            return c
    return {"text": "", "chunks": []}


def mk_cue(words):
    text = "".join(w["text"] for w in words).strip()
    return {"start": words[0]["timestamp"][0],
            "end": words[-1]["timestamp"][1], "he": text}


def build_sentence_cues(words):
    cues, cur = [], []
    for w in words:
        ts = w.get("timestamp") or [None, None]
        ws, we = ts[0], ts[1]
        if w.get("text") is None or ws is None or we is None:
            continue
        if cur:
            cstart = cur[0]["timestamp"][0]
            cend = cur[-1]["timestamp"][1]
            ctext = "".join(x["text"] for x in cur)
            gap = ws - cend if cend is not None else 0.0
            dur = we - cstart if cstart is not None else 0.0
            if gap >= PAUSE_GAP or dur >= MAX_DUR or len(ctext) >= MAX_CHARS:
                cues.append(mk_cue(cur))
                cur = []
        cur.append(w)
        wt = w["text"].strip()
        if wt and wt[-1] in SENTENCE_END:
            cues.append(mk_cue(cur))
            cur = []
    if cur:
        cues.append(mk_cue(cur))
    return cues


def dedupe_cues(cues):
    """Merge consecutive cues with identical source text whose times overlap or
    touch. faster-whisper occasionally emits the same word/phrase twice with
    overlapping timestamps, which showed up as duplicated subtitles."""
    out = []
    for c in cues:
        if out:
            prev = out[-1]
            same = c.get("he", "").strip() == prev.get("he", "").strip()
            overlap = (
                c.get("start") is not None
                and prev.get("end") is not None
                and c["start"] <= prev["end"] + 0.05
            )
            if same and overlap:
                prev["end"] = max(prev["end"], c["end"])
                continue
        out.append(c)
    return out


def extract_json_array(s):
    s = s.strip()
    s = re.sub(r"^```(?:json)?|```$", "", s, flags=re.MULTILINE).strip()
    i, j = s.find("["), s.rfind("]")
    if i != -1 and j != -1 and j > i:
        return json.loads(s[i:j + 1])
    raise ValueError("no JSON array")


def block_text(content):
    """Return the first text block. gpt-oss prepends a reasoningContent block,
    so we must scan rather than assume content[0] is text."""
    for b in content or []:
        if isinstance(b, dict) and isinstance(b.get("text"), str):
            return b["text"]
    return ""


def as_text(v):
    """Coerce a model's per-line result to a plain string. Some models return
    an array of objects (e.g. {"translation": "..."}) instead of strings."""
    if isinstance(v, str):
        return v
    if isinstance(v, dict):
        for k in ("translation", "text", "en", "output", "value"):
            if isinstance(v.get(k), str):
                return v[k]
        return json.dumps(v, ensure_ascii=False)
    return str(v)


def converse_text(model_id, system_prompt, user_text, max_tokens):
    if is_responses_model(model_id):
        return responses_text(model_id, system_prompt, user_text, max_tokens)
    r = bedrock.converse(
        modelId=model_id, system=[{"text": system_prompt}],
        messages=[{"role": "user", "content": [{"text": user_text}]}],
        inferenceConfig={"maxTokens": max_tokens},
    )
    return block_text(r["output"]["message"]["content"])


def translate_one(text, model_id, system_prompt):
    text = (text or "").strip()
    if not text:
        return ""
    out = converse_text(model_id, system_prompt,
                        json.dumps([text], ensure_ascii=False), 1024)
    try:
        return as_text(extract_json_array(out)[0])
    except Exception:
        return out.strip()


def translate_cues(cues, model_id, system_prompt, target_language):
    texts = [c["he"] for c in cues]
    results = []
    for i in range(0, len(texts), BATCH):
        seg = texts[i:i + BATCH]
        user = ("Translate these consecutive Yiddish subtitle lines to %s. "
                "Return a JSON array of the same length and order.\n%s"
                % (target_language, json.dumps(seg, ensure_ascii=False)))
        try:
            out = converse_text(model_id, system_prompt, user, 4096)
            arr = [as_text(x) for x in extract_json_array(out)]
            if len(arr) != len(seg):
                raise ValueError("count mismatch")
        except Exception:
            arr = [translate_one(t, model_id, system_prompt) for t in seg]
        results.extend(arr)
    for c, t in zip(cues, results):
        c["en"] = t
    return cues


def dedupe_translated(cues):
    """After translation, merge consecutive cues whose final translated text is
    identical and whose times overlap/touch. Catches duplicates that differ
    slightly in the Yiddish source but collapse to the same translation."""
    out = []
    for c in cues:
        if out:
            prev = out[-1]
            a = (c.get("en") or "").strip().lower()
            b = (prev.get("en") or "").strip().lower()
            overlap = (
                c.get("start") is not None
                and prev.get("end") is not None
                and c["start"] <= prev["end"] + 0.05
            )
            if a and a == b and overlap:
                prev["end"] = max(prev["end"], c["end"])
                continue
        out.append(c)
    return out


def fmt_ts(sec):
    sec = max(0.0, sec or 0.0)
    ms = int(round(sec * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return "%02d:%02d:%02d,%03d" % (h, m, s, ms)


def wrap(text):
    words = text.split()
    lines, cur = [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > LINE_LEN:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        lines.append(cur)
    if len(lines) > 2:
        lines = [lines[0], " ".join(lines[1:])]
    return "\n".join(lines)


def to_srt(cues):
    out = []
    for i, c in enumerate(cues, 1):
        start = c["start"]
        end = max(c["end"], start + 1.0)
        body = wrap((c.get("en") or c["he"]).strip())
        out.append("%d\n%s --> %s\n%s\n" % (i, fmt_ts(start), fmt_ts(end), body))
    return "\n".join(out)


def handler(event, context):
    for rec in event.get("Records", []):
        bucket = rec["s3"]["bucket"]["name"]
        key = urllib.parse.unquote_plus(rec["s3"]["object"]["key"])
        print("Reading words s3://%s/%s" % (bucket, key))

        cfg = load_config(bucket, key)
        model_id = cfg.get("modelId") or MODEL_ID
        target_language = cfg.get("targetLanguage") or "English"
        extra_prompt = cfg.get("extraPrompt") or ""
        base = cfg.get("base") or os.path.splitext(os.path.basename(key))[0]
        slug = lang_slug(target_language)
        print("model=%s target=%s base=%s slug=%s extra=%s"
              % (model_id, target_language, base, slug, bool(extra_prompt)))

        raw = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        parsed = parse_transcript(json.loads(raw))
        words = parsed.get("chunks", [])
        cues = dedupe_cues(build_sentence_cues(words))

        # Pass 1: a flowing narrative of the whole talk for context.
        full_text = parsed.get("text") or " ".join(c["he"] for c in cues)
        story = generate_story(full_text, model_id, target_language, extra_prompt)
        print("story pass: %d chars" % len(story))

        # Pass 2: per-line subtitles that consult the story.
        system_prompt = build_subtitle_prompt(target_language, extra_prompt, story)
        cues = translate_cues(cues, model_id, system_prompt, target_language)
        cues = dedupe_translated(cues)

        doc = {"text": " ".join(c.get("en", "") for c in cues),
               "story": story, "targetLanguage": target_language,
               "modelId": model_id, "cues": cues}
        s3.put_object(Bucket=OUT_BUCKET, Key="%s%s.%s.json" % (OUT_PREFIX, base, slug),
                      Body=json.dumps(doc, ensure_ascii=False).encode("utf-8"),
                      ContentType="application/json")
        s3.put_object(Bucket=OUT_BUCKET, Key="%s%s.%s.srt" % (OUT_PREFIX, base, slug),
                      Body=to_srt(cues).encode("utf-8"),
                      ContentType="text/plain; charset=utf-8")
        print("Wrote %s%s.%s.srt/.json" % (OUT_PREFIX, base, slug))
    return {"status": "ok"}
