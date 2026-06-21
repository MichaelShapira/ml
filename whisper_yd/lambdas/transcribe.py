"""whisper-invoke-transcribe

Triggered by audio/*.wav object-creates on the pipeline bucket. Submits the
audio to the async faster-whisper endpoint. If the UI (or notebook) wrote a
per-job config under jobs/<base>.json on the pipeline bucket, it is forwarded
as a sidecar <uuid>.cfg.json next to where the endpoint will drop <uuid>.out,
so the translate Lambda can pick up per-job model / target-language / extra
system-prompt parameters. No config => default behaviour (English, default
model), fully backward compatible.
"""
import os
import json
import urllib.parse
import boto3

s3 = boto3.client("s3")
smr = boto3.client("sagemaker-runtime")

ENDPOINT_NAME = os.environ["ENDPOINT_NAME"]
CONFIG_BUCKET = os.environ.get("CONFIG_BUCKET", "")
CONFIG_PREFIX = os.environ.get("CONFIG_PREFIX", "jobs/")


def _split_s3_uri(uri):
    # s3://bucket/key...
    rest = uri[len("s3://"):]
    bucket, _, key = rest.partition("/")
    return bucket, key


def _forward_config(audio_base, output_location):
    """Copy jobs/<base>.json -> <word-output>/<uuid>.cfg.json (best effort)."""
    if not CONFIG_BUCKET or not output_location:
        return
    cfg_src_key = CONFIG_PREFIX + audio_base + ".json"
    try:
        body = s3.get_object(Bucket=CONFIG_BUCKET, Key=cfg_src_key)["Body"].read()
        cfg = json.loads(body)
    except Exception as e:  # noqa: BLE001 - no config is the normal path
        print("No job config for base=%s (%s)" % (audio_base, type(e).__name__))
        return
    # Carry the original base forward so the translate output is named after it.
    cfg["base"] = audio_base
    out_bucket, out_key = _split_s3_uri(output_location)
    cfg_key = (out_key[:-4] if out_key.endswith(".out") else out_key) + ".cfg.json"
    s3.put_object(Bucket=out_bucket, Key=cfg_key,
                  Body=json.dumps(cfg, ensure_ascii=False).encode("utf-8"),
                  ContentType="application/json")
    print("Forwarded job config -> s3://%s/%s" % (out_bucket, cfg_key))


def handler(event, context):
    for rec in event.get("Records", []):
        bucket = rec["s3"]["bucket"]["name"]
        key = urllib.parse.unquote_plus(rec["s3"]["object"]["key"])
        base = os.path.splitext(os.path.basename(key))[0]
        input_uri = "s3://%s/%s" % (bucket, key)
        resp = smr.invoke_endpoint_async(
            EndpointName=ENDPOINT_NAME,
            InputLocation=input_uri,
            ContentType="audio/wav",
            Accept="application/json",
            InvocationTimeoutSeconds=3600,
        )
        out_loc = resp.get("OutputLocation")
        print("Submitted %s -> %s" % (input_uri, out_loc))
        _forward_config(base, out_loc)
    return {"status": "ok"}
