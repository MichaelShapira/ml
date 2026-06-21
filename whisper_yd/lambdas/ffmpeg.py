"""whisper-ffmpeg-extract-audio

Triggered by videos/* object-creates on the pipeline bucket. Extracts a
16 kHz mono PCM WAV with ffmpeg (from the Lambda layer at /opt/bin/ffmpeg)
and writes it to audio/<base>.wav, which in turn triggers the transcribe
Lambda. Works for both real video files and audio-only inputs.
"""
import os
import urllib.parse
import subprocess
import boto3

s3 = boto3.client("s3")
AUDIO_BUCKET = os.environ["AUDIO_BUCKET"]
AUDIO_PREFIX = os.environ.get("AUDIO_PREFIX", "audio/")
FFMPEG = "/opt/bin/ffmpeg"


def handler(event, context):
    for rec in event.get("Records", []):
        src_bucket = rec["s3"]["bucket"]["name"]
        key = urllib.parse.unquote_plus(rec["s3"]["object"]["key"])
        base = os.path.splitext(os.path.basename(key))[0]
        in_path = "/tmp/" + base + ".src"
        out_path = "/tmp/" + base + ".wav"

        print("Downloading s3://%s/%s" % (src_bucket, key))
        s3.download_file(src_bucket, key, in_path)

        cmd = [FFMPEG, "-y", "-i", in_path, "-vn",
               "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", out_path]
        proc = subprocess.run(cmd, capture_output=True)
        if proc.returncode != 0:
            print(proc.stderr.decode("utf-8", "replace")[-2000:])
            raise RuntimeError("ffmpeg failed for " + key)

        out_key = AUDIO_PREFIX + base + ".wav"
        s3.upload_file(out_path, AUDIO_BUCKET, out_key)
        print("Wrote s3://%s/%s" % (AUDIO_BUCKET, out_key))
        for p in (in_path, out_path):
            try:
                os.remove(p)
            except OSError:
                pass
    return {"status": "ok"}
