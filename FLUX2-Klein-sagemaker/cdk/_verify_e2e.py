import json, time, base64, io, boto3
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
import urllib.request, urllib.error

URL = "https://33de7hfeoeghfgrftqrvz4tjru0bkyyn.lambda-url.us-east-1.on.aws/"
REGION = "us-east-1"
BUCKET = "sagemaker-us-east-1-346399954218"
INPUT_KEY = "flux2-klein-inputs/verify-e2e.json"

# A tiny valid 2x2 PNG, raw base64 (NO data: prefix) — mirrors the UI fix.
PNG_B64 = ("iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAYAAABytg0kAAAAEklEQVR4nGNkYGD4z4"
           "ABGEcFAQAVAQX2bCzKAAAAAElFTkSuQmCC")

s3 = boto3.client("s3", region_name=REGION)
s3.put_object(Bucket=BUCKET, Key=INPUT_KEY, ContentType="application/json",
              Body=json.dumps({
                  "inputs": "replace background with a tropical beach, photorealistic",
                  "images": [PNG_B64],
                  "num_inference_steps": 6,
                  "guidance_scale": 2.5,
              }).encode())

creds = boto3.Session().get_credentials().get_frozen_credentials()
body = json.dumps({"inputLocation": f"s3://{BUCKET}/{INPUT_KEY}", "contentType": "application/json"}).encode()
req = AWSRequest(method="POST", url=URL, data=body, headers={"content-type": "application/json"})
SigV4Auth(creds, "lambda", REGION).add_auth(req)
hr = urllib.request.Request(URL, data=body, method="POST")
for k, v in req.prepare().headers.items():
    hr.add_header(k, v)
try:
    with urllib.request.urlopen(hr) as r:
        resp = json.loads(r.read().decode())
except urllib.error.HTTPError as e:
    print("HTTP", e.code, e.read().decode()); raise SystemExit(1)

print("invoke:", json.dumps(resp))
ok = resp["outputLocation"].split(f"{BUCKET}/", 1)[1]
fail = resp["failureLocation"].split(f"{BUCKET}/", 1)[1]
for i in range(60):
    try:
        s3.head_object(Bucket=BUCKET, Key=ok); print(f"SUCCESS output at {ok} (~{i*3}s)"); break
    except Exception:
        pass
    try:
        s3.head_object(Bucket=BUCKET, Key=fail)
        print("FAILURE record written:")
        print(s3.get_object(Bucket=BUCKET, Key=fail)["Body"].read().decode()[:800]); break
    except Exception:
        pass
    time.sleep(3)
else:
    print("neither output nor failure within ~180s")
