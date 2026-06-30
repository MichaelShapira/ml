"""WebGPU front-end server for SAM 3D Objects.

The browser (web/index.html) does ALL the segmentation in-browser via
transformers.js + WebGPU (SlimSAM) and renders the interactive 3D splat. This
tiny Flask server only:
  * serves the static page, and
  * proxies POST /generate {image, mask, seed} to the SageMaker async endpoint
    (the browser can't SigV4-sign SageMaker calls, so the signing happens here
    with your AWS creds / notebook role).

Run:
    export SAM3D_ENDPOINT=sam3d-objects-g6e
    export AWS_REGION=us-east-1
    pip install flask boto3
    python webapp.py            # then open the printed URL

Env (all optional except creds/region):
    SAM3D_ENDPOINT   endpoint name              (default: sam3d-objects-g6e)
    AWS_REGION       region                     (default: boto3 default)
    SAM3D_BUCKET     S3 I/O bucket              (default: sagemaker-<region>-<account>)
    PORT             http port                  (default: 7860)
"""

import json
import os

from flask import Flask, Response, request, send_from_directory

# Reuse the boto3 async-invoke factory from app.py (no SAM/torch needed here).
from app import _default_invoke_factory

WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
PORT = int(os.environ.get("PORT", "7860"))

app = Flask(__name__)
_invoke = None


def _get_invoke():
    global _invoke
    if _invoke is None:
        _invoke = _default_invoke_factory()
    return _invoke


@app.route("/")
def index():
    return send_from_directory(WEB_DIR, "index.html")


@app.route("/generate", methods=["POST"])
def generate():
    try:
        data = request.get_json(force=True)
    except Exception as exc:
        return Response(json.dumps({"error": f"bad JSON: {exc}"}), status=400, mimetype="application/json")

    image, mask = data.get("image"), data.get("mask")
    if not image or not mask:
        return Response(json.dumps({"error": "image and mask are required"}), status=400, mimetype="application/json")

    payload = {
        "image": image,
        "mask": mask,
        "seed": int(data.get("seed", 42)),
        "render_preview": False,   # the browser shows an interactive viewer instead
    }
    try:
        result = _get_invoke()(payload)
    except Exception as exc:
        return Response(json.dumps({"error": str(exc)}), status=502, mimetype="application/json")

    if "ply_b64" not in result:
        return Response(json.dumps({"error": f"endpoint returned no model: {result}"}),
                        status=502, mimetype="application/json")
    return Response(json.dumps(result), mimetype="application/json")


if __name__ == "__main__":
    print(f"SAM 3D WebGPU app on http://0.0.0.0:{PORT}  (endpoint: "
          f"{os.environ.get('SAM3D_ENDPOINT', 'sam3d-objects-g6e')})")
    app.run(host="0.0.0.0", port=PORT, threaded=True)
