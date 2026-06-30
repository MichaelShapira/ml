"""Flask app implementing the SageMaker BYOC serving contract.

SageMaker hosting requires two routes on port 8080:
  - GET  /ping         -> 200 when the container is healthy
  - POST /invocations  -> run inference on the request body

For async endpoints the request body is the JSON we PUT to S3, and whatever we
return is written to the S3 OutputLocation. We load the (heavy) pipeline lazily
on the first request so /ping returns 200 quickly during the SageMaker
startup health check, avoiding a false "unhealthy" during the long model load.
"""

import json
import logging
import traceback

import flask

import sam3d_handler as sam3d  # NOT `inference` — that name collides with the repo's notebook/inference.py

logger = logging.getLogger("sam3d.predictor")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

app = flask.Flask(__name__)


@app.route("/ping", methods=["GET"])
def ping():
    # Healthy as soon as the process is up. The model loads on first /invocations;
    # the SageMaker async invocation timeout (not the health check) covers that.
    return flask.Response(response="\n", status=200, mimetype="application/json")


@app.route("/invocations", methods=["POST"])
def invocations():
    if flask.request.content_type and "json" not in flask.request.content_type:
        return _error(415, f"Unsupported content type: {flask.request.content_type}")

    try:
        body = flask.request.get_data(as_text=True)
        data = json.loads(body) if body else {}
    except json.JSONDecodeError as exc:
        return _error(400, f"Invalid JSON body: {exc}")

    try:
        result = sam3d.reconstruct(data)
    except ValueError as exc:  # client error (bad/missing fields)
        return _error(400, str(exc))
    except Exception as exc:  # server / model error
        logger.error("Inference failed: %s\n%s", exc, traceback.format_exc())
        return _error(500, f"Inference failed: {exc}")

    return flask.Response(
        response=json.dumps(result), status=200, mimetype="application/json"
    )


def _error(code: int, message: str):
    return flask.Response(
        response=json.dumps({"error": message}), status=code, mimetype="application/json"
    )
