"""
Cross-cell state for the notebook, stored in S3.

The problem this solves: a notebook that builds up Python variables cell by cell
forces you to re-run everything after a kernel restart. With a p5 endpoint at
$63/hour and demos that take minutes each, re-running section 7 to get to section
10 is real money.

So every expensive artifact records its *location* here instead of living only in
a variable: the role ARN, the image URI, the endpoint name, and the S3 URI of
every clip and derived keyframe. State lives in S3 rather than on local disk so it
survives a kernel restart, a new Studio space, or moving to a different machine.

Usage::

    state = State(s3_client, "s3://bucket/prefix/_notebook_state.json")
    state.put("demo.t2va", {"s3_uris": [...], "request_id": "..."})
    ...
    # after a kernel restart, in any cell:
    clip = state.require("demo.t2va")["s3_uris"][0]

Deliberately dumb: a single JSON object, read-modify-write, no locking. One
notebook, one operator.
"""

from __future__ import annotations

import json
import time
import urllib.parse
from typing import Any

__all__ = ["State"]


class State:
    def __init__(self, s3_client, uri: str) -> None:
        if not uri.startswith("s3://"):
            raise ValueError(f"state uri must be s3://..., got {uri!r}")
        parsed = urllib.parse.urlparse(uri)
        self.s3 = s3_client
        self.bucket = parsed.netloc
        self.key = parsed.path.lstrip("/")
        self.uri = uri
        self._data: dict[str, Any] = {}
        self.load()

    # ------------------------------------------------------------------ io

    def load(self) -> dict[str, Any]:
        try:
            body = self.s3.get_object(Bucket=self.bucket, Key=self.key)["Body"].read()
            self._data = json.loads(body.decode())
        except Exception:  # noqa: BLE001 - absent or unreadable both mean "start fresh"
            self._data = {}
        return self._data

    def flush(self) -> None:
        self._data["_updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.s3.put_object(
            Bucket=self.bucket,
            Key=self.key,
            Body=json.dumps(self._data, indent=1, sort_keys=True).encode(),
            ContentType="application/json",
        )

    # --------------------------------------------------------------- access

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def require(self, key: str, hint: str = "") -> Any:
        """Fetch a value or fail with a message naming the cell that produces it."""
        # Re-read once: another cell (or another kernel) may have written it.
        if key not in self._data:
            self.load()
        if key not in self._data:
            extra = f" Run: {hint}" if hint else ""
            raise KeyError(f"'{key}' is not in the notebook state yet.{extra}")
        return self._data[key]

    def put(self, key: str, value: Any) -> Any:
        self._data[key] = value
        self.flush()
        return value

    def drop(self, key: str) -> None:
        self._data.pop(key, None)
        self.flush()

    def keys(self) -> list[str]:
        return sorted(k for k in self._data if not k.startswith("_"))

    def summary(self) -> str:
        if not self.keys():
            return f"state is empty ({self.uri})"
        lines = [f"state ({self.uri}, updated {self._data.get('_updated_at', '?')}):"]
        for key in self.keys():
            value = self._data[key]
            if isinstance(value, dict):
                if "s3_uris" in value:
                    rendered = f"{len(value['s3_uris'])} clip(s), {value.get('request_id', '?')}"
                else:
                    rendered = ", ".join(sorted(value)[:4])
            elif isinstance(value, list):
                rendered = f"{len(value)} items"
            else:
                rendered = str(value)
            lines.append(f"  {key:<28} {rendered}")
        return "\n".join(lines)

    # ------------------------------------------------------------ memoizing

    def cached(self, key: str, produce, *, force: bool = False, verbose: bool = True) -> Any:
        """Return the recorded value, or call ``produce()`` and record it.

        This is what makes a cell safe to re-run: the expensive call happens once,
        and every later execution is a dictionary lookup. Pass ``force=True`` to
        redo it deliberately.
        """
        if not force:
            if key not in self._data:
                self.load()
            if key in self._data:
                if verbose:
                    print(f"[cached] {key} - already done, skipping. Pass force=True to redo.")
                return self._data[key]
        value = produce()
        self.put(key, value)
        return value
