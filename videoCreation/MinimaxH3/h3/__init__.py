"""Client-side helpers for MiniMax-H3 on SageMaker.

Bump ``__version__`` whenever a module here gains a symbol the notebook imports.
The notebook's bootstrap checks it and names the files to re-copy, which turns a
partial sync from a confusing ``ImportError`` deep in a cell into one clear
message.

Submodules
----------
prompts  H3-Context-IR style prompt composers for both task families.
client   Async endpoint client and condition builders.
deploy    Model / endpoint-config / endpoint lifecycle.
iam       Dedicated least-privilege execution role for the endpoint.
codebuild Build the image in CodeBuild when there is no local Docker daemon.
state     S3-backed cross-cell notebook state.
media    ffmpeg helpers: frame extraction, audio extraction, probing, upload.
"""

__version__ = "2026.08.05.4"

from .client import H3AsyncEndpoint, Result, audio_ref, image_ref, keyframe, video_ref
from .codebuild import build_image, ensure_codebuild_role
from .deploy import EndpointSpec, delete_endpoint, deploy, wait_for_endpoint
from .iam import (
    delete_endpoint_role,
    diagnose_endpoint_role,
    endpoint_inline_policy,
    ensure_endpoint_role,
)
from .prompts import BasePrompt, RefPrompt, Shot, dialogue, on_screen_text, voiceover
from .state import State

__all__ = [
    "H3AsyncEndpoint",
    "Result",
    "keyframe",
    "image_ref",
    "video_ref",
    "audio_ref",
    "EndpointSpec",
    "deploy",
    "wait_for_endpoint",
    "delete_endpoint",
    "ensure_endpoint_role",
    "endpoint_inline_policy",
    "delete_endpoint_role",
    "diagnose_endpoint_role",
    "build_image",
    "ensure_codebuild_role",
    "State",
    "__version__",
    "BasePrompt",
    "RefPrompt",
    "Shot",
    "dialogue",
    "voiceover",
    "on_screen_text",
]
