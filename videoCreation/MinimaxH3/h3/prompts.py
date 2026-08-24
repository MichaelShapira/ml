"""
H3-Context-IR style prompt builders.

MiniMax ships H3 as a three-module system: H3-Context-IR (a hosted rewriter),
H3-Base (the open 768p checkpoint we serve), and H3-Regenerate-2K (also hosted).
Only H3-Base is open source, and the model card is blunt about the consequence:
Context-IR "is critical to the quality of the final output", so you either call
their API for it or follow the Prompting Guidance and build the structured
representation yourself.

This module is the second option. It emits exactly the shapes the two official
guides specify, so a locally served H3-Base sees the kind of input it was
trained to receive rather than a one-line prompt.

  - Base mode  (t2va / i2va / fl2va / l2va):
        <alignment instruction, omitted for t2va>

        integrated_multimodal_description: [Shot 1] ...

        overall_soundscape: ...

        non_diegetic_music: ...

  - Reference mode (ref2va), six sections in a fixed order:
        subject_definitions / summary / retention_analysis /
        detailed_description / overall_soundscape / non_diegetic_music

Source: docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md and
docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md in MiniMaxAI/MiniMax-H3.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "CAMERA_MOTION",
    "AMPLITUDE",
    "SPEED",
    "STYLES",
    "Shot",
    "BasePrompt",
    "RefPrompt",
    "dialogue",
    "voiceover",
    "on_screen_text",
]

# Vocabulary the guide enumerates. Sticking to it matters: these are the phrases
# the rewriter emits, so they are what the checkpoint saw during training.
CAMERA_MOTION = (
    "Zoom In", "Zoom Out", "Push In", "Pull Out", "Pan Left", "Pan Right",
    "Truck Left", "Truck Right", "Tilt Up", "Tilt Down", "Pedestal Up",
    "Pedestal Down", "Arc Shot", "Tracking Shot", "Static Shot",
    "Shake Slightly", "Shake Strongly", "POV", "Roll Clockwise",
    "Roll Counterclockwise",
)
AMPLITUDE = ("with small amplitude", "with large amplitude")
SPEED = ("at slow speed", "at fast speed")
STYLES = (
    "Cinematic", "live-action", "2D-animated", "3D CG", "claymation",
    "watercolor", "vintage film",
)

CUT_VERBS = (
    "the camera cuts to", "the shot cuts to", "the shot transitions to",
    "the shot changes to", "the shot switches to",
)


# --------------------------------------------------------------------------- #
# small inline helpers
# --------------------------------------------------------------------------- #


def dialogue(speaker_phrase: str, speaker_id: str, line: str, language: str = "English", *, verb: str = "says") -> str:
    """Render one spoken line.

    Identity, ID, action and delivery live *outside* ``<d>``; only the language
    tag and the verbatim words go inside it. Speaker ids are stable across shots
    (``S1``, ``S2``, ``S1,S2`` when they speak together).
    """
    sid = speaker_id.strip().lstrip("(").rstrip(")")
    return f"{speaker_phrase} ({sid}) {verb}: <d>[{language}] {line}</d>"


def voiceover(speaker_phrase: str, speaker_id: str, line: str, language: str = "English") -> str:
    """Off-screen narration.

    The guide requires the exact phrase "says in an off-screen voiceover" and a
    statement that the on-screen character's lips stay closed - otherwise the
    model animates a mouth to match narration.
    """
    sid = speaker_id.strip().lstrip("(").rstrip(")")
    return (
        f"{speaker_phrase} ({sid}) says in an off-screen voiceover: "
        f"<d>[{language}] {line}</d> while his lips remain completely closed."
    )


def on_screen_text(text: str) -> str:
    """Quote text that is literally visible in frame, verbatim and untranslated."""
    return f'"{text}"'


def _fmt_seconds(value: float) -> str:
    return f"{value:.2f}"


def _fmt_cut_time(value: float) -> str:
    minutes = int(value // 60)
    seconds = value - minutes * 60
    return f"{minutes:02d}:{seconds:06.3f}"


# --------------------------------------------------------------------------- #
# shots
# --------------------------------------------------------------------------- #


@dataclass
class Shot:
    """One shot of the timeline.

    ``cut_time`` is the moment this shot starts, in seconds. Shot 1 carries no
    timestamp (the guide is explicit about that); every later shot must start
    with a strictly increasing cut time inside the video's duration.
    """

    text: str
    cut_time: float | None = None
    cut_verb: str = "the camera cuts to"

    def render(self, index: int) -> str:
        body = self.text.strip()
        if index == 1 or self.cut_time is None:
            return f"[Shot {index}] {body}"
        return f"[Shot {index}] At {_fmt_cut_time(self.cut_time)}, {self.cut_verb} {body}"


def _render_shots(shots: list[Shot], duration_s: float) -> str:
    if not shots:
        raise ValueError("at least one shot is required")
    previous = -1.0
    parts = []
    for i, shot in enumerate(shots, start=1):
        if i > 1:
            if shot.cut_time is None:
                raise ValueError(f"shot {i} needs a cut_time (only shot 1 may omit it)")
            if shot.cut_time <= previous:
                raise ValueError(f"shot {i} cut_time {shot.cut_time} must be greater than {previous}")
            if shot.cut_time >= duration_s:
                raise ValueError(f"shot {i} cut_time {shot.cut_time} falls outside the {duration_s}s duration")
            previous = shot.cut_time
        parts.append(shot.render(i))
    return " ".join(parts)


# --------------------------------------------------------------------------- #
# base mode: t2va / i2va / fl2va / l2va
# --------------------------------------------------------------------------- #

_BASE_MODES = ("t2va", "i2va", "fl2va", "l2va")


@dataclass
class BasePrompt:
    """Composer for the text and keyframe tasks.

    ``mode`` selects the alignment instruction that must be the first line:

      t2va   no instruction, the three core fields start immediately
      i2va   Picture 1 is the frame at 0.00s
      fl2va  Picture 1 at 0.00s, Picture 2 at the final shot / end of the video
      l2va   Picture 1 is the frame the video lands on

    Note the naming: ``mode`` here is a *prompt* shape from the writing guide.
    The wire ``task`` sent to the endpoint is only ``t2va`` or ``fl2va`` -
    i2va, fl2va and l2va are all "fl2va" with one or two keyframes.
    """

    mode: str
    shots: list[Shot]
    soundscape: str
    music: str = "N/A"
    duration_s: float = 5.0
    last_shot_index: int | None = None

    def __post_init__(self) -> None:
        if self.mode not in _BASE_MODES:
            raise ValueError(f"mode must be one of {_BASE_MODES}, got {self.mode!r}")
        if not (4.0 <= self.duration_s <= 15.0):
            raise ValueError(f"duration_s must be 4..15, got {self.duration_s}")
        if not self.soundscape.strip():
            raise ValueError("soundscape is required (use 'N/A' only for a deliberately silent video)")

    @property
    def wire_task(self) -> str:
        """The ``task`` value the endpoint expects."""
        return "t2va" if self.mode == "t2va" else "fl2va"

    def _instruction(self) -> str:
        shot_n = self.last_shot_index or len(self.shots)
        end = _fmt_seconds(self.duration_s)
        if self.mode == "t2va":
            return ""
        if self.mode == "i2va":
            return (
                "For the target video, at 0.00 seconds into the target video, "
                "<Picture 1> (from [Shot 1]) is fully referenced."
            )
        if self.mode == "fl2va":
            return (
                "How the reference pictures align with the target video - "
                "Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; "
                f"Picture 2 (from Shot {shot_n}) aligns with the {end}-second mark of the target video."
            )
        return (
            "How the reference pictures align with the target video - "
            f"<Picture 1> (from [Shot {shot_n}]) aligns with the {end}-second mark of the target video."
        )

    def render(self) -> str:
        blocks = []
        instruction = self._instruction()
        if instruction:
            blocks.append(instruction)
        blocks.append("integrated_multimodal_description: " + _render_shots(self.shots, self.duration_s))
        blocks.append("overall_soundscape: " + self.soundscape.strip())
        blocks.append("non_diegetic_music: " + (self.music.strip() or "N/A"))
        return "\n\n".join(blocks)

    def __str__(self) -> str:  # so print(prompt) shows the prompt
        return self.render()


# --------------------------------------------------------------------------- #
# reference mode: ref2va
# --------------------------------------------------------------------------- #


@dataclass
class RefPrompt:
    """Composer for full-reference (omni-reference) generation.

    The six sections are emitted in the order the guide fixes. Reference labels
    (``<Subject N>``, ``<Picture N>``, ``<Video N>``, ``<Audio N>``) must match
    the *order of the conditions array* you send: the ordering is semantic and it
    advances H3's shared audio/video rotary clock, so re-ordering the same
    references is a different request.

    ``task_types`` becomes the bracketed prefix of ``summary``, e.g.
    ``["video editing", "audio reference", "audio reuse"]``.
    """

    subject_definitions: list[str]
    task_types: list[str]
    summary: str
    retention_analysis: list[str]
    shots: list[Shot]
    soundscape: str
    music: str = "N/A"
    duration_s: float = 5.0
    style_note: str = "The target video is in realistic photographic style."

    wire_task = "ref2va"

    def __post_init__(self) -> None:
        if not self.subject_definitions:
            raise ValueError("subject_definitions is required: define every label you cite later")
        if not self.task_types:
            raise ValueError("task_types is required, e.g. ['reference generation']")
        if not (4.0 <= self.duration_s <= 15.0):
            raise ValueError(f"duration_s must be 4..15, got {self.duration_s}")
        undefined = self._undefined_labels()
        if undefined:
            raise ValueError(
                "these labels are used but never defined in subject_definitions: "
                + ", ".join(sorted(undefined))
            )

    def _labels_defined(self) -> set[str]:
        import re

        found: set[str] = set()
        for line in self.subject_definitions:
            found.update(re.findall(r"<(?:Subject|Picture|Video|Audio) \d+>", line))
        return found

    def _undefined_labels(self) -> set[str]:
        import re

        used: set[str] = set()
        body = " ".join(
            [self.summary, *self.retention_analysis, *(s.text for s in self.shots), self.soundscape, self.music]
        )
        used.update(re.findall(r"<(?:Subject|Picture|Video|Audio) \d+>", body))
        return used - self._labels_defined()

    def render(self) -> str:
        detailed = self.style_note.strip()
        shots = _render_shots(self.shots, self.duration_s)
        detailed_block = f"{detailed}\n{shots}" if detailed else shots
        return "\n\n".join(
            [
                "subject_definitions:\n" + "\n".join(s.strip() for s in self.subject_definitions),
                "summary:\n" + f"[{' + '.join(self.task_types)}] " + self.summary.strip(),
                "retention_analysis:\n" + "\n".join(r.strip() for r in self.retention_analysis),
                "detailed_description:\n" + detailed_block,
                "overall_soundscape:\n" + self.soundscape.strip(),
                "non_diegetic_music:\n" + (self.music.strip() or "N/A"),
            ]
        )

    def __str__(self) -> str:
        return self.render()
