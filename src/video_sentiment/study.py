"""Video study engine: extracts keyframes, synthesizes key concepts, and generates comprehensive study guides."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from video_sentiment.config import Settings, load_settings
from video_sentiment.errors import VideoSentimentError
from video_sentiment.llm import query_llm


def extract_video_keyframes(
    video_path: Path,
    output_dir: Path,
    interval_sec: float = 10.0,
    max_frames: int = 30,
) -> list[dict]:
    """Extract keyframes from a video file at regular intervals using OpenCV."""
    try:
        import cv2
    except ImportError:
        return []

    video_path = Path(video_path)
    if not video_path.is_file():
        return []

    keyframes_dir = output_dir / "keyframes"
    keyframes_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_sec = total_frames / fps if fps > 0 else 0

    frame_step = int(fps * interval_sec)
    if frame_step <= 0:
        frame_step = int(fps * 5)

    keyframes: list[dict] = []
    current_frame = 0
    saved_count = 0

    while current_frame < total_frames and saved_count < max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        if not ret:
            break

        timestamp_sec = current_frame / fps
        filename = f"frame_{saved_count + 1:03d}_{int(timestamp_sec)}s.jpg"
        save_path = keyframes_dir / filename
        cv2.imwrite(str(save_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

        keyframes.append({
            "index": saved_count + 1,
            "timestamp_sec": round(timestamp_sec, 2),
            "timestamp_str": _format_time(timestamp_sec),
            "filename": filename,
            "path": str(save_path.relative_to(output_dir)),
        })
        saved_count += 1
        current_frame += frame_step

    cap.release()
    return keyframes


def study_video(
    result_document: dict,
    output_dir: Path,
    video_path: Optional[Path] = None,
    settings: Optional[Settings] = None,
) -> dict:
    """Study a video from its transcript, speaker sentiment, and visual frames to generate a study guide."""
    if settings is None:
        settings = load_settings()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Extract visual keyframes if video file exists
    keyframes: list[dict] = []
    if video_path and Path(video_path).is_file():
        keyframes = extract_video_keyframes(Path(video_path), output_dir)

    # 2. Extract transcript & sentiment data
    utterances = result_document.get("utterances") or []
    summary = result_document.get("summary") or {}
    source = result_document.get("source", "Video")

    full_transcript_text = "\n".join(
        f"[{_format_time(u.get('start', 0.0))}] {u.get('speaker', 'Speaker')}: {u.get('text', '')}"
        for u in utterances
    )

    # Compile target sentiments
    subject_sentiments = []
    by_speaker = summary.get("by_speaker") or {}
    for spk, subjs in by_speaker.items():
        for subj, detail in subjs.items():
            dominant = detail.get("dominant_label", "neutral")
            subject_sentiments.append(f"- {spk} on '{subj}': {dominant}")
    sentiments_summary = "\n".join(subject_sentiments)

    # 3. Construct LLM prompt for in-depth video study
    study_prompt = f"""You are a master educator and video content analyst.
Analyze the following video transcript, speaker interactions, and aspect sentiments.
Synthesize a comprehensive, high-yield study guide that will allow a student to study this video and answer any test questions about it.

Video Source: {source}

Transcript with Timestamps:
{full_transcript_text[:12000]}

Speaker Sentiments & Opinions:
{sentiments_summary[:3000]}

Please output a comprehensive JSON object with the following structure:
{{
  "overview": "Clear 2-3 paragraph summary of the entire video, its main theme, and conclusions.",
  "core_concepts": [
    {{
      "concept": "Concept or Topic Name",
      "definition": "Detailed explanation of what was taught/discussed",
      "key_takeaway": "Critical takeaway for exams or tests"
    }}
  ],
  "speaker_perspectives": [
    {{
      "speaker": "Speaker name / label",
      "main_stance": "Their general perspective and role in the discussion",
      "key_claims": ["Claim 1 with context", "Claim 2"],
      "sentiments": "Summary of what they favored or opposed"
    }}
  ],
  "timeline_events": [
    {{
      "time_range": "mm:ss - mm:ss",
      "topic": "Topic discussed during this section",
      "summary": "Brief summary of points made"
    }}
  ],
  "high_yield_test_facts": [
    "Crucial fact, number, definition, or distinction that is prime material for a test question",
    "Another critical fact or argument mentioned in the video"
  ]
}}
Return ONLY the JSON object.
"""

    system_inst = "You are an expert academic tutor and video analyst creating structured study guides."
    study_data = None
    try:
        raw_response = query_llm(
            study_prompt,
            system_instruction=system_inst,
            ollama_url=settings.ollama_url,
            ollama_model=settings.ollama_model,
            force_json=True,
            timeout=120,
        )
        study_data = _clean_json(raw_response)
    except Exception as exc:
        print(f"[Study] LLM study generation encountered: {exc}. Generating heuristic study guide.", flush=True)
        study_data = _heuristic_study_guide(utterances, summary, source)

    study_data["keyframes"] = keyframes
    study_data["source"] = source
    study_data["duration_sec"] = result_document.get("duration_sec", 0.0)

    # Save study_guide.json
    guide_json_path = output_dir / "study_guide.json"
    guide_json_path.write_text(json.dumps(study_data, indent=2, ensure_ascii=False), encoding="utf-8")

    # Save human-readable study_guide.md
    md_content = _format_study_markdown(study_data)
    guide_md_path = output_dir / "study_guide.md"
    guide_md_path.write_text(md_content, encoding="utf-8")

    return study_data


def _format_time(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


def _clean_json(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        import re
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise VideoSentimentError("Failed to parse JSON study guide from model.")


def _heuristic_study_guide(utterances: list[dict], summary: dict, source: str) -> dict:
    """Fallback study guide generator when LLM is unavailable."""
    concepts = []
    speakers_data = {}
    test_facts = []

    for u in utterances:
        spk = u.get("speaker", "Speaker")
        text = u.get("text", "")
        if spk not in speakers_data:
            speakers_data[spk] = {"claims": [], "texts": []}
        speakers_data[spk]["texts"].append(text)

        targets = u.get("targets") or []
        for t in targets:
            subj = t.get("resolved") or t.get("surface")
            if subj and subj not in [c["concept"] for c in concepts]:
                concepts.append({
                    "concept": subj,
                    "definition": f"Discussed as '{t.get('evidence', '')}' with {t.get('sentiment', 'neutral')} sentiment.",
                    "key_takeaway": f"Speaker mentions {subj} ({t.get('sentiment')}).",
                })
            if t.get("evidence"):
                test_facts.append(f"{spk} on {subj}: '{t.get('evidence')}'")

    speaker_perspectives = []
    for spk, data in speakers_data.items():
        speaker_perspectives.append({
            "speaker": spk,
            "main_stance": f"Contributed {len(data['texts'])} statements to the discussion.",
            "key_claims": data["texts"][:3],
            "sentiments": "Varied sentiment across subjects.",
        })

    return {
        "overview": f"Study summary for {source}. Contains {len(utterances)} spoken turns across {len(speakers_data)} speakers.",
        "core_concepts": concepts[:8],
        "speaker_perspectives": speaker_perspectives,
        "timeline_events": [
            {
                "time_range": f"{_format_time(u.get('start', 0.0))} - {_format_time(u.get('end', 0.0))}",
                "topic": u.get("speaker", "Speaker"),
                "summary": u.get("text", "")[:120],
            }
            for u in utterances[:10]
        ],
        "high_yield_test_facts": test_facts[:10] or ["Review transcript and speaker sentiments for specifics."],
    }


def _format_study_markdown(data: dict) -> str:
    lines = [
        f"# Video Study Guide: {data.get('source', 'Video')}",
        f"**Duration:** {_format_time(float(data.get('duration_sec', 0)))}",
        "",
        "## Executive Overview",
        data.get("overview", "No overview provided."),
        "",
        "## Core Concepts & Key Takeaways",
    ]
    for c in data.get("core_concepts", []):
        lines.append(f"### {c.get('concept', 'Concept')}")
        lines.append(f"- **Definition/Details:** {c.get('definition', '')}")
        lines.append(f"- **Key Takeaway:** {c.get('key_takeaway', '')}")
        lines.append("")

    lines.append("## Speaker Perspectives & Sentiments")
    for spk in data.get("speaker_perspectives", []):
        lines.append(f"### {spk.get('speaker', 'Speaker')}")
        lines.append(f"**Role/Stance:** {spk.get('main_stance', '')}")
        lines.append(f"**Sentiment Overview:** {spk.get('sentiments', '')}")
        lines.append("**Key Claims:**")
        for claim in spk.get("key_claims", []):
            lines.append(f"- {claim}")
        lines.append("")

    lines.append("## High-Yield Test Facts")
    for fact in data.get("high_yield_test_facts", []):
        lines.append(f"- {fact}")
    lines.append("")

    lines.append("## Chronological Timeline")
    for t in data.get("timeline_events", []):
        lines.append(f"- **[{t.get('time_range', '')}] {t.get('topic', '')}:** {t.get('summary', '')}")

    return "\n".join(lines)
