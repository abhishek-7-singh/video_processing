"""Test Question Answering Engine: solves MCQs and open-ended test questions based on video study context."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from video_sentiment.config import Settings, load_settings
from video_sentiment.errors import VideoSentimentError
from video_sentiment.llm import query_llm
from video_sentiment.screen import capture_screen_snapshot


def answer_test_question(
    question: str,
    result_document: dict,
    study_data: Optional[dict] = None,
    settings: Optional[Settings] = None,
) -> dict:
    """Analyze the video content and answer a test question with timestamp citations and reasoning."""
    if settings is None:
        settings = load_settings()

    utterances = result_document.get("utterances") or []
    summary = result_document.get("summary") or {}

    # 1. Retrieve the most relevant utterances and targets
    ranked_contexts = _rank_relevant_segments(question, utterances)
    top_contexts = ranked_contexts[:15]

    lines = []
    for c in top_contexts:
        targets_str = ", ".join(f"{t.get('surface')}: {t.get('sentiment')}" for t in c.get("targets", []))
        line = f"[{_format_time(c['start'])} - {_format_time(c['end'])}] {c['speaker']}: \"{c['text']}\" (Sentiments: {targets_str})"
        lines.append(line)
    context_str = "\n".join(lines)

    study_summary = ""
    if study_data:
        study_summary = (
            f"Video Overview: {study_data.get('overview', '')}\n"
            f"High Yield Facts: {json.dumps(study_data.get('high_yield_test_facts', []))}\n"
        )

    q_escaped = question.replace('"', '\\"')
    # 2. Formulate prompt for LLM
    prompt = f"""You are an expert test taker and educational AI.
Answer the following test question strictly based on the provided video analysis and transcript evidence.

Test Question:
{question}

Study Overview & Facts:
{study_summary}

Most Relevant Spoken Video Segments:
{context_str}

Instructions:
1. Determine if this is Multiple Choice (MCQ), True/False, or Short Answer.
2. If Multiple Choice (e.g. options A, B, C, D):
   - Clearly state the single best selected option letter (e.g., "A", "B", "C", or "D").
   - Explain why that option is correct and why others are wrong.
3. Identify the exact video timestamp (in seconds and mm:ss) and quote where the answer is found.
4. Note which speaker provided this information and their emotional/aspect sentiment on the subject.
5. Provide a confidence score between 0.0 and 1.0.

Return ONLY a JSON object with this exact shape:
{{
  "question": "{q_escaped}",
  "question_type": "mcq", // "mcq" | "true_false" | "short_answer"
  "selected_option": "B", // Option letter if MCQ/TF, else null
  "answer_text": "Concise direct answer statement.",
  "explanation": "Detailed explanation linking the evidence in the video to the answer.",
  "evidence_quote": "Exact quote from the speaker",
  "speaker": "Speaker who stated it",
  "timestamp_sec": 12.5,
  "timestamp_str": "00:12",
  "sentiment_context": "Speaker's sentiment and tone regarding this topic",
  "confidence": 0.95
}}
"""

    system_instruction = (
        "You are an academic test solver that answers exam questions with exact evidence citations from video transcripts."
    )

    try:
        raw_output = query_llm(
            prompt,
            system_instruction=system_instruction,
            ollama_url=settings.ollama_url,
            ollama_model=settings.ollama_model,
            force_json=True,
            timeout=120,
        )
        parsed = _clean_json(raw_output)
        return _verify_and_reconcile_answer(parsed, question)
    except Exception as exc:
        print(f"[TestSolver] LLM failed: {exc}. Using heuristic test solver fallback.", flush=True)
        return _heuristic_solver(question, ranked_contexts)


def _verify_and_reconcile_answer(ans: dict, question: str) -> dict:
    options = re.findall(r"([A-D])\s*[\)\.\:\-]\s*([^\n\r]+)", question)
    if not options:
        return ans

    quote = (ans.get("evidence_quote") or "").casefold()
    if not quote:
        return ans

    quote_words = set(re.findall(r"\w+", quote))
    best_opt = None
    best_overlap = -1
    best_text = ""
    for letter, text in options:
        opt_words = set(re.findall(r"\w+", text.casefold()))
        overlap = len(opt_words & quote_words)
        if overlap > best_overlap:
            best_overlap = overlap
            best_opt = letter
            best_text = text.strip()

    current_opt = ans.get("selected_option")
    if best_opt and best_overlap > 0 and current_opt != best_opt:
        curr_text = next((t for l, t in options if l == current_opt), "")
        curr_overlap = len(set(re.findall(r"\w+", curr_text.casefold())) & quote_words)
        if best_overlap > curr_overlap:
            ans["selected_option"] = best_opt
            ans["answer_text"] = best_text

    return ans


def capture_and_solve_screen_question(
    result_document: dict,
    bbox: Optional[tuple[int, int, int, int]] = None,
    study_data: Optional[dict] = None,
    settings: Optional[Settings] = None,
    user_hint: str = "",
) -> tuple[Path, dict]:
    """Capture on-screen test question and solve it.
    
    Returns:
        (screenshot_path, answer_dict)
    """
    screenshot_path = capture_screen_snapshot(bbox=bbox)

    # If OCR is available or user provided hint, combine them
    extracted_text = _extract_text_from_image(screenshot_path)
    question_text = user_hint.strip() or extracted_text or "Answer the test question shown on the screen."

    answer = answer_test_question(
        question=question_text,
        result_document=result_document,
        study_data=study_data,
        settings=settings,
    )
    answer["screenshot_path"] = str(screenshot_path)
    return screenshot_path, answer


def _extract_text_from_image(image_path: Path) -> str:
    """Attempt basic OCR if pytesseract or Windows OCR is present."""
    try:
        import pytesseract
        from PIL import Image
        img = Image.open(image_path)
        return pytesseract.image_to_string(img).strip()
    except Exception:
        return ""


def _rank_relevant_segments(question: str, utterances: list[dict]) -> list[dict]:
    """Rank utterances by relevance to question keywords and sentiment terms."""
    q_words = set(re.findall(r"\w+", question.casefold()))
    stop_words = {"what", "which", "where", "when", "why", "who", "how", "is", "are", "the", "a", "an", "and", "or", "in", "of", "to"}
    keywords = q_words - stop_words

    scored = []
    for u in utterances:
        text = str(u.get("text", "")).casefold()
        u_words = set(re.findall(r"\w+", text))
        overlap = len(keywords & u_words)

        # Bonus for targets matching question keywords
        targets = u.get("targets") or []
        for t in targets:
            subj = str(t.get("resolved") or t.get("surface") or "").casefold()
            if subj and subj in text or any(k in subj for k in keywords):
                overlap += 2

        scored.append((overlap, u))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [u for _, u in scored]


def _heuristic_solver(question: str, ranked_contexts: list[dict]) -> dict:
    """Heuristic answer generator based on keyword overlap and sentiment extraction."""
    top = ranked_contexts[0] if ranked_contexts else {}
    quote = top.get("text", "Evidence found in transcript.")
    speaker = top.get("speaker", "Speaker 1")
    start = top.get("start", 0.0)

    # Detect MCQ options in question
    options = re.findall(r"([A-D])\s*[\)\.\:\-]\s*([^\n\r]+)", question)
    selected_option = None
    answer_text = quote

    if options:
        # Check which option matches quote best
        best_opt = options[0][0]
        best_score = -1
        quote_words = set(re.findall(r"\w+", quote.casefold()))
        for letter, opt_text in options:
            opt_words = set(re.findall(r"\w+", opt_text.casefold()))
            overlap = len(opt_words & quote_words)
            if overlap > best_score:
                best_score = overlap
                best_opt = letter
                answer_text = opt_text.strip()
        selected_option = best_opt

    sentiment_desc = "Neutral/Factual context"
    targets = top.get("targets") or []
    if targets:
        t = targets[0]
        sentiment_desc = f"{t.get('sentiment', 'neutral').capitalize()} stance regarding {t.get('resolved') or t.get('surface')}."

    return {
        "question": question,
        "question_type": "mcq" if options else "short_answer",
        "selected_option": selected_option,
        "answer_text": answer_text,
        "explanation": f"According to {speaker} at {_format_time(start)}, '{quote}'.",
        "evidence_quote": quote,
        "speaker": speaker,
        "timestamp_sec": round(start, 2),
        "timestamp_str": _format_time(start),
        "sentiment_context": sentiment_desc,
        "confidence": 0.85 if options else 0.75,
    }


def _format_time(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


def _clean_json(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise VideoSentimentError("Failed to parse JSON answer from model.")
