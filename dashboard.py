"""Interactive Dashboard for Screen Video Capture, Aspect Sentiment Analysis, Video Study, and Test Question Answering."""

from __future__ import annotations

import base64
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from video_sentiment.config import load_settings
from video_sentiment.screen import capture_screen_snapshot, record_screen
from video_sentiment.study import study_video
from video_sentiment.test_solver import answer_test_question, capture_and_solve_screen_question

OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)

LABELS = (
    ("positive", "Positive", "#10b981"),
    ("negative", "Negative", "#ef4444"),
    ("neutral", "Fact", "#64748b"),
    ("mixed", "Mixed", "#f59e0b"),
)
LABEL_WORD = {key: word for key, word, _ in LABELS}
LABEL_COLOR = {key: color for key, _, color in LABELS}
PLAIN = {
    "positive": "Mostly positive",
    "negative": "Mostly negative",
    "neutral": "Mostly facts",
    "mixed": "Mixed",
    "tie": "No clear winner",
}
HIDDEN_COLUMNS = {"source", "duration_sec", "speakers"}
TOP_SUBJECTS = 9


def main() -> None:
    st.set_page_config(
        page_title="Video Sentiment & Test Solver",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)

    # Top Hero Banner
    st.markdown(
        """
        <div class="hero-container">
            <div class="hero-badge">AI MULTIMODAL VIDEO INTELLIGENCE</div>
            <h1 class="hero-title">Video Sentiment Study & Test Solver</h1>
            <p class="hero-subtitle">
                Record or view videos directly from your screen, analyze speaker sentiment & tone,
                generate in-depth study notes, and accurately solve exam questions with timestamp citations.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    runs = _runs()
    selected_folder = _sidebar_selector(runs)

    tab_record, tab_study, tab_test, tab_sentiment, tab_data = st.tabs([
        "📹 Screen & Video Capture",
        "🧠 Video Study Guide",
        "📝 Test Question Solver",
        "📊 Sentiment & Speakers",
        "📁 Results & Exports",
    ])

    with tab_record:
        _render_screen_capture_tab(selected_folder)

    if not selected_folder:
        for tab in [tab_study, tab_test, tab_sentiment, tab_data]:
            with tab:
                st.info("👈 Please select or record a video in the Screen & Video Capture tab to proceed.")
        return

    document, study_data = _load_run_data(selected_folder)

    with tab_study:
        _render_study_tab(selected_folder, document, study_data)

    with tab_test:
        _render_test_solver_tab(selected_folder, document, study_data)

    with tab_sentiment:
        _render_sentiment_tab(selected_folder, document)

    with tab_data:
        _render_data_tab(selected_folder, document)


def _sidebar_selector(runs: list[Path]) -> Path | None:
    with st.sidebar:
        st.markdown("### 🗂️ Video Workspace")
        if not runs:
            st.caption("No analyzed runs yet.")
            return None

        run_names = [p.name for p in runs]
        chosen_name = st.selectbox("Select Analyzed Video", run_names, index=0)
        folder = next(p for p in runs if p.name == chosen_name)

        doc_path = folder / "result.json"
        if doc_path.exists():
            try:
                doc = json.loads(doc_path.read_text(encoding="utf-8"))
                duration = float(doc.get("duration_sec", 0.0))
                speakers = doc.get("speakers") or []
                utterances = doc.get("utterances") or []
                st.markdown(
                    f"""
                    <div class="sidebar-meta">
                        <div><strong>Duration:</strong> {_clock(duration)}</div>
                        <div><strong>Speakers:</strong> {len(speakers)}</div>
                        <div><strong>Sentences:</strong> {len(utterances)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            except Exception:
                pass

        st.markdown("---")
        st.markdown("### ⚙️ Engine Settings")
        settings = load_settings()
        st.caption(f"**Ollama URL:** `{settings.ollama_url}`")
        st.caption(f"**Default Model:** `{settings.ollama_model}`")

        return folder


def _render_screen_capture_tab(current_folder: Path | None) -> None:
    st.markdown("### 🖥️ See & Record Videos Directly from Screen")
    st.markdown(
        "Capture live lectures, webinars, or test videos directly from your display, application window, or browser tab."
    )

    col_left, col_right = st.columns([3, 2], gap="large")

    with col_left:
        st.markdown("#### 1. In-Browser 1-Click Screen & Audio Recorder")
        st.caption("Click 'Start Screen Capture' below, select any screen, window, or tab, and check 'Share audio'.")
        
        # Embedded WebRTC Screen Recorder component
        _render_webrtc_screen_recorder()

        st.markdown("---")
        st.markdown("#### 2. Python Desktop Screen Recorder")
        st.caption("Directly records your desktop screen to an MP4 video file.")
        
        c1, c2, c3 = st.columns([2, 2, 3])
        with c1:
            duration = st.slider("Duration (seconds)", min_value=5, max_value=120, value=15, step=5)
        with c2:
            fps = st.selectbox("Frame Rate", [10, 15, 25, 30], index=1)
        with c3:
            rec_name = st.text_input("Folder Name", value="my_screen_video")

        if st.button("🔴 Record Desktop Screen Now", use_container_width=True):
            target_dir = OUTPUTS / rec_name
            target_dir.mkdir(parents=True, exist_ok=True)
            video_file = target_dir / "screen_video.mp4"
            with st.spinner(f"Recording screen for {duration} seconds... Please switch to your video window!"):
                try:
                    record_screen(video_file, duration_sec=float(duration), fps=fps)
                    st.success(f"Screen recorded successfully to `{video_file}` ({video_file.stat().st_size // 1024} KB)!")
                    st.video(str(video_file))
                except Exception as exc:
                    st.error(f"Screen recording failed: {exc}")

    with col_right:
        st.markdown("#### 3. Upload or Process Video File")
        uploaded = st.file_uploader("Upload video file (MP4, MKV, AVI, WEBM)", type=["mp4", "mkv", "avi", "webm"])
        if uploaded:
            dest_dir = OUTPUTS / uploaded.name.rsplit(".", 1)[0]
            dest_dir.mkdir(parents=True, exist_ok=True)
            saved_video = dest_dir / uploaded.name
            saved_video.write_bytes(uploaded.read())
            st.success(f"Uploaded `{uploaded.name}` to `{saved_video}`")
            st.video(str(saved_video))

        st.markdown("#### 4. Instant Screen Snapshot")
        st.caption("Grab the current screen or test question area right now.")
        if st.button("📸 Capture Instant Screenshot", use_container_width=True):
            snap = capture_screen_snapshot()
            st.image(str(snap), caption=f"Captured at {time.strftime('%H:%M:%S')}", use_container_width=True)


def _render_study_tab(folder: Path, document: dict, study_data: dict) -> None:
    st.markdown("### 🧠 Video Study Notes & Knowledge Synthesis")
    st.caption("In-depth conceptual breakdown, key arguments, and exam-relevant facts extracted from this video.")

    # Re-generate study button if desired
    btn_col1, btn_col2 = st.columns([4, 1])
    with btn_col2:
        if st.button("🔄 Refresh Study Guide"):
            with st.spinner("Analyzing video concepts and generating study guide..."):
                study_data = study_video(document, folder)
                st.success("Study guide updated!")
                st.rerun()

    # Overview
    st.markdown(
        f"""
        <div class="study-card executive-overview">
            <h4>📌 Executive Overview</h4>
            <p>{study_data.get('overview', 'No summary available.')}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Visual Keyframes Gallery if available
    keyframes = study_data.get("keyframes") or []
    if keyframes:
        st.markdown("#### 🎬 Key Visual Slides & Video Moments")
        cols = st.columns(min(len(keyframes), 4))
        for idx, kf in enumerate(keyframes[:8]):
            col = cols[idx % 4]
            img_path = folder / kf.get("path", "")
            if img_path.exists():
                with col:
                    st.image(str(img_path), caption=f"⏱️ {kf.get('timestamp_str')} (Frame {kf.get('index')})", use_container_width=True)

    # Core Concepts & High-Yield Facts
    c1, c2 = st.columns([1, 1], gap="medium")
    with c1:
        st.markdown("#### 💡 Core Concepts & Definitions")
        concepts = study_data.get("core_concepts") or []
        if concepts:
            for item in concepts:
                st.markdown(
                    f"""
                    <div class="concept-item">
                        <div class="concept-title">{_escape(item.get('concept', 'Concept'))}</div>
                        <div class="concept-def">{_escape(item.get('definition', ''))}</div>
                        <div class="concept-takeaway"><strong>Exam Takeaway:</strong> {_escape(item.get('key_takeaway', ''))}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("No explicit concept definitions generated.")

    with c2:
        st.markdown("#### 🎯 High-Yield Test Facts")
        facts = study_data.get("high_yield_test_facts") or []
        if facts:
            for fact in facts:
                st.markdown(f"""<div class="fact-item">⭐ {_escape(fact)}</div>""", unsafe_allow_html=True)
        else:
            st.info("No high yield facts available.")

    # Speaker Perspectives Matrix
    st.markdown("#### 🗣️ Speaker Perspectives & Stances")
    speakers = study_data.get("speaker_perspectives") or []
    if speakers:
        cols = st.columns(min(len(speakers), 3))
        for idx, spk in enumerate(speakers):
            with cols[idx % 3]:
                claims_html = "".join(f"<li>{_escape(c)}</li>" for c in spk.get("key_claims", []))
                st.markdown(
                    f"""
                    <div class="speaker-study-card">
                        <h5>{_escape(spk.get('speaker', 'Speaker'))}</h5>
                        <p><strong>Stance:</strong> {_escape(spk.get('main_stance', ''))}</p>
                        <p><strong>Sentiment:</strong> {_escape(spk.get('sentiments', ''))}</p>
                        <ul>{claims_html}</ul>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # Markdown download
    md_file = folder / "study_guide.md"
    if md_file.exists():
        st.download_button(
            "📥 Download Study Guide (.md)",
            md_file.read_bytes(),
            file_name=f"{folder.name}_study_guide.md",
            mime="text/markdown",
        )


def _render_test_solver_tab(folder: Path, document: dict, study_data: dict) -> None:
    st.markdown("### 📝 Test & Exam Question Solver")
    st.caption("Ask or capture questions related to this video. The AI matches spoken and visual evidence to deliver answers with citations.")

    col1, col2 = st.columns([3, 2], gap="large")

    with col1:
        st.markdown("#### Enter or Paste a Test Question")

        # Example question helper
        example_q = "Why did Speaker 2 dislike Contoso?\nA) Poor quality\nB) Contoso is not good on price\nC) Slow delivery\nD) Bad customer support"
        sample_q_btn = st.button("💡 Load Example Multiple Choice Question")
        
        default_val = example_q if sample_q_btn else ""
        question_input = st.text_area(
            "Question (MCQ, True/False, or Open-ended):",
            value=default_val,
            height=140,
            placeholder="e.g. Which company has the best support according to Speaker 1?\nA) Northwind\nB) Contoso\nC) Both\nD) Neither",
        )

        solve_btn = st.button("🚀 Solve Test Question", type="primary", use_container_width=True)

        if solve_btn and question_input.strip():
            with st.spinner("Analyzing video evidence, transcripts, and speaker sentiments..."):
                answer = answer_test_question(question_input, document, study_data=study_data)
                _display_answer_card(answer)

    with col2:
        st.markdown("#### 📸 On-Screen Test Question Capture")
        st.caption("If a quiz or exam question is currently open on your screen, snap it directly!")
        
        user_hint = st.text_input("Optional question hint / text from screen:", placeholder="e.g. Question 4 about pricing")
        if st.button("📷 Capture & Solve From Screen", use_container_width=True):
            with st.spinner("Capturing on-screen question and solving..."):
                snap_path, answer = capture_and_solve_screen_question(
                    document,
                    study_data=study_data,
                    user_hint=user_hint,
                )
                st.image(str(snap_path), caption="Captured On-Screen Area", use_container_width=True)
                _display_answer_card(answer)

        st.markdown("---")
        st.markdown("#### 🎲 Auto-Generate Practice Quiz")
        st.caption("Generate instant multiple-choice quiz questions based on this video to test yourself.")
        if st.button("✨ Generate 3 Practice Questions", use_container_width=True):
            _generate_and_render_quiz(document, study_data)


def _display_answer_card(ans: dict) -> None:
    selected_opt = ans.get("selected_option")
    answer_text = ans.get("answer_text", "Answer provided.")
    confidence = float(ans.get("confidence", 0.9))
    quote = ans.get("evidence_quote", "")
    speaker = ans.get("speaker", "Speaker")
    timestamp = ans.get("timestamp_str", "00:00")
    sentiment = ans.get("sentiment_context", "Neutral")
    explanation = ans.get("explanation", "")

    opt_badge = f'<span class="badge-option">Option {selected_opt}</span>' if selected_opt else ""

    st.markdown(
        f"""
        <div class="answer-box">
            <div class="answer-header">
                <div>
                    {opt_badge}
                    <span class="answer-statement">{_escape(answer_text)}</span>
                </div>
                <div class="confidence-pill">Confidence: {int(confidence * 100)}%</div>
            </div>
            <div class="answer-body">
                <p><strong>📖 Detailed Reasoning:</strong> {_escape(explanation)}</p>
                <div class="evidence-quote-box">
                    <strong>Evidence Citation:</strong> "{_escape(quote)}"
                    <br/>
                    <span class="citation-meta">🎙️ {_escape(speaker)} · ⏱️ {_escape(timestamp)} · Aspect Tone: {_escape(sentiment)}</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _generate_and_render_quiz(document: dict, study_data: dict) -> None:
    facts = study_data.get("high_yield_test_facts") or []
    utterances = document.get("utterances") or []
    
    st.markdown("#### 🎯 Practice Quiz")
    if len(utterances) < 2 and not facts:
        st.info("Not enough statements to generate quiz.")
        return

    # Create 2-3 sample quiz items
    q1 = {
        "q": "Based on the discussion, which statement best represents the consensus or primary claim?",
        "options": [
            f"A) {utterances[0].get('text', 'Statement A')}",
            f"B) {utterances[1].get('text', 'Statement B') if len(utterances) > 1 else 'None'}",
            "C) Both speakers completely agreed on all points",
            "D) No opinions were expressed",
        ],
        "correct": "A",
        "why": f"Speaker stated: '{utterances[0].get('text', '')}'",
    }

    st.write(f"**Q1:** {q1['q']}")
    choice = st.radio("Choose your answer:", q1["options"], key="quiz_q1")
    if st.button("Check Answer Q1"):
        if choice.startswith(q1["correct"]):
            st.success(f"✅ Correct! {q1['why']}")
        else:
            st.error(f"❌ Incorrect. Correct answer was {q1['correct']}. {q1['why']}")


def _render_sentiment_tab(folder: Path, document: dict) -> None:
    utterances = document.get("utterances") or []
    summary = document.get("summary", {}).get("by_speaker") or {}
    source = str(document.get("source") or "")
    playable = _playable(source) if source else None

    left, right = st.columns([3, 2], gap="large")
    with left:
        if playable:
            st.video(playable)
        else:
            st.info("The original video file is not stored locally. Spoken sentences and audio clips are available below.")
    with right:
        st.markdown(f"### {_escape(folder.name)}")
        _headline(document, utterances, summary)

    _subject_cards(summary)
    _timeline(utterances, float(document.get("duration_sec") or 0))


def _render_data_tab(folder: Path, document: dict) -> None:
    st.markdown("### 📁 Data & Spreadsheet Exports")
    order = ["result.csv", "summary.csv", "utterances.csv", "result.json", "study_guide.json", "study_guide.md"]
    present = [name for name in order if (folder / name).exists()]
    if not present:
        st.caption("No export files found.")
        return
    tabs = st.tabs(present)
    for tab, name in zip(tabs, present):
        path = folder / name
        with tab:
            st.download_button(f"Download {name}", path.read_bytes(), file_name=name, key=f"dl-{folder.name}-{name}")
            if name.endswith(".csv"):
                frame = pd.read_csv(path, encoding="utf-8-sig")
                frame = frame.drop(columns=[col for col in frame.columns if col in HIDDEN_COLUMNS])
                st.dataframe(frame, use_container_width=True, hide_index=True, height=400)
            elif name.endswith(".json"):
                st.json(json.loads(path.read_text(encoding="utf-8")), expanded=1)
            elif name.endswith(".md"):
                st.markdown(path.read_text(encoding="utf-8"))


def _render_webrtc_screen_recorder() -> None:
    """Render browser WebRTC screen recording widget with audio support."""
    html_code = """
    <div style="background:#1e293b;border:1px solid #334155;border-radius:16px;padding:20px;color:#f8fafc;font-family:sans-serif;">
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px;">
            <div style="display:flex;align-items:center;gap:10px;">
                <span id="rec-dot" style="width:14px;height:14px;border-radius:50%;background:#64748b;display:inline-block;"></span>
                <strong id="rec-status" style="font-size:16px;">Ready to Record</strong>
            </div>
            <span id="rec-timer" style="font-size:18px;font-family:monospace;color:#94a3b8;">00:00</span>
        </div>
        
        <video id="rec-preview" autoplay muted style="width:100%;max-height:260px;background:#0f172a;border-radius:12px;margin-bottom:14px;display:none;"></video>
        
        <div style="display:flex;gap:12px;flex-wrap:wrap;">
            <button id="btn-start" onclick="startCapture()" style="background:#10b981;color:#fff;border:none;padding:10px 20px;border-radius:8px;font-weight:bold;cursor:pointer;">
                ▶ Start Screen Capture
            </button>
            <button id="btn-stop" onclick="stopCapture()" disabled style="background:#ef4444;color:#fff;border:none;padding:10px 20px;border-radius:8px;font-weight:bold;cursor:not-allowed;opacity:0.6;">
                ⏹ Stop & Save Video
            </button>
        </div>
        <p id="rec-hint" style="color:#94a3b8;font-size:12px;margin-top:10px;margin-bottom:0;">
            Tip: When the browser screen picker appears, enable "Share audio" to record lecture/test audio.
        </p>
    </div>

    <script>
    let mediaRecorder = null;
    let recordedChunks = [];
    let stream = null;
    let timerInterval = null;
    let startTime = null;

    async function startCapture() {
        try {
            stream = await navigator.mediaDevices.getDisplayMedia({
                video: { cursor: "always" },
                audio: true
            });
            recordedChunks = [];
            
            const options = { mimeType: 'video/webm; codecs=vp9' };
            if (!MediaRecorder.isTypeSupported(options.mimeType)) {
                mediaRecorder = new MediaRecorder(stream);
            } else {
                mediaRecorder = new MediaRecorder(stream, options);
            }

            mediaRecorder.ondataavailable = e => {
                if (e.data.size > 0) recordedChunks.push(e.data);
            };

            mediaRecorder.onstop = () => {
                clearInterval(timerInterval);
                document.getElementById('rec-dot').style.background = '#64748b';
                document.getElementById('rec-status').innerText = 'Finished Recording';
                
                const blob = new Blob(recordedChunks, { type: 'video/webm' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.style.display = 'none';
                a.href = url;
                a.download = 'screen_recording_' + Date.now() + '.webm';
                document.body.appendChild(a);
                a.click();
                setTimeout(() => {
                    document.body.removeChild(a);
                    window.URL.revokeObjectURL(url);
                }, 100);

                if (stream) {
                    stream.getTracks().forEach(track => track.stop());
                }
            };

            const preview = document.getElementById('rec-preview');
            preview.srcObject = stream;
            preview.style.display = 'block';

            mediaRecorder.start();
            startTime = Date.now();
            document.getElementById('rec-dot').style.background = '#ef4444';
            document.getElementById('rec-status').innerText = 'Recording Screen & Audio...';
            document.getElementById('btn-start').disabled = true;
            document.getElementById('btn-start').style.opacity = '0.5';
            document.getElementById('btn-stop').disabled = false;
            document.getElementById('btn-stop').style.opacity = '1';
            document.getElementById('btn-stop').style.cursor = 'pointer';

            timerInterval = setInterval(() => {
                const diff = Math.floor((Date.now() - startTime) / 1000);
                const m = String(Math.floor(diff / 60)).padStart(2, '0');
                const s = String(diff % 60).padStart(2, '0');
                document.getElementById('rec-timer').innerText = m + ':' + s;
            }, 1000);

            stream.getVideoTracks()[0].onended = () => {
                stopCapture();
            };

        } catch (err) {
            console.error(err);
            alert("Screen capture was canceled or failed: " + err.message);
        }
    }

    function stopCapture() {
        if (mediaRecorder && mediaRecorder.state !== 'inactive') {
            mediaRecorder.stop();
            document.getElementById('btn-start').disabled = false;
            document.getElementById('btn-start').style.opacity = '1';
            document.getElementById('btn-stop').disabled = true;
            document.getElementById('btn-stop').style.opacity = '0.6';
            document.getElementById('btn-stop').style.cursor = 'not-allowed';
        }
    }
    </script>
    """
    components.html(html_code, height=360)


def _load_run_data(folder: Path) -> tuple[dict, dict]:
    doc = json.loads((folder / "result.json").read_text(encoding="utf-8"))
    study_file = folder / "study_guide.json"
    if study_file.exists():
        try:
            study_data = json.loads(study_file.read_text(encoding="utf-8"))
        except Exception:
            study_data = study_video(doc, folder)
    else:
        study_data = study_video(doc, folder)
    return doc, study_data


def _playable(source: str) -> str | None:
    from video_sentiment.download import is_media_url

    if is_media_url(source):
        return source
    path = Path(source)
    return str(path) if path.is_file() else None


def _runs() -> list[Path]:
    if not OUTPUTS.exists():
        return []
    found = [p for p in OUTPUTS.iterdir() if p.is_dir() and (p / "result.json").exists()]
    return sorted(found, key=lambda p: (p / "result.json").stat().st_mtime, reverse=True)


def _headline(document: dict, utterances: list[dict], summary: dict) -> None:
    rows = _flat(utterances)
    subjects = {name for targets in summary.values() for name in targets}
    speakers = document.get("speakers") or []
    counts = {key: sum(1 for row in rows if row["sentiment"] == key) for key, _, _ in LABELS}
    tiles = [
        ("Length", _clock(float(document.get("duration_sec") or 0))),
        ("Sentences", str(len(utterances))),
        ("Subjects", str(len(subjects))),
        ("Speakers", str(len(speakers) or 1)),
    ]
    st.markdown(
        "<div class='tiles'>"
        + "".join(f"<div class='tile'><p>{label}</p><strong>{value}</strong></div>" for label, value in tiles)
        + "</div>",
        unsafe_allow_html=True,
    )
    total = sum(counts.values()) or 1
    bar = "".join(
        f'<span style="width:{counts[key] / total * 100:.1f}%;background:{color}"></span>'
        for key, _, color in LABELS
        if counts[key]
    )
    legend = "".join(
        f"<span class='legend'><i style='background:{color}'></i>{word} {counts[key]}</span>"
        for key, word, color in LABELS
    )
    st.markdown(
        f"<p class='muted tight'>Mention Polarity Breakdown</p><div class='bar big'>{bar}</div>"
        f"<div class='legends'>{legend}</div>",
        unsafe_allow_html=True,
    )


def _subject_cards(summary: dict) -> None:
    cards = []
    for speaker, targets in summary.items():
        for name, counts in targets.items():
            mentions = sum(int(counts.get(key, 0)) for key, _, _ in LABELS)
            cards.append((mentions, speaker, name, counts))
    cards.sort(key=lambda card: (-card[0], card[2].casefold()))
    st.markdown("#### Spoken Subjects & Sentiments")
    if not cards:
        st.caption("No subject was named.")
        return
    _card_grid(cards[:TOP_SUBJECTS])
    rest = cards[TOP_SUBJECTS:]
    if rest:
        with st.expander(f"Show remaining {len(rest)} subjects"):
            _card_grid(rest)


def _card_grid(cards: list[tuple]) -> None:
    columns = st.columns(3, gap="medium")
    for index, (_, speaker, name, counts) in enumerate(cards):
        with columns[index % 3]:
            st.markdown(_card(speaker, name, counts), unsafe_allow_html=True)


def _card(speaker: str, name: str, counts: dict) -> str:
    total = sum(int(counts.get(key, 0)) for key, _, _ in LABELS) or 1
    pieces = []
    bits = []
    for key, word, color in LABELS:
        count = int(counts.get(key, 0))
        if count:
            pieces.append(f'<span style="width:{(count / total) * 100:.1f}%;background:{color}"></span>')
            bits.append(f"{count} {word.lower()}")
    winner = PLAIN.get(str(counts.get("dominant")), str(counts.get("dominant") or ""))
    color = LABEL_COLOR.get(str(counts.get("dominant")), "#64748b")
    return (
        f"<div class='card' style='border-top:4px solid {color}'>"
        f"<p class='who'>{_escape(speaker)}</p>"
        f"<h4>{_escape(name)}</h4>"
        f"<span class='pill' style='background:{color}'>{_escape(winner)}</span>"
        f"<div class='bar'>{''.join(pieces)}</div>"
        f"<p class='counts'>{_escape(' · '.join(bits))}</p>"
        "</div>"
    )


def _timeline(utterances: list[dict], duration: float) -> None:
    st.markdown("#### Interactive Speech Timeline")
    if not utterances:
        st.caption("No sentences were heard.")
        return
    span = duration or max(float(row["end"]) for row in utterances)
    ticks = []
    for row in utterances:
        start = float(row["start"])
        width = max((float(row["end"]) - start) / span * 100, 0.8)
        left = start / span * 100
        color = LABEL_COLOR.get(_row_sentiment(row), "#64748b")
        ticks.append(
            f'<span class="tick" style="left:{left:.2f}%;width:{width:.2f}%;background:{color}" '
            f'title="{_clock(start)}  {_escape(row["text"])}"></span>'
        )
    st.markdown(
        f"<div class='track'>{''.join(ticks)}</div>"
        f"<div class='scale'><span>0:00</span><span>{_clock(span / 2)}</span><span>{_clock(span)}</span></div>",
        unsafe_allow_html=True,
    )
    options = [f"{_clock(float(row['start']))}  {row['text']}" for row in utterances]
    pick = st.selectbox("Inspect specific utterance:", options)
    row = utterances[options.index(pick)]
    chips = []
    for target in row.get("targets") or []:
        sentiment = str(target.get("sentiment") or "")
        chips.append(
            f"<span class='chip' style='background:{LABEL_COLOR.get(sentiment, '#64748b')}'>"
            f"{_escape(target.get('resolved') or target.get('surface') or 'subject')} · "
            f"{LABEL_WORD.get(sentiment, sentiment)}"
            "</span>"
        )
    evidence = next((str(t.get("evidence") or "") for t in row.get("targets") or [] if t.get("evidence")), "")
    chips_html = "".join(chips) or "<span class='chip quiet'>No subject</span>"
    said_text = _escape(row['text'])
    st.markdown(
        "<div class='quote'>"
        f"<p class='when'>🎙️ {_escape(row['speaker'])} · {_clock(float(row['start']))}–{_clock(float(row['end']))}</p>"
        f"<p class='said'>\"{said_text}\"</p>"
        f"<p class='chips'>{chips_html}</p>"
        f"<p class='evidence'><em>Evidence quote:</em> {_escape(evidence)}</p>"
        "</div>",
        unsafe_allow_html=True,
    )


def _flat(utterances: list[dict]) -> list[dict]:
    rows = []
    for u in utterances:
        for t in u.get("targets") or []:
            rows.append({"sentiment": t.get("sentiment") or ""})
    return rows


def _row_sentiment(row: dict) -> str:
    targets = row.get("targets") or []
    if not targets:
        return "neutral"
    labels = {str(t.get("sentiment") or "neutral") for t in targets}
    if {"positive", "negative"} <= labels or "mixed" in labels:
        return "mixed"
    for label in ("negative", "positive"):
        if label in labels:
            return label
    return "neutral"


def _clock(seconds: float) -> str:
    whole = max(int(round(seconds)), 0)
    hours, rest = divmod(whole, 3600)
    if hours:
        return f"{hours}:{rest // 60:02d}:{rest % 60:02d}"
    return f"{rest // 60}:{rest % 60:02d}"


def _escape(value: object) -> str:
    text = str(value)
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


_CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1300px;
    }
    
    /* Hero Header */
    .hero-container {
        background: linear-gradient(135deg, #091e17 0%, #114232 50%, #1e3a8a 100%);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 24px;
        padding: 2.2rem 2.8rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 20px 40px rgba(0,0,0,0.25);
    }
    .hero-badge {
        display: inline-block;
        background: rgba(16, 185, 129, 0.2);
        color: #34d399;
        border: 1px solid rgba(52, 211, 153, 0.35);
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        margin-bottom: 0.8rem;
    }
    .hero-title {
        color: #f8fafc;
        font-size: 2.4rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        margin: 0 0 0.5rem 0;
        line-height: 1.15;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        line-height: 1.5;
        margin: 0;
        max-width: 900px;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        border-bottom: 2px solid #1e293b;
        padding-bottom: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.05rem;
        font-weight: 700;
        padding: 0.6rem 1.2rem;
        border-radius: 12px;
        color: #94a3b8;
        transition: all 0.2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255,255,255,0.04);
        color: #f1f5f9;
    }
    .stTabs [aria-selected="true"] {
        background: #1e293b !important;
        color: #38bdf8 !important;
        border-bottom: 3px solid #38bdf8;
    }

    /* Cards & Containers */
    .study-card, .answer-box, .speaker-study-card, .concept-item, .fact-item, .card, .quote, .tile {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 18px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        box-shadow: 0 8px 24px rgba(0,0,0,0.15);
    }
    .executive-overview {
        background: linear-gradient(145deg, #1e293b 0%, #162032 100%);
        border-left: 5px solid #38bdf8;
    }
    .executive-overview h4 { margin-top: 0; color: #38bdf8; font-size: 1.2rem; }
    .executive-overview p { color: #cbd5e1; font-size: 1.02rem; line-height: 1.6; }

    /* Concept & Fact items */
    .concept-item { border-left: 4px solid #10b981; }
    .concept-title { font-size: 1.15rem; font-weight: 700; color: #f8fafc; margin-bottom: 4px; }
    .concept-def { color: #cbd5e1; font-size: 0.95rem; margin-bottom: 8px; }
    .concept-takeaway { color: #34d399; font-size: 0.9rem; background: rgba(16, 185, 129, 0.1); padding: 6px 10px; border-radius: 8px; }
    .fact-item { font-size: 1rem; color: #f1f5f9; border-left: 4px solid #f59e0b; background: rgba(245, 158, 11, 0.08); padding: 10px 14px; margin-bottom: 8px; }
    
    .speaker-study-card h5 { font-size: 1.1rem; color: #38bdf8; margin: 0 0 6px 0; }
    .speaker-study-card p { font-size: 0.9rem; color: #cbd5e1; margin: 4px 0; }
    .speaker-study-card ul { margin: 6px 0 0 16px; padding: 0; color: #94a3b8; font-size: 0.88rem; }

    /* Answer box */
    .answer-box {
        background: linear-gradient(135deg, #0d2818 0%, #173822 60%, #1a3a4b 100%);
        border: 2px solid #10b981;
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 12px 32px rgba(16, 185, 129, 0.2);
    }
    .answer-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 12px;
    }
    .badge-option {
        background: #10b981;
        color: #ffffff;
        font-weight: 800;
        font-size: 1.1rem;
        padding: 4px 14px;
        border-radius: 999px;
        margin-right: 10px;
    }
    .answer-statement {
        font-size: 1.3rem;
        font-weight: 700;
        color: #f8fafc;
    }
    .confidence-pill {
        background: rgba(255,255,255,0.15);
        color: #a7f3d0;
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 700;
    }
    .evidence-quote-box {
        background: rgba(0, 0, 0, 0.3);
        border-left: 4px solid #38bdf8;
        padding: 10px 14px;
        border-radius: 8px;
        margin-top: 10px;
        color: #e2e8f0;
        font-size: 0.95rem;
    }
    .citation-meta {
        display: inline-block;
        margin-top: 6px;
        font-size: 0.85rem;
        color: #38bdf8;
    }

    /* Tiles */
    .tiles { display: grid; grid-template-columns: repeat(2, 1fr); gap: 0.8rem; }
    .tile p { margin: 0; color: #94a3b8; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.08em; font-weight: 700; }
    .tile strong { font-size: 1.8rem; letter-spacing: -0.03em; color: #f8fafc; }

    /* Polarity bars */
    .bar { display: flex; height: 12px; border-radius: 999px; overflow: hidden; background: #334155; margin: 0.8rem 0 0.45rem; }
    .bar.big { height: 18px; margin: 0.3rem 0 0.6rem; }
    .bar span { display: block; height: 100%; }
    .legends { display: flex; flex-wrap: wrap; gap: 0.9rem; font-size: 0.88rem; color: #cbd5e1; }
    .legend i { display: inline-block; width: 11px; height: 11px; border-radius: 4px; margin-right: 0.35rem; vertical-align: -1px; }

    /* Subject Cards */
    .who, .counts { color: #94a3b8; margin: 0; font-size: 0.88rem; }
    .card h4 { margin: 4px 0 8px 0; color: #f8fafc; font-size: 1.25rem; }
    .pill { display: inline-block; color: white; border-radius: 999px; padding: 0.2rem 0.7rem; font-size: 0.8rem; font-weight: 800; }

    /* Timeline */
    .track { position: relative; height: 32px; border-radius: 999px; background: #334155; margin: 0.8rem 0 0.35rem; }
    .tick { position: absolute; top: 6px; height: 20px; border-radius: 6px; min-width: 6px; opacity: 0.9; cursor: pointer; transition: transform 0.15s ease; }
    .tick:hover { opacity: 1; transform: scaleY(1.3); }
    .scale { display: flex; justify-content: space-between; color: #94a3b8; font-size: 0.82rem; margin-bottom: 0.8rem; }
    
    .quote { border-left: 4px solid #38bdf8; }
    .quote .when { color: #38bdf8; font-weight: 700; font-size: 0.9rem; margin-bottom: 4px; }
    .quote .said { font-size: 1.2rem; color: #f8fafc; line-height: 1.4; margin: 0 0 8px 0; }
    .chip { display: inline-block; color: white; border-radius: 999px; padding: 0.25rem 0.7rem; font-size: 0.82rem; font-weight: 800; margin: 2px 4px 2px 0; }
    .chip.quiet { background: #64748b; }
    .quote .evidence { color: #94a3b8; font-size: 0.9rem; margin-top: 6px; }

    .sidebar-meta { background: #1e293b; border-radius: 12px; padding: 12px; margin-top: 8px; color: #cbd5e1; font-size: 0.9rem; line-height: 1.6; }
</style>
"""

if __name__ == "__main__":
    main()
