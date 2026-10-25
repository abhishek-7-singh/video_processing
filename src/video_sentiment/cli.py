from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from video_sentiment import __version__
from video_sentiment.config import load_settings
from video_sentiment.errors import VideoSentimentError
from video_sentiment.download import default_output_dir, download_audio, is_media_url, safe_folder_name
from video_sentiment.pipeline import analyze_video, score_transcript
from video_sentiment.sentiment import ollama_models


def main(argv: list[str] | None = None) -> None:
    raise SystemExit(run(argv))


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="video-sentiment",
        description="Score speaker sentiment, study videos from screen, and answer test questions.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    analyze = sub.add_parser("analyze", help="Run the full pipeline on a video file or a link.")
    analyze.add_argument("video", help="A video file path, or an http(s) link such as YouTube.")
    analyze.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Folder for results. Defaults to outputs/<YouTube title or video file name>.",
    )
    analyze.add_argument("--config", type=Path)
    analyze.add_argument("--skip-tone", action="store_true")
    analyze.add_argument("--single-speaker", action="store_true")
    analyze.add_argument("--min-speakers", type=int)
    analyze.add_argument("--max-speakers", type=int)

    score = sub.add_parser("score-text", help="Score a JSON transcript. No audio or tone.")
    score.add_argument("transcript", type=Path)
    score.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Folder for results. Defaults to outputs/<transcript file name>.",
    )
    score.add_argument("--config", type=Path)

    # New: record-screen
    rec = sub.add_parser("record-screen", help="Record video directly from screen.")
    rec.add_argument("-d", "--duration", type=float, default=15.0, help="Duration in seconds (default: 15s).")
    rec.add_argument("-o", "--output", type=Path, help="Target video file path (e.g. outputs/screen_rec.mp4).")
    rec.add_argument("--fps", type=int, default=15, help="Frame rate (default: 15).")
    rec.add_argument("--analyze", action="store_true", help="Automatically analyze and study the recorded video.")
    rec.add_argument("--config", type=Path)

    # New: study
    std = sub.add_parser("study", help="Extract keyframes and generate an in-depth study guide.")
    std.add_argument("result", type=Path, help="Path to result.json or analysis folder.")
    std.add_argument("-o", "--output", type=Path, help="Folder to save study guide.")
    std.add_argument("--video", type=Path, help="Path to video file for keyframe extraction.")
    std.add_argument("--config", type=Path)

    # New: answer-test
    ans = sub.add_parser("answer-test", help="Answer test questions based on video study context.")
    ans.add_argument("result", type=Path, help="Path to result.json or analysis folder.")
    ans.add_argument("-q", "--question", help="Test question text (MCQ or open-ended).")
    ans.add_argument("-f", "--file", type=Path, help="File containing test questions.")
    ans.add_argument("--screen", action="store_true", help="Capture test question directly from the screen.")
    ans.add_argument("--config", type=Path)

    check = sub.add_parser("check", help="Show whether local models and tools are ready.")
    check.add_argument("--config", type=Path)

    args = parser.parse_args(argv)
    try:
        settings = load_settings(args.config or _local_config())
        if args.command == "analyze":
            if args.min_speakers is not None:
                settings.min_speakers = args.min_speakers
            if args.max_speakers is not None:
                settings.max_speakers = args.max_speakers
            source = str(args.video).strip()
            source_label = None
            media = Path(source)
            output = args.output or default_output_dir(source)
            print(f"Saving results to {output}", flush=True)
            if is_media_url(source):
                print(f"Downloading {source}", flush=True)
                media = download_audio(source, output)
                source_label = source
            doc = analyze_video(
                media,
                output,
                settings,
                skip_tone=args.skip_tone,
                single_speaker=args.single_speaker,
                source_label=source_label,
            )
            # Automatically generate study guide
            from video_sentiment.study import study_video
            print("Generating study guide & extracting keyframes...", flush=True)
            study_video(doc, output, video_path=media if media.is_file() else None, settings=settings)
        elif args.command == "score-text":
            output = args.output or (Path("outputs") / safe_folder_name(args.transcript.stem))
            print(f"Saving results to {output}", flush=True)
            doc = score_transcript(args.transcript, output, settings)
            from video_sentiment.study import study_video
            study_video(doc, output, settings=settings)
        elif args.command == "record-screen":
            from video_sentiment.screen import record_screen
            out_file = args.output or (Path("outputs") / "screen_recording" / "screen_video.mp4")
            print(f"Recording screen for {args.duration}s to {out_file}...", flush=True)
            record_screen(out_file, duration_sec=args.duration, fps=args.fps)
            print(f"Screen recorded successfully: {out_file}", flush=True)
            if args.analyze:
                analysis_dir = out_file.parent
                analyze_video(out_file, analysis_dir, settings)
                from video_sentiment.study import study_video
                study_video(json.loads((analysis_dir / "result.json").read_text(encoding="utf-8")), analysis_dir, video_path=out_file, settings=settings)
        elif args.command == "study":
            from video_sentiment.study import study_video
            res_path = args.result if args.result.is_file() else (args.result / "result.json")
            doc = json.loads(res_path.read_text(encoding="utf-8"))
            out_dir = args.output or res_path.parent
            study_video(doc, out_dir, video_path=args.video, settings=settings)
            print(f"Study guide saved to {out_dir / 'study_guide.md'}", flush=True)
        elif args.command == "answer-test":
            from video_sentiment.test_solver import answer_test_question, capture_and_solve_screen_question
            res_path = args.result if args.result.is_file() else (args.result / "result.json")
            doc = json.loads(res_path.read_text(encoding="utf-8"))
            study_path = res_path.parent / "study_guide.json"
            study_data = json.loads(study_path.read_text(encoding="utf-8")) if study_path.exists() else None

            if args.screen:
                print("Capturing question from screen snapshot...", flush=True)
                snap_path, answer = capture_and_solve_screen_question(doc, study_data=study_data, settings=settings)
                print(f"Screenshot saved to: {snap_path}")
                print(json.dumps(answer, indent=2))
            elif args.question:
                answer = answer_test_question(args.question, doc, study_data=study_data, settings=settings)
                print(json.dumps(answer, indent=2))
            elif args.file:
                questions = [q.strip() for q in args.file.read_text(encoding="utf-8").split("\n\n") if q.strip()]
                for idx, q in enumerate(questions, 1):
                    print(f"\n--- Question {idx} ---")
                    print(q)
                    ans = answer_test_question(q, doc, study_data=study_data, settings=settings)
                    print(f"Answer: {ans.get('selected_option') or ''} - {ans.get('answer_text')}")
                    print(f"Citation: [{ans.get('timestamp_str')}] {ans.get('speaker')}: \"{ans.get('evidence_quote')}\"")
                    print(f"Reasoning: {ans.get('explanation')}")
            else:
                print("Provide --question, --file, or --screen to answer test questions.", file=sys.stderr)
                return 1
        else:
            return check_environment(settings)
    except VideoSentimentError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def check_environment(settings) -> int:
    problems: list[str] = []
    print(f"python {sys.version.split()[0]}")
    try:
        from video_sentiment.audio import find_ffmpeg

        print(f"ffmpeg {find_ffmpeg()}")
    except VideoSentimentError as exc:
        problems.append(str(exc))
        print(f"ffmpeg missing: {exc}")

    try:
        import torch

        print(f"torch {torch.__version__} cuda={torch.cuda.is_available()}")
    except ImportError:
        problems.append("torch is not installed")
        print("torch is not installed")

    try:
        import whisperx

        print(f"whisperx {getattr(whisperx, '__version__', 'installed')}")
    except ImportError:
        problems.append("whisperx is not installed")
        print("whisperx is not installed")

    try:
        import transformers

        print(f"transformers {transformers.__version__}")
    except ImportError:
        problems.append("transformers is not installed")
        print("transformers is not installed")

    if settings.hf_token:
        print("HF_TOKEN is set")
    else:
        problems.append("HF_TOKEN is not set")
        print("HF_TOKEN is not set")

    try:
        models = ollama_models(settings.ollama_url)
        print("ollama models: " + (", ".join(models) if models else "(none)"))
        wanted = settings.ollama_model
        if not any(name == wanted or name.startswith(wanted + ":") for name in models):
            problems.append(f"Ollama model {wanted} is not pulled")
            print(f"missing model {wanted}")
    except VideoSentimentError as exc:
        problems.append(str(exc))
        print(str(exc))

    if problems:
        print(f"{len(problems)} check(s) still needed")
        return 1
    print("ready")
    return 0


def _local_config() -> Path | None:
    candidate = Path("config.yaml")
    if candidate.exists():
        return candidate
    return None


if __name__ == "__main__":
    raise SystemExit(main())
