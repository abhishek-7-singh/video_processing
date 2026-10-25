"""Unit tests for test question answering engine."""

from video_sentiment.test_solver import answer_test_question, _heuristic_solver, _rank_relevant_segments


def test_rank_relevant_segments():
    utterances = [
        {"speaker": "Speaker 1", "text": "We are evaluating Airbus versus Boeing safety records.", "targets": []},
        {"speaker": "Speaker 2", "text": "The price of tea in China has increased.", "targets": []},
    ]
    ranked = _rank_relevant_segments("What about Airbus safety?", utterances)
    assert ranked[0]["speaker"] == "Speaker 1"


def test_heuristic_solver_mcq():
    question = (
        "What was the conclusion regarding Contoso price?\n"
        "A) Contoso is affordable\n"
        "B) Contoso is not good on price\n"
        "C) It was never mentioned"
    )
    ranked = [
        {
            "speaker": "Speaker 2",
            "start": 5.0,
            "end": 8.0,
            "text": "Contoso is not good on price.",
            "targets": [{"surface": "Contoso", "sentiment": "negative", "resolved": "Contoso"}],
        }
    ]
    answer = _heuristic_solver(question, ranked)
    assert answer["question_type"] == "mcq"
    assert answer["selected_option"] == "B"
    assert "Contoso is not good on price" in answer["answer_text"]
    assert answer["speaker"] == "Speaker 2"
    assert answer["confidence"] > 0.8


def test_answer_test_question_sample():
    doc = {
        "source": "sample",
        "duration_sec": 15.0,
        "speakers": ["Speaker 1"],
        "utterances": [
            {
                "speaker": "Speaker 1",
                "start": 0.0,
                "end": 4.0,
                "text": "Northwind provides the best technical support.",
                "targets": [{"surface": "Northwind", "sentiment": "positive", "evidence": "best technical support"}],
            }
        ],
        "summary": {},
    }
    q = "Who provides the best technical support?\nA) Northwind\nB) Contoso"
    ans = answer_test_question(q, doc)
    assert ans["selected_option"] == "A" or "Northwind" in ans["answer_text"]
