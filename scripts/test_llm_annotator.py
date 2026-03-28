"""
test_llm_annotator.py
---------------------
Quick smoke test for LLMAnnotator with Gemini 2.0 Flash.

Usage:
    python scripts/test_llm_annotator.py --api_key YOUR_KEY_HERE
    # or set env var:
    $env:GEMINI_API_KEY = "YOUR_KEY_HERE"
    python scripts/test_llm_annotator.py

Get a free API key at: https://aistudio.google.com
"""

import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_path", type=str, default=".")
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument(
        "--backend", type=str, default="groq",
        choices=["groq", "gemini", "openai"],
    )
    parser.add_argument(
        "--mock", action="store_true",
        help="Skip real API calls; use fake scores to test code logic only.",
    )
    args = parser.parse_args()

    project_path = os.path.abspath(args.project_path)
    sys.path.insert(0, os.path.join(project_path, "src"))

    from pipeline.llm_runner import LLMAnnotator, TASK_CONTEXT, build_prompt, parse_llm_response

    if args.mock:
        print("[MOCK MODE] Skipping real API calls.\n")
    else:
        annotator = LLMAnnotator(backend=args.backend, api_key=args.api_key)
        print(f"Backend: {annotator.backend} | Model: {annotator.model}\n")

    # One test sample per dataset
    test_cases = [
        ("told_br",       "Esse negro macumbeiro não merece nada mesmo, lixo da sociedade."),
        ("told_br",       "Hoje foi um dia cansativo mas consegui terminar o projeto."),
        ("20_newsgroups", "Can anyone recommend a good ray tracer for rendering 3D scenes?"),
        ("20_newsgroups", "The senator voted against the bill yesterday in a close 52-48 decision."),
        ("pt_tweets",     "Que dia lindo! Fui passear no parque e amei cada momento. 😊"),
        ("pt_tweets",     "Perdi meu emprego hoje. Tô arrasado, não sei o que fazer."),
    ]

    print(f"{'Dataset':<18} {'LLM Score':<12} {'Label':<8} Reason")
    print("-" * 90)
    for dataset, text in test_cases:
        if args.mock:
            # Fake: inject synthetic JSON and test parse_llm_response
            import json
            fake_score = 0.9 if "lixo" in text or "macumbeiro" in text or "arrasado" in text else 0.1
            raw = json.dumps({"anomaly_score": fake_score, "reason": "[MOCK] Synthetic label for testing."})
            result = parse_llm_response(raw)
            result["parse_error"] = result["anomaly_score"] is None
            # Also verify build_prompt works for this dataset
            prompt = build_prompt(text, dataset)
            assert len(prompt) > 50, "build_prompt returned empty string"
        else:
            result = annotator.annotate(text, dataset)
        score = result["anomaly_score"]
        label = "ANOMALY" if (score is not None and score >= 0.6) else "normal"
        score_str = f"{score:.2f}" if score is not None else "ERROR"
        short_text = text[:40] + "..." if len(text) > 40 else text
        print(f"{dataset:<18} {score_str:<12} {label:<8} {result['reason'][:60]}")
        print(f"  Text: {short_text}\n")

    if args.mock:
        print("All mock tests passed. build_prompt and parse_llm_response work correctly.")


if __name__ == "__main__":
    main()
