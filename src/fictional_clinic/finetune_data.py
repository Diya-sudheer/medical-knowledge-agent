from __future__ import annotations

import argparse
import json
from pathlib import Path

from fictional_clinic.models import Role
from fictional_clinic.rag import LocalRetriever
from fictional_clinic.responder import LocalTemplateEngine


QUESTIONS = [
    "What should I know about Luma Cough Syndrome?",
    "How does the clinic handle Amber Fever follow-up?",
    "What is the referral path for the Moonleaf Sleep Clinic?",
    "When is a Glowstone allergy visit escalated?",
]


def make_example(role: Role, question: str) -> dict:
    retriever = LocalRetriever()
    sources = retriever.search(question)
    answer = LocalTemplateEngine().answer(role, question, sources)
    return {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a role-aware assistant for a fictional clinic learning project. "
                    "Use retrieved fictional context and follow the requested role format."
                ),
            },
            {"role": "user", "content": f"Role: {role.value}\nQuestion: {question}"},
            {"role": "assistant", "content": answer},
        ]
    }


def build_examples() -> list[dict]:
    return [make_example(role, question) for question in QUESTIONS for role in Role]


def write_jsonl(output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for example in build_examples():
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic fine-tuning JSONL.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/generated/role_examples.jsonl"),
        help="Output JSONL path.",
    )
    args = parser.parse_args()
    write_jsonl(args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

