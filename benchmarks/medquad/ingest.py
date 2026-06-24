"""Parse the MedQuAD dataset into a compact retrieval benchmark.

MedQuAD (https://github.com/abachaa/MedQuAD) is a collection of ~47k real
medical question/answer pairs gathered from 12 NIH/NCI/CDC sources. Each XML
``Document`` describes one topic (``Focus``) and holds several ``QAPair`` items,
each a real ``Question`` and an authoritative ``Answer``.

We turn that into a standard information-retrieval benchmark:

* **corpus.jsonl** -- one retrievable passage per QAPair (the answer text).
* **questions.jsonl** -- one query per QAPair (the question), whose gold target
  is the id of its own answer passage.

A retriever is then scored on whether it ranks the correct answer passage at the
top for each real question, among all the other passages as distractors.

We deliberately do **not** commit the parsed corpus (licensing + size); this
script regenerates it locally. Run::

    python -m benchmarks.medquad.ingest --source <path-to-MedQuAD-clone>

If ``--source`` is omitted, the MedQuAD repo is shallow-cloned automatically.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
DEFAULT_RAW = DATA_DIR / "medquad_raw"
MEDQUAD_REPO = "https://github.com/abachaa/MedQuAD"


def _clean(text: str | None) -> str:
    return " ".join(text.split()) if text else ""


def iter_qapairs(source: Path):
    """Yield (qid, source_name, focus, qtype, question, answer) for every QAPair."""
    for xml_path in sorted(source.rglob("*.xml")):
        try:
            root = ElementTree.parse(xml_path).getroot()
        except ElementTree.ParseError:
            continue
        focus = _clean(root.findtext("Focus"))
        src = root.get("source", xml_path.parent.name)
        for pair in root.iterfind(".//QAPair"):
            question_el = pair.find("Question")
            answer = _clean(pair.findtext("Answer"))
            if question_el is None:
                continue
            question = _clean(question_el.text)
            qid = question_el.get("qid") or f"{xml_path.stem}-{pair.get('pid', '?')}"
            qtype = question_el.get("qtype", "")
            yield qid, src, focus, qtype, question, answer


def ensure_source(source: Path | None) -> Path:
    if source is not None:
        if not source.exists():
            sys.exit(f"--source path does not exist: {source}")
        return source
    if DEFAULT_RAW.exists():
        return DEFAULT_RAW
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Cloning MedQuAD into {DEFAULT_RAW} ...")
    subprocess.run(
        ["git", "clone", "--depth", "1", MEDQUAD_REPO, str(DEFAULT_RAW)], check=True
    )
    return DEFAULT_RAW


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=None, help="MedQuAD XML root")
    parser.add_argument("--out", type=Path, default=DATA_DIR, help="output directory")
    parser.add_argument("--limit", type=int, default=1500, help="number of passages to sample")
    parser.add_argument("--min-answer-chars", type=int, default=120)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    source = ensure_source(args.source)
    args.out.mkdir(parents=True, exist_ok=True)

    seen_ids: set[str] = set()
    records: list[dict] = []
    total = 0
    for qid, src, focus, qtype, question, answer in iter_qapairs(source):
        total += 1
        if qid in seen_ids or len(answer) < args.min_answer_chars or len(question) < 8:
            continue
        seen_ids.add(qid)
        records.append(
            {"id": qid, "source": src, "focus": focus, "qtype": qtype,
             "question": question, "answer": answer}
        )

    random.Random(args.seed).shuffle(records)
    sample = records[: args.limit]

    corpus_path = args.out / "corpus.jsonl"
    questions_path = args.out / "questions.jsonl"
    with corpus_path.open("w", encoding="utf-8") as cf, \
            questions_path.open("w", encoding="utf-8") as qf:
        for rec in sample:
            cf.write(json.dumps(
                {"id": rec["id"], "source": rec["source"], "focus": rec["focus"],
                 "text": rec["answer"]}) + "\n")
            qf.write(json.dumps(
                {"qid": rec["id"], "question": rec["question"],
                 "qtype": rec["qtype"], "gold_id": rec["id"]}) + "\n")

    by_source: dict[str, int] = {}
    for rec in sample:
        by_source[rec["source"]] = by_source.get(rec["source"], 0) + 1

    print(f"Scanned {total} QAPairs; {len(records)} had usable answers.")
    print(f"Wrote {len(sample)} passages -> {corpus_path}")
    print(f"Wrote {len(sample)} questions -> {questions_path}")
    print("By source:", dict(sorted(by_source.items(), key=lambda kv: -kv[1])))


if __name__ == "__main__":
    main()
