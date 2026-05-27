import json

from fictional_clinic.finetune_data import write_jsonl


def test_fine_tuning_data_generator_writes_jsonl(tmp_path):
    output = tmp_path / "examples.jsonl"

    write_jsonl(output)

    lines = output.read_text(encoding="utf-8").splitlines()
    assert lines
    first = json.loads(lines[0])
    assert "messages" in first
    assert first["messages"][0]["role"] == "system"
    assert first["messages"][-1]["role"] == "assistant"

