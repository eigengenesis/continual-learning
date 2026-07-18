import torch

import qwen_continual_proof as qp


class BoundaryTokenizer:
    pad_token_id = 0
    eos_token_id = 99

    def __call__(self, text, **kwargs):
        del kwargs
        if isinstance(text, list):
            raise AssertionError("test tokenizer expects row-wise tokenization")
        ids = []
        if "Question" in text:
            ids.extend([7, 8, 9])
        if "Answer: " in text:
            ids.append(10)
        elif "Answer:" in text:
            ids.append(11)
        if text.startswith("A") or text.endswith("A<eos>"):
            ids.append(20)
        if "<eos>" in text:
            ids.append(99)
        return {"input_ids": ids}


def test_supervised_batch_preserves_first_target_token_after_prompt_boundary():
    batch = qp._prepare_supervised_batch(
        BoundaryTokenizer(),
        ["Question text\nAnswer: "],
        ["A<eos>"],
        "cpu",
        max_length=8,
    )

    labels = batch["labels"][0]
    supervised = labels[labels != -100]
    assert supervised.tolist() == [20, 99]


def test_supervised_batch_truncates_prompt_before_target():
    batch = qp._prepare_supervised_batch(
        BoundaryTokenizer(),
        ["Question text\nAnswer: "],
        ["A<eos>"],
        "cpu",
        max_length=2,
    )

    assert batch["input_ids"].tolist() == [[20, 99]]
    assert torch.equal(batch["labels"], batch["input_ids"])
