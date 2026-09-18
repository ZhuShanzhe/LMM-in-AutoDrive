from types import SimpleNamespace

import pytest
import torch

from control.generic_instruction_fsm import GenericInstructionFSM, ParsedInstruction


class FakeParser:
    def __init__(self):
        self.calls = []
        self.fail = False

    def parse_text(self, text, **kwargs):
        self.calls.append(text)
        if self.fail:
            raise RuntimeError("transient failure")
        return {"parse_result": {"status": "VALID", "text": text}}


def test_same_command_id_with_new_text_does_not_reuse_old_parse():
    parser = FakeParser()
    fsm = GenericInstructionFSM(parser=parser, cache_capacity=2)
    a = fsm._parser_result("keep lane", {"id": "voice"})
    b = fsm._parser_result("stop", {"id": "voice"})
    assert a != b
    assert len(parser.calls) == 2
    assert fsm._parser_result("stop", {"id": "voice"}) == b
    assert len(parser.calls) == 2


def test_parse_cache_lru_failure_retry_and_clear():
    parser = FakeParser()
    fsm = GenericInstructionFSM(parser=parser, cache_capacity=2)
    for text in ("a", "b", "a", "c"):
        fsm._parser_result(text, {})
    assert len(fsm._parse_cache) == 2
    assert ("", "b") not in fsm._parse_cache
    parser.fail = True
    assert fsm._parser_result("retry", {}) == {}
    parser.fail = False
    assert fsm._parser_result("retry", {})["status"] == "VALID"
    fsm.clear_caches()
    assert not fsm._parse_cache and not fsm._token_cache
    with pytest.raises(ValueError):
        GenericInstructionFSM(cache_capacity=0)


def test_token_cache_includes_actual_text_even_with_explicit_key():
    calls = []

    def tokenize(text, **kwargs):
        calls.append(text)
        return {"input_ids": torch.tensor([[len(text)]]),
                "attention_mask": torch.ones(1, 1)}

    inner = SimpleNamespace(
        load=lambda: None, tokenizer=tokenize, device="cpu", max_length=96,
        model=SimpleNamespace(backbone=lambda **kw: SimpleNamespace(
            last_hidden_state=kw["input_ids"].unsqueeze(-1).float())),
    )
    fsm = GenericInstructionFSM(parser=SimpleNamespace(parser=inner), cache_capacity=1)
    a = ParsedInstruction(parsed_intent="SET_SPEED", target_speed_kmh=20)
    b = ParsedInstruction(parsed_intent="SET_SPEED", target_speed_kmh=40)
    first = fsm.encode_tokens(a, cache_key="shared")
    assert fsm.encode_tokens(a, cache_key="shared") is first
    fsm.encode_tokens(b, cache_key="shared")
    assert len(calls) == 2 and "20.0" in calls[0] and "40.0" in calls[1]
    assert len(fsm._token_cache) == 1
    fsm.clear_caches()
    assert not fsm._token_cache
