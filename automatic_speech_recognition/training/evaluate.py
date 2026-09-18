import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from training.config import load_config, resolve_path
from training.dataset.manifest import ASRManifestDataset
from training.utils import corpus_cer, log_and_print, read_jsonl, save_json, setup_logging

def _load_processor(model_id_or_path: str):
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained(model_id_or_path, trust_remote_code=True)

@torch.no_grad()
def evaluate_manifest(model, processor, manifest: str, prompt: str, sample_rate: int = 16000, max_new_tokens: int = 256) -> dict:
    model.eval()
    dataset = ASRManifestDataset(read_jsonl(manifest), prompt, sample_rate)
    device = next(model.parameters()).device
    refs, hyps = [], []
    for item in dataset:
        inputs = processor.apply_chat_template(
            item["messages"][:-1], tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
        prompt_len = inputs["input_ids"].shape[-1]
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens)
        text = processor.tokenizer.decode(generated[0][prompt_len:], skip_special_tokens=True)
        refs.append(item["text"])
        hyps.append(text.strip())
    return {"samples": len(refs), "cer": round(corpus_cer(refs, hyps), 4)}

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Qwen3-ASR checkpoints")
    parser.add_argument("--config", default="configs/training/finetune.yaml")
    parser.add_argument("--checkpoint", default=None, help="path to a saved LoRA dir")
    parser.add_argument("--log-file", default="logs/training/evaluate.log", help="run log path; empty string disables it")
    args = parser.parse_args()

    setup_logging(args.log_file)
    cfg = load_config(args.config)

    model = None
    if args.checkpoint:
        from peft import PeftModel
        from training.models.lora import load_asr_model
        base = load_asr_model(cfg["model_id_or_path"], cfg.get("dtype", "bfloat16"), cfg.get("device_map", "cuda:0"))
        model = PeftModel.from_pretrained(base, args.checkpoint)
    else:
        from training.models.lora import load_asr_model
        model = load_asr_model(cfg["model_id_or_path"], cfg.get("dtype", "bfloat16"), cfg.get("device_map", "cuda:0"))

    processor = _load_processor(cfg["model_id_or_path"])
    report = {}
    for name in ("manifest", "eval_manifest"):
        path = cfg.get(name)
        if path and Path(resolve_path(path)).exists():
            metrics = evaluate_manifest(model, processor, resolve_path(path), cfg["prompt"], int(cfg["sample_rate"]), int(cfg.get("eval_max_new_tokens", 256)))
            report[name] = metrics
            log_and_print(f"{name}: {metrics}")

    save_json(report, str(ROOT / "outputs" / "asr_finetune" / "eval_cer.json"))
    log_and_print("evaluation done")

if __name__ == "__main__":
    main()