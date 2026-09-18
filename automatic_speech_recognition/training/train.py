import argparse
import logging
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.config import load_config, resolve_paths
from training.dataset.manifest import ASRCollator, ASRManifestDataset
from training.utils import log_and_print, read_jsonl, setup_logging

PATH_KEYS = ["manifest", "eval_manifest", "replay_manifest", "output_dir"]

def _load_rows(cfg) -> list:
    rows = read_jsonl(cfg["manifest"])
    replay = cfg.get("replay_manifest")
    ratio = float(cfg.get("replay_ratio", 0.3))
    if replay and Path(replay).exists():
        clean = [r for r in read_jsonl(replay) if r.get("condition", "clean") == "clean"]
        budget = int(len(rows) * ratio)
        if clean and budget > 0:
            rows += random.sample(clean, min(budget, len(clean)))
    random.shuffle(rows)
    return rows

def _build_processor(model_id_or_path: str):
    from transformers import AutoProcessor
    try:
        return AutoProcessor.from_pretrained(model_id_or_path, trust_remote_code=True)
    except Exception:
        from qwen_asr import Qwen3ASRModel
        wrapper = Qwen3ASRModel.from_pretrained(model_id_or_path, trust_remote_code=True)
        for attr in ("processor", "tokenizer"):
            if hasattr(wrapper, attr):
                return getattr(wrapper, attr)
        raise RuntimeError("no processor/tokenizer found for fine-tuning")

def run_lora(cfg, processor):
    from transformers import Trainer, TrainingArguments
    from training.models.lora import build_lora_model

    rows = _load_rows(cfg)
    if not rows:
        raise SystemExit("empty training manifest")
    model = build_lora_model(
        cfg["model_id_or_path"], cfg.get("lora", {}),
        cfg.get("dtype", "bfloat16"), cfg.get("device_map", "cuda:0"),
        cfg.get("attn_implementation") or ""
    )
    dataset = ASRManifestDataset(rows, cfg["prompt"], int(cfg["sample_rate"]))
    collator = ASRCollator(processor, int(cfg["max_length"]))
    args = TrainingArguments(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=int(cfg["batch_size"]),
        gradient_accumulation_steps=int(cfg["grad_accum"]),
        learning_rate=float(cfg["learning_rate"]),
        num_train_epochs=float(cfg["epochs"]),
        warmup_ratio=float(cfg["warmup_ratio"]),
        logging_steps=int(cfg["logging_steps"]),
        save_steps=int(cfg["save_steps"]),
        save_total_limit=2,
        bf16=bool(cfg["bf16"]),
        gradient_checkpointing=bool(cfg["gradient_checkpointing"]),
        remove_unused_columns=False,
        report_to=[],
    )
    Trainer(model=model, args=args, train_dataset=dataset, data_collator=collator).train()
    model.save_pretrained(cfg["output_dir"])
    log_and_print(f"LoRA adapter saved -> {cfg['output_dir']}")

def run_adapter(cfg, processor):
    from training.models.adapters import (freeze_all, inject_adapters, parameter_summary, save_adapters)
    from training.models.lora import load_asr_model
    import os
    
    model = load_asr_model(
        cfg["model_id_or_path"], cfg.get("dtype", "bfloat16"),
        cfg.get("device_map", "cuda:0"),
        cfg.get("attn_implementation") or ""
    )
    ad_cfg = cfg.get("adapter", {})
    adapters, _ = inject_adapters(
        model, target=ad_cfg.get("target", "audio"),
        bottleneck=int(ad_cfg.get("bottleneck", 64)),
        dropout=float(ad_cfg.get("dropout", 0.0)),
        last_n=ad_cfg.get("last_n", 6))
    freeze_all(model)
    for p in adapters.parameters():
        p.requires_grad = True
    log_and_print(f"adapter params: {parameter_summary(model)}")
    out = os.path.join(cfg["output_dir"], "adapters.pt")
    save_adapters(adapters, out)
    log_and_print(f"adapters saved -> {out}")
    log_and_print("attach a Trainer here to optimize the adapters on your manifest")

def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Qwen3-ASR")
    parser.add_argument("--config", default="configs/training/finetune.yaml")
    parser.add_argument("--log-file", default="logs/training/train.log")
    args = parser.parse_args()

    setup_logging(args.log_file)
    cfg = resolve_paths(load_config(args.config), PATH_KEYS)
    log_and_print(f"method={cfg['method']} model={cfg['model_id_or_path']}")

    processor = _build_processor(cfg["model_id_or_path"])
    if cfg["method"] == "lora":
        run_lora(cfg, processor)
    elif cfg["method"] == "adapter":
        run_adapter(cfg, processor)
    else:
        raise SystemExit("unsupported method: " + str(cfg["method"]))

if __name__ == "__main__":
    main()