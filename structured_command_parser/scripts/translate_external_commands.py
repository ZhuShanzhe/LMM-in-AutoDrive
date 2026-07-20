from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any


SYSTEM_PROMPT = """你是自动驾驶数据标注翻译员。把英文乘客驾驶指令翻译成自然、简洁、忠实的简体中文。
必须保留动作顺序、方向、数字、单位、目标对象和危险意图；不得把危险指令改写为安全建议。
只输出翻译结果，不要解释，不要添加引号或前缀。"""


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def clean_translation(text: str) -> str:
    text = text.strip().strip('"').strip("'")
    for prefix in ("翻译：", "翻译:", "中文：", "中文:"):
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()
    return text.splitlines()[0].strip()


def contains_chinese(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def preserve_urgency(english: str, chinese: str) -> str:
    urgent_terms = ("now", "immediate", "instantly", "without delay", "right away", "at once")
    if any(term in english.casefold() for term in urgent_terms) and not any(
        term in chinese for term in ("立即", "立刻", "马上", "紧急")
    ):
        return f"立即{chinese.lstrip('请')}"
    return chinese


def main() -> None:
    parser = argparse.ArgumentParser(description="Translate command JSONL with local Qwen")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = load_jsonl(args.input)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    model.eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as output:
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            prompts = [
                tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": row["text_en"]},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for row in batch
            ]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=96,
                    do_sample=False,
                    use_cache=True,
                )
            prompt_length = inputs["input_ids"].shape[1]
            for row, generated in zip(batch, outputs, strict=True):
                translated = tokenizer.decode(
                    generated[prompt_length:], skip_special_tokens=True
                )
                translated = clean_translation(translated)
                if not contains_chinese(translated):
                    retry_prompt = tokenizer.apply_chat_template(
                        [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {
                                "role": "user",
                                "content": f"必须用简体中文翻译，不得保留英文句子：{row['text_en']}",
                            },
                        ],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    retry_inputs = tokenizer(
                        retry_prompt, return_tensors="pt"
                    ).to(model.device)
                    with torch.inference_mode():
                        retry_output = model.generate(
                            **retry_inputs,
                            max_new_tokens=96,
                            do_sample=False,
                            use_cache=True,
                        )[0]
                    translated = clean_translation(
                        tokenizer.decode(
                            retry_output[retry_inputs["input_ids"].shape[1] :],
                            skip_special_tokens=True,
                        )
                    )
                row["text_zh"] = preserve_urgency(row["text_en"], translated)
                row["translation_status"] = (
                    f"MACHINE_TRANSLATED_{Path(args.model).name.upper()}"
                )
                if "expected" in row:
                    row["text"] = row["text_zh"]
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()
            print(f"translated: {min(start + len(batch), len(rows))}/{len(rows)}")


if __name__ == "__main__":
    main()
