"""
批量推理评测脚本 - 支持单模型 & 三模型对比
支持两种输入格式：
  - .txt  eval_manual_samples.txt（手工标注格式）
  - .json alpaca 格式（{"instruction":..., "input":..., "output":...}）

单模型模式（txt）：
  python scripts/batch_eval.py \
      --model /root/autodl-tmp/models/Qwen/Qwen2___5-7B-Instruct \
      --adapter /root/autodl-tmp/saves/qwen2.5-7b/qlora/sft_pro \
      --samples data/eval_manual_samples.txt \
      --out data/eval_results_sft.txt

单模型模式（json，50条测试集）：
  python scripts/batch_eval.py \
      --model /root/autodl-tmp/models/Qwen/Qwen2___5-7B-Instruct \
      --adapter /root/autodl-tmp/saves/qwen2.5-7b/qlora/sft_pro \
      --samples data/final/sft_pro_test.json \
      --out data/eval_results_sft_pro_full.txt

三模型对比模式：
  python scripts/batch_eval.py --compare \
      --model /root/autodl-tmp/models/Qwen/Qwen2___5-7B-Instruct \
      --sft-adapter   /root/autodl-tmp/saves/qwen2.5-7b/qlora/sft_pro \
      --dpo-adapter   /root/autodl-tmp/saves/qwen2.5-7b/qlora/dpo_pro \
      --samples data/final/sft_pro_test.json \
      --out data/eval_compare.txt
"""

import argparse
import json
import re
import sys
import torch


# ─────────────────────────────────────────────
# 1. 解析样本（支持 .txt 和 .json 两种格式）
# ─────────────────────────────────────────────

def parse_samples_json(json_path: str) -> list[dict]:
    """从 alpaca 格式 JSON 文件解析样本"""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    samples = []
    for i, item in enumerate(data, 1):
        samples.append({
            "idx":         i,
            "lang":        "Unknown",
            "instruction": item.get("instruction", ""),
            "input":       item.get("input", ""),
            "reference":   item.get("output", ""),
        })
    return samples


def parse_samples(txt_path: str) -> list[dict]:
    """从 eval_manual_samples.txt 解析出所有样本"""
    with open(txt_path, encoding="utf-8") as f:
        content = f.read().replace("\r\n", "\n").replace("\r", "\n")

    blocks = re.split(r"={50,}", content)
    samples = []

    for block in blocks:
        if "[Instruction]" not in block:
            continue

        def extract_between(start_tag, end_tag, block=block):
            """提取两个标签之间的内容，end_tag 只需前缀匹配"""
            pattern = rf"\[{re.escape(start_tag)}\]\n(.*?)\n\[{re.escape(end_tag)}"
            m = re.search(pattern, block, re.DOTALL)
            return m.group(1).strip() if m else ""

        lang_m = re.search(r"语言:\s*(.+)", block)
        idx_m  = re.search(r"样本\s*(\d+)/", block)

        instruction = extract_between("Instruction", "Input")
        code        = extract_between("Input - 待审查代码", "标准答案")
        reference   = extract_between("标准答案", "模型输出")

        if not instruction or not code:
            continue

        samples.append({
            "idx":         int(idx_m.group(1)) if idx_m else len(samples) + 1,
            "lang":        lang_m.group(1).strip() if lang_m else "Unknown",
            "instruction": instruction,
            "input":       code,
            "reference":   reference,
        })

    return samples


# ─────────────────────────────────────────────
# 2. 加载模型
# ─────────────────────────────────────────────

def load_model(model_path: str, adapter_path: str = None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print(f"  加载 tokenizer: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True
    )

    print(f"  加载模型: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    if adapter_path:
        print(f"  加载 adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, tokenizer


# ─────────────────────────────────────────────
# 3. 单条推理
# ─────────────────────────────────────────────

SYSTEM_PROMPT = "你是一个专业的代码审查助手，能够识别代码中的问题并给出改进建议。"

def build_prompt(tokenizer, instruction: str, code: str) -> str:
    messages = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": f"{instruction}\n\n```\n{code}\n```"},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def infer_one(model, tokenizer, instruction: str, code: str,
              max_new_tokens: int = 512) -> str:
    prompt = build_prompt(tokenizer, instruction, code)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


# ─────────────────────────────────────────────
# 4. 写输出文件
# ─────────────────────────────────────────────

def write_single(samples: list[dict], out_path: str, model_label: str):
    lines = []
    lines.append("=" * 70)
    lines.append(f"代码审查助手 评测结果 - {model_label}")
    lines.append(f"评分维度：问题识别准确性 / 改进建议可行性 / 有无误报")
    lines.append("=" * 70)

    for s in samples:
        lines.append(f"\n【样本 {s['idx']}】语言: {s['lang']}")
        lines.append("-" * 70)
        lines.append(f"[Instruction]\n{s['instruction']}")
        lines.append(f"\n[Input - 待审查代码]\n{s['input']}")
        lines.append(f"\n[标准答案]\n{s['reference']}")
        lines.append(f"\n[模型输出 - {model_label}]\n{s.get('prediction', '（未生成）')}")
        lines.append(f"\n[人工评分] 准确性: /5  可行性: /5  误报: 有/无  总评: /10")
        lines.append("=" * 70)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n结果已保存: {out_path}")


def write_compare(samples: list[dict], out_path: str):
    lines = []
    lines.append("=" * 70)
    lines.append("代码审查助手 三模型对比评测")
    lines.append("模型：Base / SFT / SFT+DPO")
    lines.append("=" * 70)

    for s in samples:
        lines.append(f"\n【样本 {s['idx']}】语言: {s['lang']}")
        lines.append("-" * 70)
        lines.append(f"[Instruction]\n{s['instruction']}")
        lines.append(f"\n[Input - 待审查代码]\n{s['input']}")
        lines.append(f"\n[标准答案]\n{s['reference']}")
        lines.append(f"\n{'─' * 35} Base 模型 {'─' * 35}")
        lines.append(s.get("base", "（未生成）"))
        lines.append(f"\n{'─' * 35} SFT 模型 {'─' * 35}")
        lines.append(s.get("sft", "（未生成）"))
        lines.append(f"\n{'─' * 35} SFT+DPO 模型 {'─' * 35}")
        lines.append(s.get("dpo", "（未生成）"))
        lines.append(f"\n[人工评分]")
        lines.append(f"  Base:    准确性: /5  可行性: /5  误报: 有/无  总评: /10")
        lines.append(f"  SFT:     准确性: /5  可行性: /5  误报: 有/无  总评: /10")
        lines.append(f"  SFT+DPO: 准确性: /5  可行性: /5  误报: 有/无  总评: /10")
        lines.append("=" * 70)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n对比结果已保存: {out_path}")


# ─────────────────────────────────────────────
# 5. 主流程
# ─────────────────────────────────────────────

def run_inference(model, tokenizer, samples: list[dict],
                  key: str, label: str) -> list[dict]:
    print(f"\n开始推理 [{label}]，共 {len(samples)} 条...")
    for i, s in enumerate(samples, 1):
        print(f"  [{i}/{len(samples)}] 样本{s['idx']} ({s['lang']})...", end=" ", flush=True)
        s[key] = infer_one(model, tokenizer, s["instruction"], s["input"])
        print("✓")
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       required=True, help="基座模型路径")
    parser.add_argument("--adapter",     default=None,  help="单模型模式: adapter 路径")
    parser.add_argument("--sft-adapter", default=None,  help="对比模式: SFT adapter 路径")
    parser.add_argument("--dpo-adapter", default=None,  help="对比模式: DPO adapter 路径")
    parser.add_argument("--samples",     required=True, help="eval_manual_samples.txt 路径")
    parser.add_argument("--out",         required=True, help="输出文件路径")
    parser.add_argument("--compare",     action="store_true", help="三模型对比模式")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    args = parser.parse_args()

    print(f"解析样本文件: {args.samples}")
    if args.samples.endswith(".json"):
        samples = parse_samples_json(args.samples)
    else:
        samples = parse_samples(args.samples)
    print(f"共解析到 {len(samples)} 条样本")

    if not samples:
        print("错误：未解析到任何样本，请检查文件格式")
        sys.exit(1)

    if args.compare:
        # ── 三模型对比模式 ──
        # Base 模型
        print("\n[1/3] Base 模型（无 adapter）")
        model, tokenizer = load_model(args.model, adapter_path=None)
        samples = run_inference(model, tokenizer, samples, key="base", label="Base")
        del model
        torch.cuda.empty_cache()

        # SFT 模型
        if args.sft_adapter:
            print("\n[2/3] SFT 模型")
            model, tokenizer = load_model(args.model, adapter_path=args.sft_adapter)
            samples = run_inference(model, tokenizer, samples, key="sft", label="SFT")
            del model
            torch.cuda.empty_cache()
        else:
            print("\n[2/3] 跳过 SFT（未提供 --sft-adapter）")

        # DPO 模型
        if args.dpo_adapter:
            print("\n[3/3] SFT+DPO 模型")
            model, tokenizer = load_model(args.model, adapter_path=args.dpo_adapter)
            samples = run_inference(model, tokenizer, samples, key="dpo", label="SFT+DPO")
            del model
            torch.cuda.empty_cache()
        else:
            print("\n[3/3] 跳过 DPO（未提供 --dpo-adapter）")

        write_compare(samples, args.out)

    else:
        # ── 单模型模式 ──
        label = "SFT" if args.adapter else "Base"
        model, tokenizer = load_model(args.model, adapter_path=args.adapter)
        samples = run_inference(model, tokenizer, samples,
                                key="prediction", label=label)
        write_single(samples, args.out, model_label=label)

    # 同时保存 JSON 方便后续分析
    json_out = args.out.replace(".txt", ".json")
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    print(f"JSON 结果已保存: {json_out}")


if __name__ == "__main__":
    main()
