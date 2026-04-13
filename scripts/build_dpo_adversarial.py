"""
对抗样本 DPO 数据补充脚本

问题背景：
  build_dpo_data_v3.py 的 on-policy 采样对对抗 prompt 无效：
    - 对抗 prompt（无代码输入）上 SFT_v3 每次都拒绝 → 所有输出相似 → gap < 1.5 → 全部过滤

解决方案：
  对抗对的 chosen/rejected 来自两个不同模型，不需要 judge 打分：
    - chosen   = 训练数据里的 reference（正确拒绝/引导提交代码），已人工标注
    - rejected = base 模型采样（无 adapter），几乎必然顺从回答无关问题

过滤逻辑（规则化，无需 LLM judge）：
  - 丢弃：base 模型恰好也拒绝了（两者都在拒绝，对没有信号）
  - 丢弃：chosen 和 rejected 前 80 字高度相似（>= 0.7 Jaccard）
  - 保留：rejected 实质性不同（longer + no refusal phrases）

用法：
  python scripts/build_dpo_adversarial.py \\
      --model /root/autodl-tmp/models/Qwen/Qwen2___5-7B-Instruct \\
      --prompts data/final/sft_v3_train.json \\
      --out data/dpo_v3_adversarial.json \\
      --n-samples 3   # 每个 prompt 采样几次（取最"顺从"的那个）

  # 合并到已有 DPO 数据
  python scripts/build_dpo_adversarial.py ... --merge data/dpo_v3_onpolicy.json
"""

import argparse
import json
import random
import re
from pathlib import Path
from typing import Optional


SYSTEM_PROMPT = "你是一个专业的代码审查助手，能够识别代码中的问题并给出改进建议。"

# 拒绝/引导短语 —— 出现在 rejected 里就说明 base 也拒了，这对没价值
REFUSAL_PHRASES = [
    "请提供代码",
    "请提供具体的代码",
    "请将代码",
    "没有代码可供审查",
    "无法进行有效的代码审查",
    "无法审查",
    "需要提供代码",
    "请粘贴代码",
    "请贴出代码",
    "代码片段不完整",
    "没有实际代码",
    "无代码输入",
    "请给出代码",
]


# ═══════════════════════════════════════════════════════════
# 1. 工具函数
# ═══════════════════════════════════════════════════════════

def is_refusal(text: str) -> bool:
    """判断一段输出是否是"请提供代码"式的拒绝引导。"""
    text_lower = text.lower()
    return any(phrase in text for phrase in REFUSAL_PHRASES) and len(text) < 400


def jaccard_similarity(a: str, b: str) -> float:
    """粗粒度 token Jaccard 相似度（中文按字符，英文按空格分词）。"""
    def tokenize(s):
        # 中英文混合：保留汉字 + 字母数字，其余当分隔符
        tokens = re.findall(r'[\u4e00-\u9fff]|[a-zA-Z0-9]+', s[:200])
        return set(tokens)
    a_set, b_set = tokenize(a), tokenize(b)
    if not a_set or not b_set:
        return 0.0
    return len(a_set & b_set) / len(a_set | b_set)


def is_valid_pair(chosen: str, rejected: str) -> tuple[bool, str]:
    """返回 (是否保留, 原因)。"""
    # 1. rejected 太短（base 可能没生成）
    if len(rejected.strip()) < 30:
        return False, f"rejected 太短 ({len(rejected)} chars)"

    # 2. base 模型也拒绝了
    if is_refusal(rejected):
        return False, "base 模型也拒绝了"

    # 3. chosen 和 rejected 过于相似（无训练价值）
    sim = jaccard_similarity(chosen, rejected)
    if sim >= 0.65:
        return False, f"相似度过高 (jaccard={sim:.2f})"

    return True, "ok"


# ═══════════════════════════════════════════════════════════
# 2. 加载 base 模型（不加 adapter）
# ═══════════════════════════════════════════════════════════

def load_base_model(model_path: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"  加载 tokenizer: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"  加载 base 模型（无 adapter）: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


# ═══════════════════════════════════════════════════════════
# 3. 采样（对抗 prompt 用高温度，让 base 更容易"顺从"）
# ═══════════════════════════════════════════════════════════

def build_messages(item: dict) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    instruction = (item.get("instruction") or "").strip()
    # 对抗样本：input 为空，直接用 instruction
    messages.append({"role": "user", "content": instruction})
    return messages


def sample_outputs(model, tokenizer, messages: list[dict],
                   n: int = 3, max_new_tokens: int = 512) -> list[str]:
    """对抗 prompt 采样 n 次（高温度，让 base 尽量给出"顺从"的多样输出）。"""
    import torch

    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    # 高温度让 base 更"愿意"回答
    temperatures = [0.8, 1.0, 1.1][:n]
    while len(temperatures) < n:
        temperatures.append(1.0)

    outputs = []
    for t in temperatures:
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=t,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
            )
        new_ids = out_ids[0][input_len:]
        text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        outputs.append(text)

    return outputs


def pick_all_rejected(chosen: str, candidates: list[str]) -> list[str]:
    """返回所有有效的 rejected 候选（全部保留，增加对抗对数量）。"""
    valid = []
    for c in candidates:
        if not is_refusal(c) and len(c.strip()) >= 30:
            ok, _ = is_valid_pair(chosen, c)
            if ok:
                valid.append(c)
    # 去重（不同温度可能生成相同内容）
    seen = set()
    deduped = []
    for c in valid:
        key = c[:80]
        if key not in seen:
            seen.add(key)
            deduped.append(c)
    return deduped


# ═══════════════════════════════════════════════════════════
# 4. 加载对抗 prompts
# ═══════════════════════════════════════════════════════════

def load_adversarial_prompts(path: str, seed: int = 42) -> list[dict]:
    """从训练数据里筛出纯对抗样本（input 为空 + 无 history）。"""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    adversarial = [
        item for item in data
        if not (item.get("input") or "").strip()
        and not item.get("history")
    ]

    random.seed(seed)
    random.shuffle(adversarial)
    print(f"  共找到 {len(adversarial)} 个对抗样本（input 为空 + 无 history）")
    return adversarial


# ═══════════════════════════════════════════════════════════
# 5. 主流程
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",        required=True, help="base 模型路径（不加 adapter）")
    parser.add_argument("--prompts",      required=True, help="SFT v3 训练数据 JSON")
    parser.add_argument("--out",          required=True, help="输出 DPO pairs JSON 路径")
    parser.add_argument("--merge",        default=None,  help="若指定，将结果追加合并到该文件（如 dpo_v3_onpolicy.json）")
    parser.add_argument("--n-samples",    type=int, default=3, help="每 prompt 采样几次（取最顺从的一个）")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-prompts",  type=int, default=0,
                        help="最多处理多少对抗 prompt（0=全部）")
    parser.add_argument("--seed",         type=int, default=42)
    args = parser.parse_args()

    # ── 加载 prompts ──
    print("\n[1/3] 加载对抗 prompt 池")
    prompts = load_adversarial_prompts(args.prompts, args.seed)
    if args.max_prompts > 0:
        prompts = prompts[:args.max_prompts]
        print(f"  限制处理 {len(prompts)} 个（--max-prompts）")

    # ── 加载模型 ──
    print("\n[2/3] 加载 base 模型")
    model, tokenizer = load_base_model(args.model)

    # ── 主循环 ──
    print(f"\n[3/3] 开始生成对抗 DPO 对（每 prompt 采样 {args.n_samples} 次）\n")
    pairs = []
    skipped_base_refused = 0
    skipped_other = 0

    for i, item in enumerate(prompts, 1):
        instr_preview = (item.get("instruction") or "")[:50]
        chosen = (item.get("output") or "").strip()

        if not chosen:
            print(f"  [{i}/{len(prompts)}] 跳过：reference 为空")
            skipped_other += 1
            continue

        print(f"  [{i}/{len(prompts)}] {instr_preview}...", end=" ", flush=True)

        try:
            messages = build_messages(item)
            candidates = sample_outputs(
                model, tokenizer, messages,
                n=args.n_samples,
                max_new_tokens=args.max_new_tokens,
            )

            rejected_list = pick_all_rejected(chosen, candidates)

            if not rejected_list:
                print(f"✗ base 全部拒绝或相似（{len(candidates)} 候选）")
                skipped_base_refused += 1
                continue

            for rejected in rejected_list:
                pairs.append({
                    "instruction":    item.get("instruction", ""),
                    "input":          "",
                    "history":        [],
                    "chosen":         chosen,
                    "rejected":       rejected,
                    "chosen_score":   5.0,
                    "rejected_score": 1.0,
                    "source":         "adversarial_base_vs_ref",
                })
            print(f"✓  +{len(rejected_list)} pairs "
                  f"(total {len(pairs)}, chosen {len(chosen)}c)")

        except Exception as e:
            print(f"✗ 异常: {e}")
            skipped_other += 1
            continue

    # ── 保存 ──
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)

    print(f"""
╔══════════════════════════════════════╗
  对抗 DPO 数据生成完成
  成功对:          {len(pairs)} 条
  base 也拒绝/相似: {skipped_base_refused} 条（过滤）
  其他异常/空:     {skipped_other} 条
  输出: {out_path}
╚══════════════════════════════════════╝""")

    # ── 可选：合并到现有 DPO 数据 ──
    if args.merge and pairs:
        merge_path = Path(args.merge)
        existing = []
        if merge_path.exists():
            with open(merge_path, encoding="utf-8") as f:
                existing = json.load(f)
            print(f"\n合并到 {merge_path}（原有 {len(existing)} 对）")

        merged = existing + pairs
        with open(merge_path, "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)
        print(f"合并后共 {len(merged)} 对，已写回 {merge_path}")


if __name__ == "__main__":
    main()
