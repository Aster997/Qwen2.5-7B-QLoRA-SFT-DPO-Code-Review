"""
检查训练数据的 token 长度分布，找出超过 cutoff_len 的样本

用法：
  python scripts/check_token_length.py \
      --model /root/autodl-tmp/models/Qwen/Qwen2___5-7B-Instruct \
      --data data/final/sft_pro_train.json \
      --cutoff 2048
"""

import argparse
import json
from collections import Counter

SYSTEM_PROMPT = "你是一个专业的代码审查助手，能够识别代码中的问题并给出改进建议。"


def build_full_text(tokenizer, item: dict) -> str:
    messages = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": f"{item['instruction']}\n\n```\n{item['input']}\n```"},
        {"role": "assistant", "content": item["output"]},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   required=True, help="tokenizer 路径")
    parser.add_argument("--data",    required=True, help="训练集 JSON 路径")
    parser.add_argument("--cutoff",  type=int, default=2048)
    parser.add_argument("--filter-out", default=None, help="输出过滤后的 JSON 路径（可选）")
    args = parser.parse_args()

    from transformers import AutoTokenizer
    print(f"加载 tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    with open(args.data, encoding="utf-8") as f:
        data = json.load(f)
    print(f"共 {len(data)} 条样本\n")

    lengths = []
    over_limit = []

    for i, item in enumerate(data):
        text = build_full_text(tokenizer, item)
        n = len(tokenizer(text)["input_ids"])
        lengths.append(n)
        if n > args.cutoff:
            over_limit.append((i, n, item.get("instruction", "")[:40]))

    # 统计分布
    buckets = [512, 1024, 1536, 2048, 2560, 3072, 4096, 99999]
    labels  = ["≤512", "512-1024", "1024-1536", "1536-2048",
               "2048-2560", "2560-3072", "3072-4096", ">4096"]
    counts = [0] * len(labels)
    for l in lengths:
        for j, b in enumerate(buckets):
            if l <= b:
                counts[j] += 1
                break

    print("=" * 50)
    print(f"Token 长度分布（cutoff={args.cutoff}）")
    print("=" * 50)
    for label, cnt in zip(labels, counts):
        bar = "█" * (cnt * 40 // len(data))
        print(f"  {label:<12} {cnt:>4} 条  {bar}")

    print(f"\n  最短: {min(lengths)}")
    print(f"  最长: {max(lengths)}")
    print(f"  平均: {sum(lengths)//len(lengths)}")
    print(f"  中位: {sorted(lengths)[len(lengths)//2]}")
    print(f"\n  超过 {args.cutoff} token 的样本: {len(over_limit)} 条 ({len(over_limit)/len(data)*100:.1f}%)")

    if over_limit:
        print(f"\n  前10条超长样本：")
        for idx, n, instr in over_limit[:10]:
            print(f"    样本[{idx}] {n} tokens | {instr}...")

    # 可选：输出过滤后的数据集
    if args.filter_out:
        kept = [item for i, item in enumerate(data) if lengths[i] <= args.cutoff]
        with open(args.filter_out, "w", encoding="utf-8") as f:
            json.dump(kept, f, ensure_ascii=False, indent=2)
        print(f"\n  过滤后保留 {len(kept)} 条 → {args.filter_out}")


if __name__ == "__main__":
    main()
