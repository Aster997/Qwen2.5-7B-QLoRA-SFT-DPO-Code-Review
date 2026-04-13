"""
检查训练数据的 token 长度分布，找出超过 cutoff_len 的样本

v3 更新：支持三种样本类型
  - 普通样本：instruction + input + output
  - 对抗样本：instruction + output（input 为空）
  - 多轮样本：history + instruction + output

用法：
  python scripts/check_token_length.py \
      --model /root/autodl-tmp/models/Qwen/Qwen2___5-7B-Instruct \
      --data data/final/sft_v3_train.json \
      --cutoff 2048
"""

import argparse
import json

SYSTEM_PROMPT = "你是一个专业的代码审查助手，能够识别代码中的问题并给出改进建议。"


def build_user_content(instruction: str, input_text: str) -> str:
    """按训练时的拼接方式构造 user 消息内容。"""
    instruction = instruction or ""
    input_text = input_text or ""
    if input_text.strip():
        return f"{instruction}\n\n```\n{input_text}\n```"
    return instruction


def build_full_text(tokenizer, item: dict) -> str:
    """把 Alpaca 格式（含可选 history）还原成 chat_template 文本。"""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # 多轮历史（Alpaca 格式的 history 是 [[user, assistant], ...]）
    history = item.get("history") or []
    for turn in history:
        if isinstance(turn, (list, tuple)) and len(turn) == 2:
            u, a = turn
            messages.append({"role": "user", "content": u})
            messages.append({"role": "assistant", "content": a})

    # 当前轮
    messages.append({
        "role": "user",
        "content": build_user_content(item.get("instruction", ""), item.get("input", "")),
    })
    messages.append({"role": "assistant", "content": item.get("output", "")})

    return tokenizer.apply_chat_template(messages, tokenize=False)


def sample_type(item: dict) -> str:
    if item.get("history"):
        return "multiturn"
    if not (item.get("input") or "").strip():
        return "adversarial"
    return "normal"


def percentile(sorted_list, p):
    """简单分位数实现，p ∈ [0, 100]"""
    if not sorted_list:
        return 0
    k = (len(sorted_list) - 1) * p / 100
    f, c = int(k), min(int(k) + 1, len(sorted_list) - 1)
    return int(sorted_list[f] + (sorted_list[c] - sorted_list[f]) * (k - f))


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
    by_type = {"normal": [], "adversarial": [], "multiturn": []}

    for i, item in enumerate(data):
        text = build_full_text(tokenizer, item)
        n = len(tokenizer(text)["input_ids"])
        lengths.append(n)
        by_type[sample_type(item)].append(n)
        if n > args.cutoff:
            over_limit.append((i, n, sample_type(item), (item.get("instruction") or "")[:40]))

    # 分桶分布
    buckets = [512, 1024, 1536, 2048, 2560, 3072, 4096, 99999]
    labels  = ["≤512", "512-1024", "1024-1536", "1536-2048",
               "2048-2560", "2560-3072", "3072-4096", ">4096"]
    counts = [0] * len(labels)
    for l in lengths:
        for j, b in enumerate(buckets):
            if l <= b:
                counts[j] += 1
                break

    print("=" * 60)
    print(f"Token 长度分布（cutoff={args.cutoff}）")
    print("=" * 60)
    for label, cnt in zip(labels, counts):
        bar = "█" * (cnt * 40 // max(len(data), 1))
        print(f"  {label:<12} {cnt:>4} 条  {bar}")

    # 分位数
    sl = sorted(lengths)
    print("\n  统计指标：")
    print(f"    最短 / 中位 / 平均 / 最长 = {min(lengths)} / {sl[len(sl)//2]} / {sum(lengths)//len(lengths)} / {max(lengths)}")
    print(f"    P50 / P90 / P95 / P99    = {percentile(sl,50)} / {percentile(sl,90)} / {percentile(sl,95)} / {percentile(sl,99)}")
    print(f"\n  超过 {args.cutoff} token 的样本: {len(over_limit)} 条 ({len(over_limit)/len(data)*100:.1f}%)")

    # 分类型统计
    print("\n  按样本类型分布：")
    for t, ls in by_type.items():
        if not ls:
            print(f"    {t:<12} 0 条")
            continue
        ls_sorted = sorted(ls)
        over = sum(1 for x in ls if x > args.cutoff)
        print(f"    {t:<12} {len(ls):>4} 条 | 中位 {ls_sorted[len(ls_sorted)//2]:>4} | P95 {percentile(ls_sorted,95):>4} | P99 {percentile(ls_sorted,99):>4} | 超限 {over}")

    if over_limit:
        print(f"\n  前10条超长样本：")
        for idx, n, t, instr in over_limit[:10]:
            print(f"    样本[{idx}] {n:>5} tokens [{t:<11}] {instr}...")

    # cutoff 建议
    print("\n  ───── cutoff_len 建议 ─────")
    p95, p99 = percentile(sl, 95), percentile(sl, 99)
    # 取 P99 向上对齐到 256
    suggested = ((p99 + 255) // 256) * 256
    print(f"    P95={p95}, P99={p99}")
    print(f"    推荐 cutoff_len = {suggested}（P99 向上取整到 256 的倍数）")
    print(f"    当前 cutoff_len = {args.cutoff} → 截断 {len(over_limit)} 条 ({len(over_limit)/len(data)*100:.1f}%)")

    # 可选：输出过滤后的数据集
    if args.filter_out:
        kept = [item for i, item in enumerate(data) if lengths[i] <= args.cutoff]
        with open(args.filter_out, "w", encoding="utf-8") as f:
            json.dump(kept, f, ensure_ascii=False, indent=2)
        print(f"\n  过滤后保留 {len(kept)} 条 → {args.filter_out}")


if __name__ == "__main__":
    main()
