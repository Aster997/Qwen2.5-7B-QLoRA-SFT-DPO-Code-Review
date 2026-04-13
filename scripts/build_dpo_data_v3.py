"""
DPO v3 数据构建脚本（on-policy）

设计原则：
  - **on-policy**：rejected 来自 sft_v3 模型本身的实际采样输出，不是 guess
  - **多样化采样**：对每个 prompt 采样 N 个输出（默认 4），温度 0.7-1.0 制造差异
  - **LLM-as-judge 打分**：DeepSeek V3 对 N 个输出 1-5 分打分
  - **梯度筛选**：只保留 best - worst >= threshold 的对（默认 1.5）——信号太弱的跳过

为什么这样设计：
  - v2 DPO 退化是因为 off-policy: chosen/rejected 都是"猜"的，模型从没真正输出过 rejected
  - on-policy 解决：模型确实会生成 rejected 这种响应，DPO 训练就能真正"压低"它
  - 判分阈值避免引入噪声对（两个都差不多的输出配对会让模型学乱）

用法：

  # 1) 从 sft_v3_train.json 采样 1000 条 prompt，生成 DPO 对
  python scripts/build_dpo_data_v3.py \\
      --model /root/autodl-tmp/models/Qwen/Qwen2___5-7B-Instruct \\
      --adapter /root/autodl-tmp/saves/qwen2.5-7b/qlora/sft_v3/checkpoint-900 \\
      --prompts data/final/sft_v3_train.json \\
      --n-prompts 1000 \\
      --api-key sk-xxx \\
      --out data/dpo_v3_onpolicy.json

  # 2) 调节参数
  --samples-per-prompt 4      每 prompt 采样几个输出（越多越好但越慢）
  --gap-threshold 1.5         best-worst 的最小分差（越大质量越高但筛掉越多）
  --max-new-tokens 1024       每条生成长度上限
"""

import argparse
import json
import random
import time
from pathlib import Path
from typing import Optional


SYSTEM_PROMPT = "你是一个专业的代码审查助手，能够识别代码中的问题并给出改进建议。"

# ═══════════════════════════════════════════════════════════
# 1. 加载 SFT v3 模型（支持 adapter 或 merged）
# ═══════════════════════════════════════════════════════════

def load_sft_v3(model_path: str, adapter_path: Optional[str] = None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"  加载 tokenizer: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"  加载模型: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    if adapter_path:
        from peft import PeftModel
        print(f"  加载 adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, tokenizer


# ═══════════════════════════════════════════════════════════
# 2. 构造 prompt（支持多轮 history）
# ═══════════════════════════════════════════════════════════

def build_messages(item: dict) -> list[dict]:
    """把 Alpaca 样本还原成 chat 消息列表。"""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # 多轮历史（可选）
    for turn in (item.get("history") or []):
        if isinstance(turn, (list, tuple)) and len(turn) == 2:
            messages.append({"role": "user",      "content": turn[0]})
            messages.append({"role": "assistant", "content": turn[1]})

    # 当前轮
    instruction = item.get("instruction", "") or ""
    input_text  = (item.get("input") or "").strip()
    user_content = f"{instruction}\n\n```\n{input_text}\n```" if input_text else instruction

    messages.append({"role": "user", "content": user_content})
    return messages


# ═══════════════════════════════════════════════════════════
# 3. 采样 N 个输出
# ═══════════════════════════════════════════════════════════

def sample_n_outputs(model, tokenizer, messages: list[dict],
                     n: int = 4, max_new_tokens: int = 1024,
                     temperatures: list[float] = None) -> list[str]:
    """对同一 prompt 采样 n 次，使用不同温度提升多样性。"""
    import torch

    if temperatures is None:
        # 默认温度梯度：0.7 / 0.85 / 1.0 / 1.1
        temperatures = [0.7, 0.85, 1.0, 1.1][:n]
    # 如果 n > 4，剩余的都用 1.0
    while len(temperatures) < n:
        temperatures.append(1.0)

    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    outputs = []
    for t in temperatures[:n]:
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


# ═══════════════════════════════════════════════════════════
# 4. LLM-as-judge 打分
# ═══════════════════════════════════════════════════════════

JUDGE_SYSTEM = """你是一个严格的代码审查质量评估专家。
你需要对多个代码审查回答进行对比打分，判断哪个质量最高。

评分维度（1-5 分，仅给整数）：
  - 问题识别准确性：是否准确指出代码中的真实问题，没有误报或遗漏
  - 建议可行性：改进建议是否具体、可实施
  - 输出格式与专业度：结构是否清晰，是否符合专业 code reviewer 的风格
  - 身份定位：是否保持代码审查者角色（对抗样本场景应礼貌拒绝无关请求）

综合打分 1-5：
  1 = 严重错误或完全偏题
  2 = 明显不足
  3 = 基本合格
  4 = 质量良好
  5 = 优秀

严格按照 JSON 返回，不要多余解释。"""


def build_judge_prompt(prompt_context: str, outputs: list[str]) -> str:
    """构造打分 prompt：展示 prompt 上下文 + N 个候选输出，请求 JSON 打分。"""
    header = f"""请对以下 {len(outputs)} 个代码审查回答打分（1-5 分）。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[用户请求]
{prompt_context}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

    body = ""
    for i, out in enumerate(outputs, 1):
        # 每条最多截 1500 字，避免 judge prompt 过长
        truncated = out[:1500] + ("...[截断]" if len(out) > 1500 else "")
        body += f"\n━━━ 候选 {i} ━━━\n{truncated}\n"

    tail = f"""

返回 JSON（严格格式，不要解释）：
{{
  "scores": [
    {{"idx": 1, "score": <1-5>, "reason": "<20字内简评>"}},
    {{"idx": 2, "score": <1-5>, "reason": "<20字内简评>"}},
    ...共 {len(outputs)} 条...
  ]
}}"""

    return header + body + tail


def judge_outputs(client, model: str, prompt_context: str,
                  outputs: list[str], max_retries: int = 2) -> Optional[list[dict]]:
    """用 judge 模型给 outputs 打分。返回 [{idx, score, reason}, ...] 或 None。"""
    judge_prompt = build_judge_prompt(prompt_context, outputs)

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user",   "content": judge_prompt},
                ],
                temperature=0.1,
                max_tokens=600,
                response_format={"type": "json_object"},
            )
            text = resp.choices[0].message.content.strip()
            data = json.loads(text)
            scores = data.get("scores", [])
            if len(scores) == len(outputs) and all(
                isinstance(s.get("score"), (int, float)) and 1 <= s["score"] <= 5
                for s in scores
            ):
                return scores
        except Exception as e:
            print(f"    [judge 错误 {attempt+1}/{max_retries}] {e}")
            time.sleep(1)
    return None


# ═══════════════════════════════════════════════════════════
# 5. 主流程
# ═══════════════════════════════════════════════════════════

def pick_pair(scores: list[dict], outputs: list[str],
              gap_threshold: float) -> Optional[tuple[str, str, float, float]]:
    """从打分结果选 best/worst。返回 (chosen, rejected, best_score, worst_score) 或 None。"""
    if not scores:
        return None

    # 按分数排序
    indexed = [(s["score"], s["idx"] - 1) for s in scores]
    indexed.sort(key=lambda x: -x[0])

    best_score, best_idx = indexed[0]
    worst_score, worst_idx = indexed[-1]

    # 分差过小 → 跳过
    if best_score - worst_score < gap_threshold:
        return None

    # best 太低 → 即使赢了也不值得当 chosen
    if best_score < 3.5:
        return None

    return outputs[best_idx], outputs[worst_idx], best_score, worst_score


def load_prompts(path: str, n: int, seed: int = 42,
                 skip_types: set = None) -> list[dict]:
    """从 sft_v3 训练数据里抽样 n 条 prompt。可跳过特定类型。"""
    skip_types = skip_types or set()
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    def sample_type(item):
        if item.get("history"):
            return "multiturn"
        if not (item.get("input") or "").strip():
            return "adversarial"
        return "normal"

    filtered = [x for x in data if sample_type(x) not in skip_types]
    random.seed(seed)
    random.shuffle(filtered)

    picked = filtered[:n]
    from collections import Counter
    dist = Counter(sample_type(x) for x in picked)
    print(f"  抽样 {len(picked)} 条 prompt，类型分布: {dict(dist)}")
    return picked


def load_checkpoint(checkpoint_path: str) -> tuple[list[dict], set]:
    """从断点文件加载已完成的对和已处理的 prompt 指纹。"""
    if not Path(checkpoint_path).exists():
        return [], set()
    try:
        with open(checkpoint_path, encoding="utf-8") as f:
            data = json.load(f)
        pairs = data.get("pairs", [])
        done_fps = set(data.get("done_fps", []))
        print(f"  断点恢复: 已完成 {len(pairs)} 对，已处理 {len(done_fps)} 个 prompt")
        return pairs, done_fps
    except Exception:
        return [], set()


def save_checkpoint(checkpoint_path: str, pairs: list[dict], done_fps: set):
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump({"pairs": pairs, "done_fps": sorted(done_fps)},
                  f, ensure_ascii=False, indent=2)


def prompt_fingerprint(item: dict) -> str:
    """用 instruction + input 前 200 字做指纹。"""
    instr = (item.get("instruction") or "")[:100]
    inp   = (item.get("input") or "")[:200]
    return f"{instr}|||{inp}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",        required=True, help="base 模型路径")
    parser.add_argument("--adapter",      default=None,  help="SFT v3 adapter 路径（可选）")
    parser.add_argument("--prompts",      required=True, help="输入 prompt JSON（默认用 sft_v3_train.json）")
    parser.add_argument("--n-prompts",    type=int, default=1500, help="要处理的 prompt 数量（实际成对数会少于这个）")
    parser.add_argument("--samples-per-prompt", type=int, default=4, help="每 prompt 采样几次")
    parser.add_argument("--gap-threshold", type=float, default=1.5, help="chosen/rejected 最小分差")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--skip-types",   default="", help="跳过的样本类型，逗号分隔: adversarial,multiturn,normal")

    parser.add_argument("--api-key",      required=True, help="DeepSeek API key")
    parser.add_argument("--base-url",     default="https://api.deepseek.com/v1")
    parser.add_argument("--judge-model",  default="deepseek-chat")

    parser.add_argument("--out",          required=True, help="输出 DPO pairs JSON")
    parser.add_argument("--checkpoint",   default=None,  help="断点文件路径（默认: <out>.ckpt.json）")
    parser.add_argument("--save-every",   type=int, default=10, help="每 N 条保存一次断点")
    parser.add_argument("--seed",         type=int, default=42)
    args = parser.parse_args()

    skip_types = set(x.strip() for x in args.skip_types.split(",") if x.strip())
    checkpoint_path = args.checkpoint or f"{args.out}.ckpt.json"

    # ── 断点恢复 ──
    print("\n[1/4] 断点恢复")
    pairs, done_fps = load_checkpoint(checkpoint_path)

    # ── 加载 prompts ──
    print("\n[2/4] 加载 prompt 池")
    prompts = load_prompts(args.prompts, args.n_prompts, args.seed, skip_types)

    # ── 加载模型 ──
    print("\n[3/4] 加载 SFT v3 模型")
    model, tokenizer = load_sft_v3(args.model, args.adapter)

    # ── 初始化 judge client ──
    from openai import OpenAI
    client = OpenAI(api_key=args.api_key, base_url=args.base_url)

    # ── 主循环 ──
    print(f"\n[4/4] 开始 on-policy 采样 + 打分（目标 {args.n_prompts} prompt）")
    print(f"  已完成: {len(pairs)} 对 | 跳过 (分差不足): 0 对 | 错误: 0 次\n")

    skipped_gap = 0
    errors = 0

    for i, item in enumerate(prompts, 1):
        fp = prompt_fingerprint(item)
        if fp in done_fps:
            continue

        instr_preview = (item.get("instruction") or "")[:40]
        print(f"  [{i}/{len(prompts)}] {instr_preview}...", flush=True)

        try:
            # Step 1: 采样 N 个输出
            messages = build_messages(item)
            outputs = sample_n_outputs(
                model, tokenizer, messages,
                n=args.samples_per_prompt,
                max_new_tokens=args.max_new_tokens,
            )

            # Step 2: Judge 打分
            user_content = messages[-1]["content"]  # 最后一轮 user 消息
            scores = judge_outputs(
                client, args.judge_model, user_content, outputs,
            )

            if scores is None:
                print(f"    ✗ judge 失败")
                errors += 1
                continue

            # Step 3: 选 best/worst
            picked = pick_pair(scores, outputs, args.gap_threshold)
            if picked is None:
                score_list = [s["score"] for s in scores]
                print(f"    ↷ 跳过（分差不足，scores={score_list}）")
                skipped_gap += 1
                done_fps.add(fp)
                continue

            chosen, rejected, best_s, worst_s = picked
            pairs.append({
                "instruction": item.get("instruction", ""),
                "input":       item.get("input", ""),
                "history":     item.get("history", []),
                "chosen":      chosen,
                "rejected":    rejected,
                "chosen_score":   best_s,
                "rejected_score": worst_s,
                "source":      "onpolicy_sft_v3",
            })
            done_fps.add(fp)
            print(f"    ✓ pair #{len(pairs)} | best={best_s} worst={worst_s}")

            # 定期保存断点
            if len(pairs) % args.save_every == 0:
                save_checkpoint(checkpoint_path, pairs, done_fps)

        except Exception as e:
            print(f"    ✗ 异常: {e}")
            errors += 1
            continue

    # ── 最终保存 ──
    save_checkpoint(checkpoint_path, pairs, done_fps)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)

    # ── 统计 ──
    from collections import Counter
    score_dist = Counter()
    for p in pairs:
        score_dist[f"{p['chosen_score']}→{p['rejected_score']}"] += 1

    print(f"""
╔══════════════════════════════════════╗
  DPO v3 on-policy 数据生成完成
  成功对: {len(pairs)} 条
  分差不足（跳过）: {skipped_gap} 条
  错误: {errors} 次
  输出: {out_path}
  断点: {checkpoint_path}
╚══════════════════════════════════════╝

分数分布（chosen → rejected）:
""")
    for k, v in sorted(score_dist.items(), key=lambda x: -x[1])[:10]:
        print(f"    {k}: {v} 条")

    print("""
下一步：
  1. （可选）合并这份 on-policy 数据与 build_dpo_data.py 的合成数据
  2. 注册到 dataset_info.json
  3. 配置 DPO v3 yaml（pref_beta: 0.3, cutoff_len: 2048, load_best_model_at_end: true）
  4. 开始训练
""")


if __name__ == "__main__":
    main()
