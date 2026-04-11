"""
SFT 数据质量过滤脚本（critic）

用 deepseek-chat 当 critic，对每条 (instruction, input, output) 样本打 4 个维度分（1-5），
并给一个总评。总评 ≤3 的样本会被淘汰。

设计要点：
  - **批量评分**：一次 API 调用评 5 条样本，复用 system prompt，省约 30% 输入费
  - **断点续跑**：评分结果增量保存，重跑会跳过已评分的样本
  - **三类样本特殊处理**：
      普通样本   → 评 4 维度
      多轮样本   → 评连贯性 + 准确性
      对抗样本   → 评是否成功拒绝 + 引导话术质量

用法：
  python scripts/filter_sft_data.py \\
      --api-key sk-xxx \\
      --base-url https://api.deepseek.com/v1 \\
      --model deepseek-chat \\
      --in  data/sft_main.json \\
      --out data/sft_main_filtered.json \\
      --report data/sft_main_report.json \\
      --threshold 3 \\
      --batch-size 5
"""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Optional

try:
    from openai import OpenAI
except ImportError:
    raise ImportError("请先安装 openai 库：pip install openai")


# ══════════════════════════════════════════════
# 1. Critic Prompt
# ══════════════════════════════════════════════

CRITIC_SYSTEM_PROMPT = """你是一个严格的训练数据质检员，负责评估"代码审查助手"训练样本的质量。

你会收到多条样本，每条包含：
- instruction: 用户的指令
- input: 待审查的代码（可能为空）
- output: 助手的回答
- (可选) history: 多轮对话历史

请对每条样本从以下维度打分（1-5 分，1 最差，5 最优），然后给一个 1-5 的总评：

【普通代码审查样本】
1. 准确性：output 指出的问题是否真实存在于 input 代码里？
2. 完整性：是否漏掉了 input 中明显的问题？
3. 无幻觉：output 是否引用了 input 中根本不存在的函数/变量/字段？
4. 改进质量：output 给出的"改进代码"是否语法正确、确实修复了问题？

【对抗样本】（input 通常为空，instruction 是非代码审查请求）
1. 拒绝得体：是否礼貌但坚定地拒绝了非代码审查请求？
2. 不被诱导：是否避免了角色扮演/泄露 prompt/生成恶意代码等陷阱？
3. 引导有效：是否主动把话题拉回了代码审查？
4. 简洁度：回答是否在 80-150 字范围内，没有冗余说教？

【多轮追问样本】（含 history 字段）
1. 连贯性：output 是否引用了 history 中的内容，没有答非所问？
2. 准确性：回答是否技术正确？
3. 上下文利用：是否真的理解了用户基于第一轮的追问意图？
4. 完整性：回答是否充分？

总评打分参考：
- 5 分：质量极高，可直接进训练集
- 4 分：质量好，有小瑕疵但不影响训练
- 3 分：质量一般，建议淘汰（边界）
- 2 分：明显问题（幻觉/逻辑错/答非所问）
- 1 分：完全错误，必须淘汰

严格按 JSON 数组返回，每条样本一个对象，按输入顺序：
{"scores": [
  {"id": 1, "type": "普通"|"对抗"|"多轮", "d1": 5, "d2": 4, "d3": 5, "d4": 4, "overall": 4, "reason": "一句话理由"},
  {"id": 2, ...}
]}

只返回 JSON，不要任何额外文字。"""


def build_batch_prompt(batch: list[dict]) -> str:
    """把 N 条样本拼成一个评分请求。"""
    parts = ["请对以下 {} 条样本打分：\n".format(len(batch))]
    for i, item in enumerate(batch, 1):
        parts.append(f"━━━━━━━ 样本 {i} ━━━━━━━")
        # 多轮样本特殊渲染
        if item.get("history"):
            parts.append("【类型】多轮追问")
            for turn_idx, (u, a) in enumerate(item["history"], 1):
                parts.append(f"[历史第{turn_idx}轮 user]: {u}")
                parts.append(f"[历史第{turn_idx}轮 assistant]: {a}")
            parts.append(f"[本轮 instruction]: {item['instruction']}")
            if item.get("input"):
                parts.append(f"[本轮 input]: {item['input']}")
            parts.append(f"[本轮 output]: {item['output']}")
        else:
            # 通过 input 是否为空粗略判断对抗样本
            sample_type = "对抗" if not item.get("input") else "普通"
            parts.append(f"【类型】{sample_type}")
            parts.append(f"[instruction]: {item['instruction']}")
            parts.append(f"[input]: {item.get('input', '')}")
            parts.append(f"[output]: {item['output']}")
        parts.append("")
    parts.append(
        "请按 JSON 格式返回 scores 数组，id 从 1 到 {} 对应上面的顺序。".format(len(batch))
    )
    return "\n".join(parts)


# ══════════════════════════════════════════════
# 2. JSON 提取
# ══════════════════════════════════════════════

def extract_json(text: str) -> Optional[dict]:
    """从模型输出中提取 JSON。"""
    import re
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    m = re.search(r"(\{.*\})", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    return None


# ══════════════════════════════════════════════
# 3. 样本指纹（用于断点续跑去重）
# ══════════════════════════════════════════════

def sample_fingerprint(item: dict) -> str:
    """根据样本内容生成稳定指纹，用于跳过已评分的样本。"""
    if item.get("history"):
        text = item["history"][0][0] + "||" + item.get("instruction", "") + "||" + item.get("output", "")
    else:
        text = item.get("instruction", "") + "||" + item.get("input", "") + "||" + item.get("output", "")
    return hashlib.md5(text.encode("utf-8")).hexdigest()


# ══════════════════════════════════════════════
# 4. API 调用（带重试）
# ══════════════════════════════════════════════

def call_critic(
    client: OpenAI,
    model: str,
    batch: list[dict],
    use_json_mode: bool = True,
    max_retries: int = 3,
) -> Optional[list[dict]]:
    """调用 critic API 给一批样本打分，返回 scores 列表。"""
    prompt = build_batch_prompt(batch)
    for attempt in range(1, max_retries + 1):
        try:
            kwargs = dict(
                model=model,
                messages=[
                    {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,  # critic 要稳定
                max_tokens=2048,
            )
            if use_json_mode:
                kwargs["response_format"] = {"type": "json_object"}

            response = client.chat.completions.create(**kwargs)
            text = response.choices[0].message.content.strip()
            data = extract_json(text)

            if data is None or "scores" not in data:
                print(f"    [警告] 第{attempt}次：无法解析 critic 返回，重试...")
                continue

            scores = data["scores"]
            if not isinstance(scores, list) or len(scores) != len(batch):
                print(f"    [警告] 第{attempt}次：scores 长度 {len(scores) if isinstance(scores, list) else '?'} != batch {len(batch)}，重试...")
                continue

            # 校验每条 score 都有 overall 字段
            ok = True
            for s in scores:
                if "overall" not in s or not isinstance(s["overall"], (int, float)):
                    ok = False
                    break
            if not ok:
                print(f"    [警告] 第{attempt}次：scores 缺少 overall 字段，重试...")
                continue

            return scores

        except Exception as e:
            err = str(e)
            if use_json_mode and ("response_format" in err or "json_object" in err):
                print(f"    [提示] 模型不支持 json_object，自动降级")
                use_json_mode = False
                continue
            print(f"    [警告] 第{attempt}次：API 错误 {e}，等待重试...")
            time.sleep(2 ** attempt)

    return None


# ══════════════════════════════════════════════
# 5. 主流程
# ══════════════════════════════════════════════

def filter_dataset(
    client: OpenAI,
    model: str,
    in_path: str,
    out_path: str,
    report_path: str,
    threshold: float,
    batch_size: int,
    use_json_mode: bool = True,
):
    # 加载原始数据
    with open(in_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    print(f"[加载] 原始数据 {len(raw_data)} 条")

    # 加载已有评分（断点续跑）
    scores_cache: dict[str, dict] = {}
    if os.path.exists(report_path):
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            for entry in cached.get("entries", []):
                scores_cache[entry["fingerprint"]] = entry
            print(f"[恢复] 已加载 {len(scores_cache)} 条历史评分")
        except Exception as e:
            print(f"[警告] 报告文件解析失败，从头评分: {e}")

    # 找出还需要评分的样本
    pending: list[tuple[int, dict]] = []
    for idx, item in enumerate(raw_data):
        fp = sample_fingerprint(item)
        if fp not in scores_cache:
            pending.append((idx, item))

    print(f"[待评] 还需评分 {len(pending)} 条 / 共 {len(raw_data)} 条\n")

    # 批量评分
    total_batches = (len(pending) + batch_size - 1) // batch_size
    batch_no = 0

    for start in range(0, len(pending), batch_size):
        batch_no += 1
        chunk = pending[start : start + batch_size]
        items = [item for _, item in chunk]

        print(f"[{batch_no}/{total_batches}] 评分 {len(items)} 条...")
        scores = call_critic(client, model, items, use_json_mode=use_json_mode)

        if scores is None:
            print(f"    ✗ 该批次评分失败，跳过")
            continue

        for (orig_idx, item), score in zip(chunk, scores):
            fp = sample_fingerprint(item)
            scores_cache[fp] = {
                "fingerprint": fp,
                "orig_idx": orig_idx,
                "type": score.get("type", "未知"),
                "d1": score.get("d1"),
                "d2": score.get("d2"),
                "d3": score.get("d3"),
                "d4": score.get("d4"),
                "overall": score.get("overall"),
                "reason": score.get("reason", ""),
            }

        # 增量保存评分报告
        if batch_no % 5 == 0:
            _save_report(report_path, scores_cache)
            print(f"    → 已保存评分报告（{len(scores_cache)} 条）")

    # 最终保存评分报告
    _save_report(report_path, scores_cache)

    # 按 threshold 过滤
    kept = []
    dropped = []
    for idx, item in enumerate(raw_data):
        fp = sample_fingerprint(item)
        entry = scores_cache.get(fp)
        if entry is None:
            print(f"[跳过] 样本 {idx} 未评分（API 失败），保留")
            kept.append(item)
            continue
        overall = entry.get("overall")
        if overall is None or overall <= threshold:
            dropped.append({"orig_idx": idx, "overall": overall, "reason": entry.get("reason", "")})
        else:
            kept.append(item)

    # 保存过滤后的数据
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(kept, f, ensure_ascii=False, indent=2)

    # 打印统计
    print(f"\n{'='*50}")
    print(f"过滤完成")
    print(f"  原始    : {len(raw_data)} 条")
    print(f"  保留    : {len(kept)} 条 ({len(kept)/len(raw_data)*100:.1f}%)")
    print(f"  淘汰    : {len(dropped)} 条 ({len(dropped)/len(raw_data)*100:.1f}%)")
    print(f"  阈值    : overall > {threshold}")
    print(f"  输出    : {out_path}")
    print(f"  报告    : {report_path}")
    print(f"{'='*50}")

    # 分数分布
    overall_dist = {}
    for entry in scores_cache.values():
        o = entry.get("overall")
        if o is None:
            continue
        bucket = int(o)
        overall_dist[bucket] = overall_dist.get(bucket, 0) + 1
    print("\n总评分布：")
    for k in sorted(overall_dist.keys()):
        print(f"  {k} 分: {overall_dist[k]} 条")


def _save_report(path: str, scores_cache: dict):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"entries": list(scores_cache.values())}, f, ensure_ascii=False, indent=2)


# ══════════════════════════════════════════════
# 6. 入口
# ══════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="SFT 数据质量过滤脚本（critic）")
    parser.add_argument("--api-key",       required=True, help="API Key")
    parser.add_argument("--base-url",      default="https://api.deepseek.com/v1", help="API Base URL")
    parser.add_argument("--model",         default="deepseek-chat", help="Critic 模型，推荐 deepseek-chat (V3)")
    parser.add_argument("--in",            dest="in_path", required=True, help="输入数据文件")
    parser.add_argument("--out",           required=True, help="过滤后数据输出路径")
    parser.add_argument("--report",        required=True, help="评分报告输出路径（JSON）")
    parser.add_argument("--threshold",     type=float, default=3.0,
                        help="淘汰阈值：overall <= threshold 的样本被淘汰，默认 3")
    parser.add_argument("--batch-size",    type=int, default=5, help="每次 API 调用评分的样本数")
    parser.add_argument("--no-json-mode",  action="store_true", help="禁用 response_format json_object")
    args = parser.parse_args()

    use_json_mode = not args.no_json_mode

    print(f"模型     : {args.model}")
    print(f"接口     : {args.base_url}")
    print(f"输入     : {args.in_path}")
    print(f"输出     : {args.out}")
    print(f"报告     : {args.report}")
    print(f"阈值     : overall > {args.threshold}")
    print(f"批大小   : {args.batch_size}")
    print(f"JSON模式 : {'关闭' if not use_json_mode else '开启'}\n")

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)

    filter_dataset(
        client=client,
        model=args.model,
        in_path=args.in_path,
        out_path=args.out,
        report_path=args.report,
        threshold=args.threshold,
        batch_size=args.batch_size,
        use_json_mode=use_json_mode,
    )


if __name__ == "__main__":
    main()
