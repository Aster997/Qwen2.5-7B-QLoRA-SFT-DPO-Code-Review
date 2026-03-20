"""
最终数据集构建脚本

功能：
  1. 加载生成的数据（来源B），做近似去重清洗
  2. 检查并补充「代码正确」样本（如不足则调 API 补生成）
  3. 可选：合并来源A（手动收集的真实数据）
  4. 划分 训练集 / 验证集 / 测试集（测试集保证场景覆盖）
  5. 写入 LLaMA-Factory 的 dataset_info.json

用法：
  # 只清洗，不补充代码正确样本
  python scripts/build_final_dataset.py --input data/sft_generated.json

  # 清洗 + 调 API 补充代码正确样本到100条
  python scripts/build_final_dataset.py --input data/sft_generated.json \\
      --fill-correct --api-key <YOUR_DEEPSEEK_API_KEY> --base-url https://api.deepseek.com/v1

  # 清洗 + 合并来源A
  python scripts/build_final_dataset.py --input data/sft_generated.json \\
      --source-a data/sft_source_a.json
"""

import json
import random
import argparse
import hashlib
import time
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional


# ──────────────────────────────────────────────
# 1. 近似去重
# ──────────────────────────────────────────────

def fingerprint(code: str) -> str:
    return hashlib.md5(" ".join(code.split()).encode()).hexdigest()


def similarity(a: str, b: str) -> float:
    # 只比较前600字符，提高速度
    return SequenceMatcher(None, a[:600], b[:600]).ratio()


def dedup(data: list[dict], sim_threshold: float = 0.90) -> tuple[list[dict], int]:
    """
    两步去重：
      1. 精确去重（MD5）
      2. 近似去重（SequenceMatcher，O(n^2) 但数据量小可接受）
    返回 (去重后数据, 删除条数)
    """
    # 精确去重
    seen_fps = set()
    exact_clean = []
    for item in data:
        fp = fingerprint(item["input"])
        if fp not in seen_fps:
            seen_fps.add(fp)
            exact_clean.append(item)
    exact_removed = len(data) - len(exact_clean)

    # 近似去重
    kept = []
    kept_inputs = []
    near_removed = 0
    for item in exact_clean:
        code = item["input"]
        is_dup = False
        for existing in kept_inputs[-200:]:   # 只和最近200条比，避免O(n^2)过慢
            if similarity(code, existing) >= sim_threshold:
                is_dup = True
                break
        if not is_dup:
            kept.append(item)
            kept_inputs.append(code)
        else:
            near_removed += 1

    total_removed = exact_removed + near_removed
    print(f"  精确去重删除: {exact_removed} 条")
    print(f"  近似去重删除: {near_removed} 条（阈值 {sim_threshold*100:.0f}%）")
    print(f"  去重后剩余  : {len(kept)} 条")
    return kept, total_removed


# ──────────────────────────────────────────────
# 2. 检测「代码正确」样本
# ──────────────────────────────────────────────

# 基于实际DeepSeek生成样本校准的关键词
# 核心信号：锦上添花、可以直接使用、整体设计合理（无[严重]/[中等]时）
CORRECT_CODE_KEYWORDS = [
    "锦上添花",           # 强信号：建议只是"画蛇添足"
    "可以直接使用",        # 强信号：代码可以直接用
    "没有发现明显问题",
    "没有明显问题",
    "代码写得很好",
    "代码质量良好",
    "代码质量不错",
    "整体写得",
    "no issues found",
    "well-written code",
    "looks good overall",
    "没有发现问题",
]

def is_correct_code_sample(item: dict) -> bool:
    output = item["output"].lower()
    return any(kw.lower() in output for kw in CORRECT_CODE_KEYWORDS)


def count_correct_samples(data: list[dict]) -> int:
    return sum(1 for item in data if is_correct_code_sample(item))


# ──────────────────────────────────────────────
# 3. 补充「代码正确」样本
# ──────────────────────────────────────────────

CORRECT_CODE_INSTRUCTIONS = [
    "请对这段代码进行全面的代码评审。",
    "这段代码准备上线，请帮忙做 Code Review。",
    "帮我看看这段代码写得怎么样。",
    "这段代码有问题吗？",
    "请审查并评估这段代码的质量。",
    "你是一名资深工程师，请帮我审查这段代码。",
    "请像 Tech Lead 一样审查这段代码。",
    "这是我重构后的代码，质量有提升吗？",
    "Perform a code review on this code.",
    "帮我 review 一下，谢谢。",
]

# 「代码正确」场景的具体约束
CORRECT_CODE_SCENARIOS = [
    ("Python", "使用 bcrypt 存储用户密码，参数化查询防SQL注入"),
    ("Python", "带重试和超时的 HTTP 客户端封装"),
    ("Python", "用 contextmanager 管理数据库连接，含事务回滚"),
    ("Go",     "使用 errgroup 并发请求并处理错误"),
    ("Go",     "带 graceful shutdown 的 HTTP 服务器"),
    ("Go",     "使用 sync.Once 的线程安全单例"),
    ("JavaScript", "使用 DOMPurify 防 XSS 的用户输入处理"),
    ("JavaScript", "带错误处理和重试的 fetch 封装"),
    ("TypeScript", "带类型保护的 API 响应解析"),
    ("Java",   "使用 try-with-resources 管理数据库连接"),
    ("Java",   "不可变值对象的正确实现"),
    ("SQL",    "使用窗口函数的分页查询（游标分页）"),
    ("Python", "使用 asyncio.Semaphore 控制并发的异步爬虫"),
    ("Go",     "带超时和取消的 context 传递"),
    ("Python", "类型安全的配置文件读取，含验证"),
]

CORRECT_CODE_SYSTEM_PROMPT = """你是一个专业的代码生成助手，负责生成用于训练代码审查 AI 的数据。
每次生成一个完整的训练样本，严格以 JSON 格式返回，包含三个字段：
- instruction: 用户的提问
- input: 代码片段（写得很好的代码）
- output: 代码审查回答（肯定优点，最多提1-2个细微改进建议）

只返回 JSON，不要任何额外解释"""


def generate_correct_sample(client, model: str, instruction: str, lang: str, scenario: str) -> Optional[dict]:
    prompt = f"""生成一个「代码写得好」的代码审查训练样本：

约束：
- 编程语言：{lang}
- 场景：{scenario}
- 代码要求：正确处理了边界情况、资源管理、错误处理，遵循最佳实践
- 代码行数：20-40行

用户指令（instruction 字段原文）："{instruction}"

output 要求：
- 肯定代码的优点，说明哪里写得好、为什么
- 最多提1-2个无关紧要的细微改进（如"可以加一行注释"），不是必须的
- 语气积极，不要挑剔
- 不要用"没有问题"这种空洞表述，要具体说好在哪里
- 长度100-300字即可，不需要完整的Markdown模板

返回格式（严格 JSON）：
{{"instruction": "...", "input": "...", "output": "..."}}"""

    try:
        from openai import OpenAI
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": CORRECT_CODE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.8,
            max_tokens=1500,
            response_format={"type": "json_object"},
        )
        text = response.choices[0].message.content.strip()
        # 尝试提取JSON
        try:
            sample = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                sample = json.loads(match.group())
            else:
                return None

        if all(k in sample for k in ("instruction", "input", "output")):
            if len(sample["input"]) > 50 and len(sample["output"]) > 30:
                return sample
    except Exception as e:
        print(f"    [错误] {e}")
    return None


def fill_correct_samples(
    data: list[dict],
    target_count: int,
    client,
    model: str,
) -> list[dict]:
    """补充代码正确样本到 target_count 条"""
    current = count_correct_samples(data)
    needed = target_count - current
    print(f"\n  当前代码正确样本: {current} 条，目标: {target_count} 条，需补充: {needed} 条")

    if needed <= 0:
        print("  无需补充")
        return data

    added = 0
    scenarios = CORRECT_CODE_SCENARIOS.copy()
    random.shuffle(scenarios)
    scenario_cycle = scenarios * (needed // len(scenarios) + 2)

    for i in range(needed * 3):  # 最多尝试3倍次数
        if added >= needed:
            break
        lang, scenario = scenario_cycle[i % len(scenario_cycle)]
        instruction = random.choice(CORRECT_CODE_INSTRUCTIONS)
        print(f"  [{added+1}/{needed}] 生成代码正确样本: {lang} | {scenario[:20]}...")
        sample = generate_correct_sample(client, model, instruction, lang, scenario)
        if sample:
            data.append(sample)
            added += 1
            print(f"    ✓ 成功")
        else:
            print(f"    ✗ 失败，跳过")
        time.sleep(0.3)

    print(f"  补充完成，新增 {added} 条代码正确样本")
    return data


# ──────────────────────────────────────────────
# 4. 测试集构建（保证场景覆盖）
# ──────────────────────────────────────────────

def detect_language(code: str) -> str:
    patterns = {
        "Python":     [r"\bdef \w+\(", r"\bimport \w+", r"\bclass \w+:"],
        "Go":         [r"\bfunc \w+\(", r":=", r"\bfmt\."],
        "JavaScript": [r"\bconst \w+\s*=", r"=>\s*\{", r"\.then\("],
        "TypeScript": [r": \w+\[\]", r"interface \w+"],
        "Java":       [r"\bpublic \w+ \w+\(", r"\bimport java\."],
        "SQL":        [r"\bSELECT\b", r"\bFROM\b"],
    }
    scores = {lang: sum(1 for p in pats if re.search(p, code)) for lang, pats in patterns.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "Other"


def build_test_set(data: list[dict], test_size: int = 50) -> tuple[list[dict], list[dict]]:
    """
    构建测试集，保证覆盖：每种主要语言 ≥ 4 条，代码正确样本 ≥ 5 条
    剩余作为训练+验证集
    """
    random.shuffle(data)

    test = []
    remaining = list(data)

    # 目标覆盖
    lang_quota = {"Python": 8, "Go": 7, "JavaScript": 7, "Java": 5, "SQL": 4, "TypeScript": 4}
    correct_quota = 5
    lang_filled = {lang: 0 for lang in lang_quota}
    correct_filled = 0

    # 先按配额选
    for item in data:
        if len(test) >= test_size:
            break
        lang = detect_language(item["input"])
        is_correct = is_correct_code_sample(item)

        selected = False
        if is_correct and correct_filled < correct_quota:
            test.append(item)
            correct_filled += 1
            selected = True
        elif lang in lang_quota and lang_filled.get(lang, 0) < lang_quota[lang]:
            test.append(item)
            lang_filled[lang] = lang_filled.get(lang, 0) + 1
            selected = True

        if selected:
            remaining.remove(item)

    # 如果还没到 test_size，随机补齐
    shortfall = test_size - len(test)
    if shortfall > 0:
        extra = random.sample(remaining, min(shortfall, len(remaining)))
        test.extend(extra)
        for item in extra:
            remaining.remove(item)

    print(f"\n  测试集构成（共{len(test)}条）：")
    lang_dist = Counter(detect_language(item["input"]) for item in test)
    for lang, cnt in lang_dist.most_common():
        print(f"    {lang:<12} {cnt} 条")
    correct_in_test = count_correct_samples(test)
    print(f"    代码正确样本: {correct_in_test} 条")

    return test, remaining


# ──────────────────────────────────────────────
# 5. 写入 dataset_info.json
# ──────────────────────────────────────────────

def register_datasets(dataset_info_path: str, datasets: dict):
    """
    datasets = {
        "my_sft_train_v2": {"file": "sft_train_v2.json", "desc": "..."},
        ...
    }
    """
    with open(dataset_info_path, "r", encoding="utf-8") as f:
        info = json.load(f)

    for name, meta in datasets.items():
        info[name] = {
            "file_name": meta["file"],
            "columns": {"prompt": "instruction", "query": "input", "response": "output"},
        }
        print(f"  注册数据集: {name} → {meta['file']}")

    with open(dataset_info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)


# ──────────────────────────────────────────────
# 6. 主流程
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="构建最终训练数据集")
    parser.add_argument("--input",        required=True,  help="来源B生成数据路径")
    parser.add_argument("--source-a",     default=None,   help="来源A数据路径（可选）")
    parser.add_argument("--source-c",     default=None,   help="来源C数据路径（可选）")
    parser.add_argument("--fill-correct", action="store_true", help="是否调API补充代码正确样本")
    parser.add_argument("--correct-target", type=int, default=80, help="目标代码正确样本数（默认80）")
    parser.add_argument("--api-key",      default=None)
    parser.add_argument("--base-url",     default="https://api.deepseek.com/v1")
    parser.add_argument("--model",        default="deepseek-chat")
    parser.add_argument("--test-size",    type=int, default=50,  help="测试集大小")
    parser.add_argument("--val-ratio",    type=float, default=0.07, help="验证集比例")
    parser.add_argument("--out-dir",      default="data",  help="输出目录")
    parser.add_argument("--prefix",       default="sft_v2", help="输出文件名前缀")
    parser.add_argument("--seed",         type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # ── 加载数据 ──
    print("\n[1/6] 加载数据")
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"  来源B: {len(data)} 条")

    if args.source_a:
        with open(args.source_a, "r", encoding="utf-8") as f:
            source_a = json.load(f)
        data.extend(source_a)
        print(f"  来源A: {len(source_a)} 条")

    if args.source_c:
        with open(args.source_c, "r", encoding="utf-8") as f:
            source_c = json.load(f)
        data.extend(source_c)
        print(f"  来源C: {len(source_c)} 条")

    print(f"  合并后总量: {len(data)} 条")

    # ── 去重 ──
    print("\n[2/6] 近似去重（阈值 90%）")
    data, removed = dedup(data, sim_threshold=0.90)

    # ── 补充代码正确样本 ──
    print(f"\n[3/6] 检查「代码正确」样本")
    current_correct = count_correct_samples(data)
    print(f"  当前检测到: {current_correct} 条")

    if args.fill_correct and args.api_key:
        from openai import OpenAI
        client = OpenAI(api_key=args.api_key, base_url=args.base_url)
        data = fill_correct_samples(data, args.correct_target, client, args.model)
    elif current_correct < 20:
        print(f"  [提示] 代码正确样本不足，建议加 --fill-correct --api-key YOUR_KEY 补充")
    else:
        print(f"  代码正确样本充足，跳过补充")

    # ── 随机打乱 ──
    random.shuffle(data)
    print(f"\n[4/6] 总数据量: {len(data)} 条")

    # ── 划分测试集 ──
    print("\n[5/6] 划分数据集")
    test_set, trainval = build_test_set(data, test_size=args.test_size)

    val_size  = max(1, int(len(trainval) * args.val_ratio))
    val_set   = trainval[:val_size]
    train_set = trainval[val_size:]

    print(f"\n  训练集: {len(train_set)} 条")
    print(f"  验证集: {len(val_set)} 条")
    print(f"  测试集: {len(test_set)} 条")
    print(f"  合计  : {len(train_set)+len(val_set)+len(test_set)} 条")

    # ── 保存文件 ──
    print("\n[6/6] 保存文件")
    out_dir = Path(args.out_dir)

    files = {
        f"{args.prefix}_train": (train_set, f"{args.prefix}_train.json"),
        f"{args.prefix}_val":   (val_set,   f"{args.prefix}_val.json"),
        f"{args.prefix}_test":  (test_set,  f"{args.prefix}_test.json"),
    }

    for name, (split, filename) in files.items():
        path = out_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(split, f, ensure_ascii=False, indent=2)
        print(f"  保存: {path}  ({len(split)} 条)")

    # ── 注册到 dataset_info.json ──
    dataset_info_path = out_dir / "dataset_info.json"
    if dataset_info_path.exists():
        register_datasets(
            str(dataset_info_path),
            {name: {"file": filename} for name, (_, filename) in files.items()},
        )

    # ── 最终报告 ──
    print(f"""
╔══════════════════════════════════════╗
  数据集构建完成
  训练集  {len(train_set):>5} 条  →  {args.prefix}_train.json
  验证集  {len(val_set):>5} 条  →  {args.prefix}_val.json
  测试集  {len(test_set):>5} 条  →  {args.prefix}_test.json
╚══════════════════════════════════════╝

下一步：更新训练配置 yaml 文件中的 dataset 字段：
  dataset: {args.prefix}_train
  val_size: 0   # 已单独划分，不需要再自动划分
""")


if __name__ == "__main__":
    main()
