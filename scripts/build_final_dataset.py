"""
最终数据集构建脚本

功能：
  1. 加载多个来源的过滤后数据，做近似去重清洗
  2. 兼容三种样本结构：普通 / 对抗样本（input 为空）/ 多轮样本（含 history）
  3. 划分 训练集 / 验证集 / 测试集
     - 测试集保证：每种主要语言 + 代码正确 + 对抗 + 多轮 都有覆盖
  4. 写入 LLaMA-Factory 的 dataset_info.json（不写显式 columns 映射，避免覆盖默认 alpaca 字段）

用法：
  python scripts/build_final_dataset.py \\
      --main         data/sft_main_filtered.json \\
      --correct      data/sft_correct_filtered.json \\
      --adversarial  data/sft_adversarial_filtered.json \\
      --multiturn    data/sft_multiturn_filtered.json \\
      --github       data/sft_github.json \\
      --out-dir      data/final \\
      --prefix       sft_v3 \\
      --test-size    60

可选：合并旧的手动收集集
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
# 0. 样本类型工具
# ──────────────────────────────────────────────

def sample_type(item: dict) -> str:
    """识别样本类型：multiturn / adversarial / normal"""
    if item.get("history"):
        return "multiturn"
    if not item.get("input"):
        return "adversarial"
    return "normal"


def dedup_text(item: dict) -> str:
    """
    抽取用于去重的文本：
    - 多轮: history 第一轮 user（含代码）
    - 对抗: instruction（input 为空）
    - 普通: input 代码
    """
    t = sample_type(item)
    if t == "multiturn":
        return item["history"][0][0]
    if t == "adversarial":
        return item.get("instruction", "")
    return item["input"]


# ──────────────────────────────────────────────
# 1. 近似去重
# ──────────────────────────────────────────────

def fingerprint(text: str) -> str:
    return hashlib.md5(" ".join(text.split()).encode()).hexdigest()


def similarity(a: str, b: str) -> float:
    # 只比较前600字符，提高速度
    return SequenceMatcher(None, a[:600], b[:600]).ratio()


def dedup(data: list[dict], sim_threshold: float = 0.90) -> tuple[list[dict], int]:
    """
    两步去重（按样本类型分桶，桶内去重，避免对抗样本和普通代码互相比较）：
      1. 精确去重（MD5）
      2. 近似去重（SequenceMatcher）
    返回 (去重后数据, 删除条数)
    """
    # 按类型分桶
    buckets: dict[str, list[dict]] = {"normal": [], "adversarial": [], "multiturn": []}
    for item in data:
        buckets[sample_type(item)].append(item)

    all_kept: list[dict] = []
    total_exact = 0
    total_near = 0

    for type_name, items in buckets.items():
        if not items:
            continue

        # 精确去重
        seen_fps = set()
        exact_clean = []
        for item in items:
            fp = fingerprint(dedup_text(item))
            if fp not in seen_fps:
                seen_fps.add(fp)
                exact_clean.append(item)
        exact_removed = len(items) - len(exact_clean)
        total_exact += exact_removed

        # 近似去重
        kept = []
        kept_texts = []
        near_removed = 0
        for item in exact_clean:
            text = dedup_text(item)
            is_dup = False
            for existing in kept_texts[-200:]:   # 只和最近 200 条比
                if similarity(text, existing) >= sim_threshold:
                    is_dup = True
                    break
            if not is_dup:
                kept.append(item)
                kept_texts.append(text)
            else:
                near_removed += 1
        total_near += near_removed

        all_kept.extend(kept)
        print(f"  [{type_name}] 输入 {len(items)} → 精确删 {exact_removed} → 近似删 {near_removed} → 保留 {len(kept)}")

    total_removed = total_exact + total_near
    print(f"  小计：精确 {total_exact} + 近似 {total_near} = {total_removed} 条删除")
    print(f"  去重后剩余: {len(all_kept)} 条")
    return all_kept, total_removed


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
    if not code:
        return "N/A"
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


def build_test_set(data: list[dict], test_size: int = 60) -> tuple[list[dict], list[dict]]:
    """
    构建测试集，保证覆盖：
      - 每种主要语言（普通样本）有最低配额
      - 代码正确样本 ≥ 5
      - 对抗样本 ≥ 6（必须有，否则边界场景没法测）
      - 多轮样本 ≥ 4
    剩余作为训练+验证集
    """
    random.shuffle(data)

    test: list[dict] = []
    remaining = list(data)

    # ── 类型配额 ──
    lang_quota = {"Python": 8, "Go": 6, "JavaScript": 6, "Java": 4, "SQL": 3, "TypeScript": 3}
    correct_quota = 5
    adversarial_quota = 6
    multiturn_quota = 4

    lang_filled = {lang: 0 for lang in lang_quota}
    correct_filled = 0
    adv_filled = 0
    mt_filled = 0

    # 先按配额选
    for item in data:
        if len(test) >= test_size:
            break
        st = sample_type(item)
        selected = False

        if st == "multiturn":
            if mt_filled < multiturn_quota:
                test.append(item)
                mt_filled += 1
                selected = True

        elif st == "adversarial":
            if adv_filled < adversarial_quota:
                test.append(item)
                adv_filled += 1
                selected = True

        else:  # normal
            is_correct = is_correct_code_sample(item)
            lang = detect_language(item["input"])
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

    # 如果还没到 test_size，从剩余中按类型均衡补齐
    shortfall = test_size - len(test)
    if shortfall > 0:
        extra = random.sample(remaining, min(shortfall, len(remaining)))
        test.extend(extra)
        for item in extra:
            remaining.remove(item)

    # ── 打印分布 ──
    print(f"\n  测试集构成（共{len(test)}条）：")
    type_dist = Counter(sample_type(item) for item in test)
    for t, cnt in type_dist.most_common():
        print(f"    [{t:<12}] {cnt} 条")
    normal_items = [item for item in test if sample_type(item) == "normal"]
    if normal_items:
        lang_dist = Counter(detect_language(item["input"]) for item in normal_items)
        print(f"  普通样本语言分布：")
        for lang, cnt in lang_dist.most_common():
            print(f"    {lang:<12} {cnt} 条")
    correct_in_test = sum(1 for item in normal_items if is_correct_code_sample(item))
    print(f"  代码正确样本: {correct_in_test} 条")

    return test, remaining


# ──────────────────────────────────────────────
# 5. 写入 dataset_info.json
# ──────────────────────────────────────────────

def register_datasets(dataset_info_path: str, datasets: dict):
    """
    把生成的训练/验证/测试集注册到 LLaMA-Factory 的 dataset_info.json。

    注意：不写显式 columns 映射，让 alpaca 默认字段生效
    （instruction / input / output / history 都能被自动识别），
    否则多轮样本的 history 字段会被忽略。

    datasets = {
        "sft_v3_train": {"file": "final/sft_v3_train.json"},
        ...
    }
    """
    with open(dataset_info_path, "r", encoding="utf-8") as f:
        info = json.load(f)

    for name, meta in datasets.items():
        info[name] = {"file_name": meta["file"]}
        print(f"  注册数据集: {name} → {meta['file']}")

    with open(dataset_info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)


# ──────────────────────────────────────────────
# 6. 主流程
# ──────────────────────────────────────────────

def _load_source(path: Optional[str], label: str) -> list[dict]:
    """加载一个来源文件，不存在则返回空列表。"""
    if not path:
        return []
    if not Path(path).exists():
        print(f"  [{label:<12}] 跳过（文件不存在: {path}）")
        return []
    with open(path, "r", encoding="utf-8") as f:
        items = json.load(f)
    print(f"  [{label:<12}] {len(items):>5} 条  ←  {path}")
    return items


def main():
    parser = argparse.ArgumentParser(description="构建最终训练数据集（多来源整合 + 切分）")

    # ── 数据来源（至少要有 main） ──
    parser.add_argument("--main",         required=True,
                        help="主集（普通+困难混合，DeepSeek 生成 + filter 过滤后的文件）")
    parser.add_argument("--correct",      default=None,
                        help="代码正确集（filter 过滤后的文件）")
    parser.add_argument("--adversarial",  default=None,
                        help="对抗样本集（filter 过滤后的文件）")
    parser.add_argument("--multiturn",    default=None,
                        help="多轮追问集（filter 过滤后的文件）")
    parser.add_argument("--github",       default=None,
                        help="GitHub 真实代码集（scrape_github.py 生成）")
    parser.add_argument("--source-a",     default=None,
                        help="（可选）旧的手动收集集，向后兼容")

    # ── 切分参数 ──
    parser.add_argument("--test-size",    type=int, default=60,  help="测试集大小（默认 60）")
    parser.add_argument("--val-ratio",    type=float, default=0.05, help="验证集比例（默认 0.05）")
    parser.add_argument("--sim-threshold", type=float, default=0.90,
                        help="近似去重阈值，默认 0.90")

    # ── 输出 ──
    parser.add_argument("--out-dir",      default="data/final",  help="输出目录")
    parser.add_argument("--prefix",       default="sft_v3", help="输出文件名前缀")
    parser.add_argument("--seed",         type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # ── [1/5] 加载所有来源 ──
    print("\n[1/5] 加载数据源")
    sources = [
        ("main",        args.main),
        ("correct",     args.correct),
        ("adversarial", args.adversarial),
        ("multiturn",   args.multiturn),
        ("github",      args.github),
        ("source-a",    args.source_a),
    ]
    data: list[dict] = []
    for label, path in sources:
        data.extend(_load_source(path, label))
    print(f"  ─────────────────────")
    print(f"  合并总量    : {len(data):>5} 条")

    if not data:
        print("[错误] 没有任何数据加载，退出")
        return

    # ── [2/5] 按类型分桶去重 ──
    print(f"\n[2/5] 按类型分桶去重（阈值 {args.sim_threshold*100:.0f}%）")
    data, removed = dedup(data, sim_threshold=args.sim_threshold)

    # ── [3/5] 类型分布统计 ──
    type_dist = Counter(sample_type(item) for item in data)
    correct_count = sum(
        1 for item in data
        if sample_type(item) == "normal" and is_correct_code_sample(item)
    )
    print(f"\n[3/5] 去重后类型分布")
    print(f"  普通      : {type_dist.get('normal', 0):>5} 条（其中代码正确 ~{correct_count} 条）")
    print(f"  对抗      : {type_dist.get('adversarial', 0):>5} 条")
    print(f"  多轮      : {type_dist.get('multiturn', 0):>5} 条")

    # 健康检查
    if type_dist.get("adversarial", 0) < 20:
        print(f"  [警告] 对抗样本不足 20 条，边界测试可能失败")
    if type_dist.get("multiturn", 0) < 20:
        print(f"  [警告] 多轮样本不足 20 条")
    if correct_count < 50:
        print(f"  [警告] 代码正确样本不足 50 条，模型可能仍偏向'挑刺'")

    # ── [4/5] 划分 train / val / test ──
    print("\n[4/5] 划分数据集")
    random.shuffle(data)
    test_set, trainval = build_test_set(data, test_size=args.test_size)

    val_size  = max(1, int(len(trainval) * args.val_ratio))
    val_set   = trainval[:val_size]
    train_set = trainval[val_size:]

    print(f"\n  训练集: {len(train_set)} 条")
    print(f"  验证集: {len(val_set)} 条")
    print(f"  测试集: {len(test_set)} 条")
    print(f"  合计  : {len(train_set)+len(val_set)+len(test_set)} 条")

    # ── [5/5] 保存 + 注册 ──
    print("\n[5/5] 保存文件")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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
    # dataset_info.json 在 data/ 下，file_name 字段是相对 data/ 的相对路径
    import os
    dataset_info_path = Path("data") / "dataset_info.json"
    if dataset_info_path.exists():
        print(f"\n  注册到 {dataset_info_path}")
        try:
            rel_dir_str = os.path.relpath(out_dir, "data").replace("\\", "/")
        except ValueError:
            rel_dir_str = str(out_dir).replace("\\", "/")

        register_datasets(
            str(dataset_info_path),
            {
                name: {"file": f"{rel_dir_str}/{filename}" if rel_dir_str != "." else filename}
                for name, (_, filename) in files.items()
            },
        )
    else:
        print(f"  [提示] {dataset_info_path} 不存在，跳过注册")

    # ── 最终报告 ──
    print(f"""
╔══════════════════════════════════════╗
  数据集构建完成
  训练集  {len(train_set):>5} 条  →  {args.prefix}_train.json
  验证集  {len(val_set):>5} 条  →  {args.prefix}_val.json
  测试集  {len(test_set):>5} 条  →  {args.prefix}_test.json
╚══════════════════════════════════════╝

下一步：更新训练配置 yaml 中的 dataset 字段：
  dataset: {args.prefix}_train,alpaca_zh_demo,identity
  eval_dataset: {args.prefix}_val
  val_size: 0.0   # 已单独划分
""")


if __name__ == "__main__":
    main()
