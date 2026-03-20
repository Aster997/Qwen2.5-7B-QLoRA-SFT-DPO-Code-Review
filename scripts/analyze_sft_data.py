"""
SFT 数据集质量分析脚本
用途：诊断过拟合的数据原因，量化重复率、多样性、分布等指标
用法：python scripts/analyze_sft_data.py --data data/my_sft_data.json
"""

import json
import re
import argparse
from collections import Counter
from difflib import SequenceMatcher


# ─────────────────────────────────────────────
# 1. 工具函数
# ─────────────────────────────────────────────

def load_data(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize(text: str) -> str:
    """去掉空白和标点，用于粗粒度去重"""
    return re.sub(r"\s+", " ", text.strip().lower())


def similarity(a: str, b: str) -> float:
    """两段文本的相似度（0~1）"""
    return SequenceMatcher(None, a[:500], b[:500]).ratio()


def detect_language(code: str) -> str:
    """从代码片段粗略判断编程语言"""
    patterns = {
        "Python":     [r"\bdef \w+\(", r"\bimport \w+", r"\bclass \w+:", r":\s*\n\s+"],
        "Go":         [r"\bfunc \w+\(", r"\bpackage \w+", r":=", r"\bfmt\."],
        "JavaScript": [r"\bconst \w+\s*=", r"\bfunction \w+\(", r"=>\s*\{", r"\.then\("],
        "TypeScript": [r": \w+\[\]", r"interface \w+", r": string", r": number"],
        "Java":       [r"\bpublic \w+ \w+\(", r"\bimport java\.", r"System\.out"],
        "SQL":        [r"\bSELECT\b", r"\bINSERT\b", r"\bUPDATE\b", r"\bFROM\b"],
        "Shell":      [r"\$\(", r"\becho\b", r"\bif \[", r"#!/bin/bash"],
    }
    scores = {lang: 0 for lang in patterns}
    for lang, pats in patterns.items():
        for pat in pats:
            if re.search(pat, code, re.IGNORECASE):
                scores[lang] += 1
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "Other"


def detect_problem_category(output: str) -> list[str]:
    """从 output 文本中识别审查问题类别"""
    categories = {
        "安全":   ["SQL注入", "XSS", "注入", "密钥", "权限", "认证", "路径遍历", "CSRF", "泄漏", "硬编码"],
        "错误处理": ["异常", "错误", "except", "error", "panic", "崩溃", "忽略", "泄漏"],
        "性能":   ["性能", "慢", "阻塞", "O(n)", "循环", "缓存", "索引", "并发", "连接池"],
        "资源泄漏": ["未关闭", "泄漏", "连接", "句柄", "with语句", "finally"],
        "设计":   ["设计", "职责", "耦合", "重复", "魔法数字", "可读性", "重构"],
        "并发":   ["竞态", "死锁", "线程", "协程", "同步", "原子", "Semaphore"],
        "空指针":  ["None", "null", "NullPointer", "空值", "不存在"],
    }
    found = []
    for cat, keywords in categories.items():
        if any(kw in output for kw in keywords):
            found.append(cat)
    return found if found else ["其他"]


# ─────────────────────────────────────────────
# 2. 各项分析函数
# ─────────────────────────────────────────────

def analyze_dedup(data: list[dict]) -> dict:
    """分析 input 重复率"""
    total = len(data)
    inputs = [normalize(item["input"]) for item in data]

    # 精确去重（完全相同）
    exact_unique = len(set(inputs))

    # 找出重复的 input（出现2次以上）
    counter = Counter(inputs)
    duplicates = {k: v for k, v in counter.items() if v > 1}

    # 找出重复组：同一 input 配了不同 instruction
    input_to_instructions = {}
    for item in data:
        key = normalize(item["input"])
        inst = item["instruction"]
        input_to_instructions.setdefault(key, []).append(inst)

    multi_instruction = {
        k: v for k, v in input_to_instructions.items()
        if len(set(v)) > 1
    }

    return {
        "total": total,
        "exact_unique_inputs": exact_unique,
        "duplicate_ratio": round(1 - exact_unique / total, 3),
        "inputs_with_multiple_instructions": len(multi_instruction),
        "top_duplicates": sorted(duplicates.items(), key=lambda x: -x[1])[:5],
    }


def analyze_instructions(data: list[dict]) -> dict:
    """分析 instruction 多样性"""
    instructions = [item["instruction"] for item in data]
    counter = Counter(instructions)
    unique_count = len(counter)

    # 找出高频指令
    top_instructions = counter.most_common(15)

    # 粗分类指令语义
    semantic_groups = {
        "全面审查":  ["全面", "代码评审", "code review", "审查"],
        "反模式":    ["反模式", "anti-pattern"],
        "安全审计":  ["安全", "security", "漏洞", "审计"],
        "上线检查":  ["上线", "生产", "部署"],
        "最佳实践":  ["最佳实践", "不符合", "改进"],
        "bug查找":   ["bug", "错误", "问题", "风险"],
        "可读性":    ["可读", "可维护", "维护性"],
        "Tech Lead": ["tech lead", "lead"],
    }
    group_counts = {}
    for item in data:
        inst_lower = item["instruction"].lower()
        matched = False
        for group, keywords in semantic_groups.items():
            if any(kw in inst_lower for kw in keywords):
                group_counts[group] = group_counts.get(group, 0) + 1
                matched = True
                break
        if not matched:
            group_counts["其他"] = group_counts.get("其他", 0) + 1

    return {
        "total_instructions": len(instructions),
        "unique_instructions": unique_count,
        "instruction_diversity_ratio": round(unique_count / len(instructions), 3),
        "top_instructions": top_instructions,
        "semantic_groups": group_counts,
    }


def analyze_output(data: list[dict]) -> dict:
    """分析 output 的模板化程度和长度分布"""
    outputs = [item["output"] for item in data]
    lengths = [len(o) for o in outputs]

    # 检查固定模板特征
    template_markers = {
        "含「代码审查报告」标题": sum(1 for o in outputs if "代码审查报告" in o),
        "含「问题分析」章节":     sum(1 for o in outputs if "问题分析" in o),
        "含「改进代码」章节":     sum(1 for o in outputs if "改进代码" in o),
        "含「最佳实践」章节":     sum(1 for o in outputs if "最佳实践" in o),
        "含「[严重]」标签":       sum(1 for o in outputs if "[严重]" in o),
        "含「[中等]」标签":       sum(1 for o in outputs if "[中等]" in o),
        "含 emoji (审查图标)":     sum(1 for o in outputs if "\U0001f50d" in o),
    }

    # 长度分布
    def bucket(l):
        if l < 300:   return "短(<300字)"
        if l < 800:   return "中(300-800字)"
        if l < 1500:  return "长(800-1500字)"
        return "很长(>1500字)"

    length_dist = Counter(bucket(l) for l in lengths)

    # output 是否包含代码块
    has_code_block = sum(1 for o in outputs if "```" in o)

    return {
        "avg_length": round(sum(lengths) / len(lengths)),
        "min_length": min(lengths),
        "max_length": max(lengths),
        "length_distribution": dict(length_dist),
        "template_markers": template_markers,
        "template_ratio": round(
            template_markers["含「代码审查报告」标题"] / len(outputs), 3
        ),
        "has_code_block_ratio": round(has_code_block / len(outputs), 3),
    }


def analyze_languages(data: list[dict]) -> dict:
    """分析代码语言分布"""
    lang_counter = Counter(detect_language(item["input"]) for item in data)
    total = len(data)
    return {
        lang: {"count": cnt, "ratio": round(cnt / total, 3)}
        for lang, cnt in lang_counter.most_common()
    }


def analyze_problem_categories(data: list[dict]) -> dict:
    """分析审查问题类别分布"""
    cat_counter = Counter()
    for item in data:
        for cat in detect_problem_category(item["output"]):
            cat_counter[cat] += 1
    total = len(data)
    return {
        cat: {"count": cnt, "ratio": round(cnt / total, 3)}
        for cat, cnt in cat_counter.most_common()
    }


def find_near_duplicates(data: list[dict], threshold: float = 0.85, sample_size: int = 200) -> list:
    """
    抽样检测近似重复（相似度 > threshold 的 input 对）
    注意：全量比较是 O(n^2)，大数据集只抽样检测
    """
    import random
    sample = random.sample(data, min(sample_size, len(data)))
    near_dups = []
    for i in range(len(sample)):
        for j in range(i + 1, len(sample)):
            sim = similarity(sample[i]["input"], sample[j]["input"])
            if sim > threshold:
                near_dups.append({
                    "similarity": round(sim, 3),
                    "input_a": sample[i]["input"][:80].replace("\n", " "),
                    "input_b": sample[j]["input"][:80].replace("\n", " "),
                    "instruction_a": sample[i]["instruction"],
                    "instruction_b": sample[j]["instruction"],
                })
    return sorted(near_dups, key=lambda x: -x["similarity"])[:10]


# ─────────────────────────────────────────────
# 3. 报告输出
# ─────────────────────────────────────────────

def print_section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_report(data: list[dict], near_dup_sample: int = 200):
    print("\n" + "█" * 60)
    print("  SFT 数据集质量诊断报告")
    print("█" * 60)

    # ── 1. 重复分析 ──
    print_section("1. 数据重复分析")
    dedup = analyze_dedup(data)
    print(f"  总样本数         : {dedup['total']}")
    print(f"  唯一 input 数    : {dedup['exact_unique_inputs']}")
    print(f"  重复率           : {dedup['duplicate_ratio']*100:.1f}%  ", end="")
    ratio = dedup['duplicate_ratio']
    if ratio > 0.3:
        print("【严重】模型会过拟合")
    elif ratio > 0.1:
        print("【警告】建议去重")
    else:
        print("【正常】")
    print(f"  同一代码配多种指令: {dedup['inputs_with_multiple_instructions']} 个")
    print()
    print("  重复最多的 input（前5条）：")
    for raw_input, count in dedup["top_duplicates"]:
        preview = raw_input[:60].replace("\n", " ")
        print(f"    × {count} | {preview}...")

    # ── 2. Instruction 分析 ──
    print_section("2. Instruction 多样性分析")
    inst = analyze_instructions(data)
    print(f"  总 instruction 数  : {inst['total_instructions']}")
    print(f"  唯一 instruction 数: {inst['unique_instructions']}")
    print(f"  多样性比           : {inst['instruction_diversity_ratio']*100:.1f}%  ", end="")
    div = inst['instruction_diversity_ratio']
    if div < 0.01:
        print("【严重】指令极度重复")
    elif div < 0.05:
        print("【警告】指令变化很少")
    else:
        print("【尚可】")
    print()
    print("  高频 instruction（前15条）：")
    for inst_text, cnt in inst["top_instructions"]:
        print(f"    {cnt:>4}×  {inst_text}")
    print()
    print("  语义分组分布：")
    for group, cnt in sorted(inst["semantic_groups"].items(), key=lambda x: -x[1]):
        bar = "▓" * (cnt * 30 // inst["total_instructions"])
        print(f"    {group:<10} {cnt:>5}条  {bar}")

    # ── 3. Output 分析 ──
    print_section("3. Output 模板化分析")
    out = analyze_output(data)
    print(f"  平均长度       : {out['avg_length']} 字符")
    print(f"  最短 / 最长    : {out['min_length']} / {out['max_length']} 字符")
    print(f"  含代码块比例   : {out['has_code_block_ratio']*100:.0f}%")
    print(f"  固定模板比例   : {out['template_ratio']*100:.0f}%  ", end="")
    if out['template_ratio'] > 0.8:
        print("【警告】输出高度模板化，缺乏多样性")
    else:
        print("【正常】")
    print()
    print("  模板特征命中率：")
    for marker, cnt in out["template_markers"].items():
        pct = cnt / len(data) * 100
        print(f"    {marker:<20} {pct:>5.1f}%")
    print()
    print("  长度分布：")
    for bucket, cnt in sorted(out["length_distribution"].items()):
        bar = "▓" * (cnt * 30 // len(data))
        print(f"    {bucket:<15} {cnt:>5}条  {bar}")

    # ── 4. 语言分布 ──
    print_section("4. 编程语言分布")
    langs = analyze_languages(data)
    for lang, info in langs.items():
        bar = "▓" * int(info["ratio"] * 40)
        print(f"  {lang:<12} {info['count']:>5}条  {info['ratio']*100:>5.1f}%  {bar}")

    # ── 5. 问题类别分布 ──
    print_section("5. 审查问题类别分布")
    cats = analyze_problem_categories(data)
    for cat, info in cats.items():
        bar = "▓" * int(info["ratio"] * 30)
        print(f"  {cat:<8} {info['count']:>5}条  {info['ratio']*100:>5.1f}%  {bar}")
    # 检查是否有"正确代码"样本
    good_code = sum(
        1 for item in data
        if any(kw in item["output"] for kw in ["没有明显", "写得很好", "代码质量良好", "no issues", "整体良好"])
    )
    print(f"\n  含「代码正确/无问题」样本: {good_code} 条  ", end="")
    if good_code == 0:
        print("【警告】模型会学到「任何代码都有问题」的偏见")
    else:
        print()

    # ── 6. 近似重复 ──
    print_section(f"6. 近似重复检测（抽样 {near_dup_sample} 条，阈值 85%）")
    near_dups = find_near_duplicates(data, threshold=0.85, sample_size=near_dup_sample)
    if not near_dups:
        print("  未检测到近似重复样本（抽样范围内）")
    else:
        print(f"  发现 {len(near_dups)} 对近似重复：")
        for nd in near_dups[:5]:
            print(f"\n  相似度 {nd['similarity']*100:.0f}%")
            print(f"    A [{nd['instruction_a']}] {nd['input_a']}...")
            print(f"    B [{nd['instruction_b']}] {nd['input_b']}...")

    # ── 总结 ──
    print_section("总结与建议")
    issues = []
    if dedup['duplicate_ratio'] > 0.1:
        issues.append(f"[重复] input 重复率 {dedup['duplicate_ratio']*100:.0f}%，需先去重")
    if inst['instruction_diversity_ratio'] < 0.02:
        issues.append(f"[指令] 仅 {inst['unique_instructions']} 种唯一指令，需扩充至 50+ 种")
    if out['template_ratio'] > 0.8:
        issues.append(f"[输出] {out['template_ratio']*100:.0f}% 的样本使用固定模板，需增加输出多样性")
    if good_code == 0:
        issues.append("[偏见] 无「代码正确」样本，建议加入 10% 此类数据")

    lang_vals = list(langs.values())
    if lang_vals and lang_vals[0]["ratio"] > 0.5:
        top_lang = list(langs.keys())[0]
        issues.append(f"[语言] {top_lang} 占比过高（{lang_vals[0]['ratio']*100:.0f}%），需平衡多语言")

    if issues:
        print("  发现以下问题：")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
    else:
        print("  数据质量良好，无明显问题。")

    print()


# ─────────────────────────────────────────────
# 4. 入口
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SFT 数据集质量分析")
    parser.add_argument("--data", default="data/my_sft_data.json", help="数据集路径")
    parser.add_argument("--sample", type=int, default=300, help="近似重复检测的抽样量")
    parser.add_argument("--out", default="data_analysis_report.txt", help="报告输出文件（默认输出到文件）")
    args = parser.parse_args()

    data = load_data(args.data)

    import io, sys
    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf

    print(f"\n加载数据集: {args.data}")
    print(f"加载完成，共 {len(data)} 条样本")
    print_report(data, near_dup_sample=args.sample)

    sys.stdout = old_stdout
    report_text = buf.getvalue()

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"报告已保存到: {args.out}")


if __name__ == "__main__":
    main()
