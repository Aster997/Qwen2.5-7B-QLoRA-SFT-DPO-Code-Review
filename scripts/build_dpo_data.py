"""
DPO 数据构建脚本

来源1：从 SFT 评测结果提取（reference=chosen, 模型输出=rejected）
        用 --exclude-txt 排除人工精读集，保持三模型对比公平
来源2：DeepSeek 合成生成，针对三大弱点

用法：
  # 提取（排除人工精读集）+ 合成生成（推荐）
  python scripts/build_dpo_data.py \
      --eval-json data/eval_results_sft_pro_full.json \
      --exclude-txt data/eval_manual_samples.txt \
      --api-key <YOUR_DEEPSEEK_API_KEY> \
      --synth-target 360 \
      --out data/dpo_data.json
"""

import argparse
import json
import random
import re
import time
from pathlib import Path
from typing import Optional


# ──────────────────────────────────────────────
# 1. 从评测结果提取 DPO 对
# ──────────────────────────────────────────────

def load_exclude_set(txt_path: str) -> set[str]:
    """从 eval_manual_samples.txt 提取代码内容，用于排除比对（取前200字符作为指纹）"""
    with open(txt_path, encoding="utf-8") as f:
        content = f.read().replace("\r\n", "\n").replace("\r", "\n")

    exclude = set()
    blocks = re.split(r"={50,}", content)
    for block in blocks:
        m = re.search(r"\[Input - 待审查代码\]\n(.*?)\n\[标准答案", block, re.DOTALL)
        if m:
            code = m.group(1).strip()
            exclude.add(code[:200])  # 前200字符作为指纹
    print(f"  排除集加载: {len(exclude)} 条（来自人工精读集）")
    return exclude


def extract_from_eval(eval_json_path: str,
                      exclude_txt_path: str = None) -> list[dict]:
    """
    从 batch_eval.py 生成的 JSON 中提取 DPO 对。
    reference = chosen，模型预测 = rejected。
    exclude_txt_path: 人工精读集 txt 路径，其中的样本会被排除。
    """
    with open(eval_json_path, encoding="utf-8") as f:
        data = json.load(f)

    exclude_set = set()
    if exclude_txt_path:
        exclude_set = load_exclude_set(exclude_txt_path)

    pairs = []
    skipped_exclude = 0
    skipped_quality = 0

    for item in data:
        # 排除人工精读集样本（用代码前200字符匹配）
        code_fp = item.get("input", "")[:200]
        if code_fp in exclude_set:
            skipped_exclude += 1
            continue

        chosen   = item.get("reference", "").strip()
        rejected = item.get("prediction", "").strip()

        if not chosen or not rejected:
            skipped_quality += 1
            continue

        # 过滤条件：chosen 明显比 rejected 长且完整
        if len(chosen) < 100 or len(rejected) < 50:
            skipped_quality += 1
            continue
        if len(chosen) < len(rejected) * 1.2:
            skipped_quality += 1
            continue

        pairs.append({
            "instruction": item["instruction"],
            "input":       item["input"],
            "chosen":      chosen,
            "rejected":    rejected,
            "source":      "eval_extract",
        })

    print(f"  排除人工精读集: {skipped_exclude} 条")
    print(f"  质量过滤跳过:   {skipped_quality} 条")
    print(f"  最终提取:       {len(pairs)} 对")
    return pairs


# ──────────────────────────────────────────────
# 2. 合成 DPO 数据配置
# ──────────────────────────────────────────────

# ── 针对弱点A：安全漏洞识别 ──
SECURITY_SCENARIOS = [
    ("Python",     "使用用户输入拼接 SQL 查询字符串", "SQL注入"),
    ("Python",     "将用户上传的文件路径直接传给 open()", "路径遍历"),
    ("Python",     "将密钥硬编码在源码中并提交到代码仓库", "密钥泄露"),
    ("JavaScript", "直接将用户输入插入 innerHTML", "XSS注入"),
    ("JavaScript", "OAuth回调中未验证 state 参数直接作为路径", "路径遍历"),
    ("Go",         "WebSocket Upgrader 的 CheckOrigin 始终返回 true", "CSRF风险"),
    ("Go",         "使用 fmt.Sprintf 拼接 SQL 语句", "SQL注入"),
    ("Java",       "反序列化来自不可信来源的数据", "反序列化漏洞"),
    ("Java",       "日志中记录了完整的用户密码", "敏感信息泄露"),
    ("TypeScript", "使用 eval() 执行用户提供的字符串", "代码注入"),
    ("Python",     "使用 MD5 存储用户密码", "弱哈希算法"),
    ("Go",         "HTTP 接口未校验 Authorization header", "未授权访问"),
    ("JavaScript", "将 JWT secret 硬编码为短字符串", "弱密钥"),
    ("Python",     "使用 pickle 反序列化用户上传的文件", "远程代码执行"),
    ("Java",       "XML 解析器未禁用外部实体", "XXE注入"),
]

SECURITY_REJECTED_PATTERNS = [
    "只提到代码风格问题，完全没有发现安全漏洞",
    "只提到缺少日志记录，没有识别出安全风险",
    "只建议添加注释，忽略了明显的安全问题",
    "将安全漏洞误判为性能问题",
    "给出了正面评价，但漏掉了严重安全隐患",
]

# ── 针对弱点B：代码正确样本 ──
CORRECT_CODE_SCENARIOS = [
    ("Python",     "使用 bcrypt + salt 正确存储密码", "安全"),
    ("Go",         "带 context 和 timeout 的 HTTP 客户端", "资源管理"),
    ("Java",       "使用 try-with-resources 管理数据库连接", "资源管理"),
    ("TypeScript", "使用 discriminated union 做类型安全的状态机", "类型安全"),
    ("Python",     "使用 dataclasses + slots 的不可变值对象", "设计"),
    ("Go",         "使用 errgroup 并发请求并正确聚合错误", "并发"),
    ("JavaScript", "使用 DOMPurify 处理用户输入防 XSS", "安全"),
    ("Python",     "使用 contextmanager 管理事务并正确回滚", "事务"),
    ("Java",       "线程安全的单例模式（双重检查锁）", "并发"),
    ("SQL",        "使用窗口函数实现游标分页", "性能"),
]

CORRECT_CODE_REJECTED_PATTERNS = [
    "无中生有，把好代码说成有严重问题",
    "对正确的最佳实践提出不必要的修改建议",
    "误判了代码的设计意图，给出了反向建议",
    "过度批评代码风格，忽视了代码在安全和性能上的优点",
]

# ── 针对弱点C：复杂代码多问题识别 ──
COMPLEX_SCENARIOS = [
    ("Python",     "混合了SQL注入+资源泄漏+没有超时的爬虫代码", ["SQL注入", "资源泄漏", "缺少超时"]),
    ("Go",         "混合了竞态条件+错误吞咽+panic滥用的并发服务", ["竞态条件", "错误处理", "panic"]),
    ("JavaScript", "混合了XSS+硬编码密钥+无错误处理的前端代码", ["XSS", "密钥暴露", "错误处理"]),
    ("Java",       "混合了线程不安全+连接泄漏+空指针的业务代码", ["线程安全", "资源泄漏", "空指针"]),
    ("Python",     "混合了性能问题+类型错误+日志不当的数据处理", ["N+1查询", "类型错误", "日志"]),
    ("TypeScript", "混合了类型不安全+异步竞态+内存泄漏的React组件", ["类型安全", "竞态", "内存泄漏"]),
]

COMPLEX_REJECTED_PATTERNS = [
    "只发现了3个问题中的1个，遗漏了更重要的安全问题",
    "发现了表面问题但漏掉了根本性的架构缺陷",
    "只提到了最明显的问题，完全没有分析潜在的并发风险",
]

# ── 针对弱点D：SQL 代码误判 ──
SQL_SCENARIOS = [
    ("SQL", "缺少索引导致全表扫描的慢查询", "缺少索引", "硬编码了表名或列名"),
    ("SQL", "使用字符串拼接构造 SQL 的注入漏洞", "SQL注入", "只提到查询性能问题"),
    ("SQL", "N+1 查询问题（循环内单条查询）", "N+1查询", "指出了变量命名不规范"),
    ("SQL", "重复子查询可以用 WITH/CTE 优化", "DRY/可维护性", "误判为SQL注入风险"),
    ("SQL", "缺少事务导致数据不一致的批量更新", "缺少事务", "只提到性能优化"),
    ("SQL", "SELECT * 导致不必要的数据传输", "查询优化", "说代码逻辑有错误"),
    ("SQL", "游标分页性能差，应改用 keyset 分页", "分页性能", "误判为权限控制问题"),
    ("SQL", "GROUP BY 后 HAVING 过滤可提前用 WHERE", "查询效率", "只提到代码风格"),
]

SQL_REJECTED_PATTERNS = [
    "把查询性能问题误判为安全漏洞",
    "识别了代码风格问题，完全忽略了真正的性能瓶颈",
    "把正确的 SQL 写法说成有逻辑错误",
    "把 DRY 问题误判为 SQL 注入风险",
    "只提到硬编码，没有发现真正的索引缺失或 N+1 问题",
]

# ── 针对弱点E：并发误诊 ──
CONCURRENCY_SCENARIOS = [
    ("Go",     "goroutine 泄漏（没有退出机制的 goroutine）", "goroutine泄漏", "捏造了不存在的数据竞争"),
    ("Java",   "正确使用 synchronized 的线程安全代码", "线程安全正确", "错误指出存在竞态条件"),
    ("Python", "使用 asyncio 正确处理并发的代码", "异步正确", "误判为阻塞操作"),
    ("Go",     "channel 方向声明不规范但逻辑正确", "代码规范", "发明了不存在的死锁风险"),
    ("Java",   "使用 volatile 不够但逻辑上无竞争的代码", "并发优化", "夸大了并发风险"),
    ("Python", "multiprocessing 正确使用 Queue 通信", "进程通信正确", "误判为线程不安全"),
    ("Go",     "使用 sync.Mutex 正确保护共享状态", "互斥锁正确", "错误建议改用 channel"),
    ("Java",   "AtomicInteger 正确使用的计数器", "原子操作正确", "说应该加 synchronized"),
]

CONCURRENCY_REJECTED_PATTERNS = [
    "捏造了代码中不存在的数据竞争问题",
    "把正确的并发处理说成有死锁风险",
    "对没有并发问题的代码发出了误报的竞态警告",
    "把单线程代码错误地分析为多线程安全问题",
    "夸大了理论上的并发风险，实际代码逻辑是正确的",
]


# ──────────────────────────────────────────────
# 3. 生成单条 DPO 对
# ──────────────────────────────────────────────

SYSTEM_PROMPT_DPO = """你是一个专业的代码审查训练数据生成助手。
你的任务是生成用于 DPO（偏好对齐）训练的数据对。
每个数据对包含：
- instruction: 用户的代码审查指令
- input: 待审查的代码
- chosen: 高质量的代码审查回答
- rejected: 有明显缺陷的代码审查回答

只返回 JSON，不要任何额外解释。"""


def generate_security_pair(client, model: str, lang: str, scenario: str, vuln_type: str) -> Optional[dict]:
    """生成安全漏洞识别类 DPO 对"""
    rejected_pattern = random.choice(SECURITY_REJECTED_PATTERNS)
    instructions = [
        "请对这段代码进行安全审计。",
        "帮我找出这段代码中的安全漏洞。",
        "这段代码准备上线，请帮忙做 Code Review。",
        "请对这段代码进行全面的代码评审。",
        "你是一名资深工程师，请帮我审查这段代码。",
    ]
    instruction = random.choice(instructions)

    prompt = f"""生成一个 DPO 训练数据对，用于训练模型正确识别安全漏洞。

场景配置：
- 编程语言：{lang}
- 漏洞场景：{scenario}
- 漏洞类型：{vuln_type}
- 用户指令："{instruction}"

要求：
1. input（代码）：20-40行，包含一个 {vuln_type} 漏洞，代码其他部分写得正常
2. chosen（好回答）：
   - 准确识别并命名 {vuln_type} 漏洞
   - 解释漏洞危害
   - 给出修复后的代码
   - 长度 200-500 字
3. rejected（差回答）：{rejected_pattern}
   - 完全没提到 {vuln_type}
   - 长度 80-200 字，看起来是一个正常的（但遗漏关键点的）代码审查

返回格式（严格 JSON）：
{{"instruction": "...", "input": "...", "chosen": "...", "rejected": "..."}}"""

    return _call_api(client, model, prompt)


def generate_correct_code_pair(client, model: str, lang: str, scenario: str, quality_aspect: str) -> Optional[dict]:
    """生成代码正确类 DPO 对"""
    rejected_pattern = random.choice(CORRECT_CODE_REJECTED_PATTERNS)
    instructions = [
        "帮我 review 一下，谢谢。",
        "这是我重构后的代码，质量有提升吗？",
        "这段代码可以直接用吗？",
        "请审查并评估这段代码的质量。",
    ]
    instruction = random.choice(instructions)

    prompt = f"""生成一个 DPO 训练数据对，用于训练模型正确处理"代码写得好"的情况（给出正面评价，而不是无中生有找问题）。

场景配置：
- 编程语言：{lang}
- 代码场景：{scenario}（代码正确实现了 {quality_aspect} 的最佳实践）
- 用户指令："{instruction}"

要求：
1. input（代码）：25-45行，代码写得很好，正确处理了 {quality_aspect} 相关的关键点
2. chosen（好回答）：
   - 肯定代码的优点，具体说明哪里做得好
   - 最多提1个细微的可选改进（如"可以加一行注释"）
   - 语气积极，不挑剔
   - 长度 150-300 字
3. rejected（差回答）：{rejected_pattern}
   - 无中生有地批评代码，或误解了代码的设计意图
   - 长度 150-300 字

返回格式（严格 JSON）：
{{"instruction": "...", "input": "...", "chosen": "...", "rejected": "..."}}"""

    return _call_api(client, model, prompt)


def generate_complex_pair(client, model: str, lang: str, scenario: str, issues: list[str]) -> Optional[dict]:
    """生成复杂多问题类 DPO 对"""
    rejected_pattern = random.choice(COMPLEX_REJECTED_PATTERNS)
    instructions = [
        "请对这段代码进行全面的代码评审。",
        "帮我找出这段代码中所有可能的 bug。",
        "这段代码有哪些潜在的生产环境风险？",
        "你是一名资深工程师，请帮我审查这段代码。",
    ]
    instruction = random.choice(instructions)
    issues_str = "、".join(issues)

    prompt = f"""生成一个 DPO 训练数据对，用于训练模型全面识别复杂代码中的多个问题。

场景配置：
- 编程语言：{lang}
- 代码场景：{scenario}
- 包含的问题：{issues_str}
- 用户指令："{instruction}"

要求：
1. input（代码）：30-50行，包含全部 {len(issues)} 个问题，问题有显式的也有隐式的
2. chosen（好回答）：
   - 识别出全部 {len(issues)} 个问题：{issues_str}
   - 每个问题说明危害和修复方向
   - 给出优先级排序（哪个最重要先修）
   - 长度 300-600 字
3. rejected（差回答）：{rejected_pattern}
   - 只识别出 {issues[0]}，完全遗漏 {", ".join(issues[1:])}
   - 长度 150-250 字

返回格式（严格 JSON）：
{{"instruction": "...", "input": "...", "chosen": "...", "rejected": "..."}}"""

    return _call_api(client, model, prompt)


def generate_sql_pair(client, model: str, scenario: str, real_issue: str, false_issue: str) -> Optional[dict]:
    """生成 SQL 误判类 DPO 对"""
    rejected_pattern = random.choice(SQL_REJECTED_PATTERNS)
    instructions = [
        "请对这段 SQL 进行代码评审。",
        "帮我检查这段查询有没有问题。",
        "这段 SQL 准备上线，请帮忙审查。",
        "请对这段代码进行全面的代码评审。",
    ]
    instruction = random.choice(instructions)

    prompt = f"""生成一个 DPO 训练数据对，用于训练模型正确分析 SQL 代码问题（避免误判）。

场景配置：
- 代码类型：SQL
- 真实问题：{real_issue}（{scenario}）
- 错误答案会误判为：{false_issue}
- 用户指令："{instruction}"

要求：
1. input（SQL代码）：15-30行，包含 {real_issue} 问题，其他部分写得正常
2. chosen（好回答）：
   - 准确识别 {real_issue} 问题
   - 解释该问题在生产环境中的危害
   - 给出修复后的 SQL
   - 长度 150-350 字
3. rejected（差回答）：{rejected_pattern}
   - 把问题误诊为"{false_issue}"
   - 没有提到真正的 {real_issue}
   - 长度 100-200 字

返回格式（严格 JSON）：
{{"instruction": "...", "input": "...", "chosen": "...", "rejected": "..."}}"""

    return _call_api(client, model, prompt)


def generate_concurrency_pair(client, model: str, lang: str, scenario: str, real_issue: str, false_alarm: str) -> Optional[dict]:
    """生成并发误诊类 DPO 对"""
    rejected_pattern = random.choice(CONCURRENCY_REJECTED_PATTERNS)
    instructions = [
        "请对这段代码进行全面的代码评审。",
        "这段代码在高并发下有问题吗？",
        "帮我 review 一下并发相关的代码。",
        "你是一名资深工程师，请帮我审查这段代码。",
    ]
    instruction = random.choice(instructions)

    prompt = f"""生成一个 DPO 训练数据对，用于训练模型避免在并发代码分析中产生误报。

场景配置：
- 编程语言：{lang}
- 代码场景：{scenario}
- 真实问题（如有）：{real_issue}
- 错误答案会误报：{false_alarm}
- 用户指令："{instruction}"

要求：
1. input（代码）：20-40行，代码的并发处理是正确的（或问题很轻微），不存在 {false_alarm}
2. chosen（好回答）：
   - 准确说明并发处理是正确的，解释为什么
   - 如有真实问题（{real_issue}），正确指出
   - 不要捏造不存在的竞态条件或死锁
   - 长度 150-300 字
3. rejected（差回答）：{rejected_pattern}
   - 错误声称代码存在 {false_alarm}
   - 提供了错误的"修复"建议
   - 长度 150-250 字

返回格式（严格 JSON）：
{{"instruction": "...", "input": "...", "chosen": "...", "rejected": "..."}}"""

    return _call_api(client, model, prompt)


def _call_api(client, model: str, prompt: str) -> Optional[dict]:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system",  "content": SYSTEM_PROMPT_DPO},
                {"role": "user",    "content": prompt},
            ],
            temperature=0.8,
            max_tokens=2000,
            response_format={"type": "json_object"},
        )
        text = response.choices[0].message.content.strip()
        sample = json.loads(text)
        required = {"instruction", "input", "chosen", "rejected"}
        if required.issubset(sample.keys()):
            # 基本质量检查
            if (len(sample["input"]) > 100 and
                len(sample["chosen"]) > len(sample["rejected"]) * 1.1 and
                len(sample["chosen"]) > 150):
                return sample
    except Exception as e:
        print(f"    [错误] {e}")
    return None


# ──────────────────────────────────────────────
# 4. 主流程
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-json",     default=None,  help="SFT 评测结果 JSON 路径")
    parser.add_argument("--exclude-txt",   default=None,  help="人工精读集 txt 路径（排除用，保证三模型对比公平）")
    parser.add_argument("--api-key",       default=None)
    parser.add_argument("--base-url",      default="https://api.deepseek.com/v1")
    parser.add_argument("--model",         default="deepseek-chat")
    parser.add_argument("--synth-target",  type=int, default=360, help="合成数据目标条数")
    parser.add_argument("--out",           required=True, help="输出 JSON 路径")
    parser.add_argument("--seed",          type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    all_pairs = []

    # ── 来源1：从评测结果提取 ──
    if args.eval_json:
        print("\n[1/2] 从评测结果提取 DPO 对")
        pairs = extract_from_eval(args.eval_json, args.exclude_txt)
        all_pairs.extend(pairs)
    else:
        print("\n[1/2] 跳过（未提供 --eval-json）")

    # ── 来源2：合成生成 ──
    if args.api_key and args.synth_target > 0:
        from openai import OpenAI
        client = OpenAI(api_key=args.api_key, base_url=args.base_url)

        # 五类按 3:2:2:1.5:1.5 分配
        n_sec         = int(args.synth_target * 0.30)
        n_correct     = int(args.synth_target * 0.20)
        n_complex     = int(args.synth_target * 0.20)
        n_sql         = int(args.synth_target * 0.15)
        n_concurrency = args.synth_target - n_sec - n_correct - n_complex - n_sql

        print(f"\n[2/2] 合成生成 DPO 数据（目标 {args.synth_target} 条）")
        print(f"  安全漏洞类:   {n_sec} 条")
        print(f"  代码正确类:   {n_correct} 条")
        print(f"  复杂多问题类: {n_complex} 条")
        print(f"  SQL误判类:    {n_sql} 条")
        print(f"  并发误诊类:   {n_concurrency} 条")

        # 安全漏洞类
        print(f"\n  生成安全漏洞类...")
        scenarios = SECURITY_SCENARIOS * (n_sec // len(SECURITY_SCENARIOS) + 2)
        random.shuffle(scenarios)
        added = 0
        for i in range(n_sec * 3):
            if added >= n_sec:
                break
            lang, scenario, vuln = scenarios[i % len(scenarios)]
            print(f"    [{added+1}/{n_sec}] {lang} | {vuln}...", end=" ", flush=True)
            pair = generate_security_pair(client, args.model, lang, scenario, vuln)
            if pair:
                pair["source"] = "synth_security"
                all_pairs.append(pair)
                added += 1
                print("✓")
            else:
                print("✗")
            time.sleep(0.5)

        # 代码正确类
        print(f"\n  生成代码正确类...")
        scenarios = CORRECT_CODE_SCENARIOS * (n_correct // len(CORRECT_CODE_SCENARIOS) + 2)
        random.shuffle(scenarios)
        added = 0
        for i in range(n_correct * 3):
            if added >= n_correct:
                break
            lang, scenario, aspect = scenarios[i % len(scenarios)]
            print(f"    [{added+1}/{n_correct}] {lang} | {aspect}...", end=" ", flush=True)
            pair = generate_correct_code_pair(client, args.model, lang, scenario, aspect)
            if pair:
                pair["source"] = "synth_correct"
                all_pairs.append(pair)
                added += 1
                print("✓")
            else:
                print("✗")
            time.sleep(0.5)

        # 复杂多问题类
        print(f"\n  生成复杂多问题类...")
        scenarios = COMPLEX_SCENARIOS * (n_complex // len(COMPLEX_SCENARIOS) + 2)
        random.shuffle(scenarios)
        added = 0
        for i in range(n_complex * 3):
            if added >= n_complex:
                break
            lang, scenario, issues = scenarios[i % len(scenarios)]
            print(f"    [{added+1}/{n_complex}] {lang} | {issues[0]}+...", end=" ", flush=True)
            pair = generate_complex_pair(client, args.model, lang, scenario, issues)
            if pair:
                pair["source"] = "synth_complex"
                all_pairs.append(pair)
                added += 1
                print("✓")
            else:
                print("✗")
            time.sleep(0.5)

        # SQL 误判类
        print(f"\n  生成 SQL 误判类...")
        scenarios = SQL_SCENARIOS * (n_sql // len(SQL_SCENARIOS) + 2)
        random.shuffle(scenarios)
        added = 0
        for i in range(n_sql * 3):
            if added >= n_sql:
                break
            _, scenario, real_issue, false_issue = scenarios[i % len(scenarios)]
            print(f"    [{added+1}/{n_sql}] SQL | {real_issue}...", end=" ", flush=True)
            pair = generate_sql_pair(client, args.model, scenario, real_issue, false_issue)
            if pair:
                pair["source"] = "synth_sql"
                all_pairs.append(pair)
                added += 1
                print("✓")
            else:
                print("✗")
            time.sleep(0.5)

        # 并发误诊类
        print(f"\n  生成并发误诊类...")
        scenarios = CONCURRENCY_SCENARIOS * (n_concurrency // len(CONCURRENCY_SCENARIOS) + 2)
        random.shuffle(scenarios)
        added = 0
        for i in range(n_concurrency * 3):
            if added >= n_concurrency:
                break
            lang, scenario, real_issue, false_alarm = scenarios[i % len(scenarios)]
            print(f"    [{added+1}/{n_concurrency}] {lang} | {false_alarm[:20]}...", end=" ", flush=True)
            pair = generate_concurrency_pair(client, args.model, lang, scenario, real_issue, false_alarm)
            if pair:
                pair["source"] = "synth_concurrency"
                all_pairs.append(pair)
                added += 1
                print("✓")
            else:
                print("✗")
            time.sleep(0.5)
    else:
        print("\n[2/2] 跳过合成生成（未提供 --api-key）")

    # ── 打乱并保存 ──
    random.shuffle(all_pairs)

    # 统计来源分布
    from collections import Counter
    source_dist = Counter(p["source"] for p in all_pairs)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_pairs, f, ensure_ascii=False, indent=2)

    print(f"""
╔══════════════════════════════════════╗
  DPO 数据构建完成
  总计: {len(all_pairs)} 条
  来源分布:
    评测提取:     {source_dist.get('eval_extract', 0):>4} 条
    合成-安全:    {source_dist.get('synth_security', 0):>4} 条
    合成-正确:    {source_dist.get('synth_correct', 0):>4} 条
    合成-复杂:    {source_dist.get('synth_complex', 0):>4} 条
    合成-SQL:     {source_dist.get('synth_sql', 0):>4} 条
    合成-并发:    {source_dist.get('synth_concurrency', 0):>4} 条
  输出: {out_path}
╚══════════════════════════════════════╝

下一步：
  1. 注册数据集到 data/dataset_info.json
  2. 配置 DPO 训练 yaml
  3. 运行训练
""")


if __name__ == "__main__":
    main()
