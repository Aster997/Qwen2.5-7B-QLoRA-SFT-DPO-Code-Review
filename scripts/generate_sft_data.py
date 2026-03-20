"""
SFT 数据生成脚本（来源 B：合成数据）

设计原则：
  - 按配比矩阵（语言 × 场景 × 难度）逐格生成，保证分布均匀
  - 每次生成时注入唯一的场景约束，从根源避免代码重复
  - 支持断点续跑：中断后重新运行会跳过已完成的格子
  - 每生成10条自动保存，防止意外丢失

用法：
  python scripts/generate_sft_data.py \\
      --api-key YOUR_KEY \\
      --base-url https://api.openai.com/v1 \\
      --model gpt-4o-mini \\
      --target 750 \\
      --out data/sft_generated.json

支持任何 OpenAI 兼容接口（DeepSeek / Qwen / ZhipuAI / 本地 vLLM 等）
"""

import json
import random
import time
import argparse
import hashlib
import os
from pathlib import Path
from typing import Optional

try:
    from openai import OpenAI
except ImportError:
    raise ImportError("请先安装 openai 库：pip install openai")


# ══════════════════════════════════════════════
# 1. 配比矩阵配置
# ══════════════════════════════════════════════

# 编程语言及目标占比（加起来 = 1.0）
LANGUAGE_WEIGHTS = {
    "Python":     0.35,
    "JavaScript": 0.20,
    "Go":         0.15,
    "Java":       0.10,
    "TypeScript": 0.08,
    "SQL":        0.07,
    "Rust":       0.05,
}

# 场景类别及目标占比
SCENE_WEIGHTS = {
    "安全漏洞":  0.20,   # SQL注入/XSS/密钥硬编码/路径遍历
    "错误处理":  0.18,   # 异常忽略/panic/裸except
    "性能问题":  0.15,   # N+1查询/阻塞IO/内存泄漏
    "资源管理":  0.12,   # 连接未关闭/文件句柄/上下文管理
    "并发问题":  0.10,   # 竞态条件/死锁/协程误用
    "设计问题":  0.10,   # 职责过重/魔法数字/过度嵌套
    "代码正确":  0.10,   # 整体写得好，输出正面评价
    "可读性":    0.05,   # 命名/注释/复杂度
}

# 难度及目标占比
DIFFICULTY_WEIGHTS = {
    "简单": 0.30,   # 1个显而易见的问题，代码10-20行
    "中等": 0.50,   # 2-3个混合问题，有显式和隐式
    "困难": 0.20,   # 细微问题，或代码整体正确但有改进空间
}

# ══════════════════════════════════════════════
# 2. 场景约束池（保证代码唯一性的关键）
#    每个格子生成时从对应池中随机抽取 context，
#    让模型无法生成同一段代码
# ══════════════════════════════════════════════

SCENARIO_CONTEXTS = {
    "安全漏洞": [
        "用户登录认证模块", "商品搜索功能", "文件上传接口", "用户注册流程",
        "密码重置邮件发送", "管理员权限校验", "支付金额计算接口", "报表数据导出",
        "OAuth 第三方登录回调", "API Token 校验中间件", "用户头像上传",
        "评论内容渲染", "站内信发送功能", "日志写入模块", "数据备份脚本",
        "批量导入 CSV 数据", "用户个人信息修改", "短信验证码发送", "二维码生成接口",
    ],
    "错误处理": [
        "读取配置文件", "调用第三方 HTTP API", "解析 JSON 响应", "数据库批量插入",
        "定时任务执行器", "消息队列消费者", "文件批量处理脚本", "PDF 生成模块",
        "图片压缩处理", "发送邮件通知", "WebSocket 连接管理", "缓存读取模块",
        "数据库迁移脚本", "日志轮转处理", "远程文件下载", "ZIP 压缩解压",
        "Excel 数据读取", "音视频转码任务", "服务健康检查", "依赖服务探活",
    ],
    "性能问题": [
        "用户列表分页查询", "商品推荐接口", "统计报表生成", "批量发送通知",
        "搜索结果排序", "数据聚合计算", "全量数据同步", "树形结构遍历",
        "图片批量处理", "日志分析统计", "排行榜计算", "好友关系查询",
        "订单状态更新", "库存扣减逻辑", "评分计算接口", "标签匹配算法",
        "数据去重处理", "时序数据写入", "事件流处理", "定时聚合任务",
    ],
    "资源管理": [
        "数据库连接池使用", "文件读写操作", "网络 Socket 通信", "临时文件创建清理",
        "线程池任务提交", "HTTP 客户端会话", "内存映射文件处理", "数据库事务管理",
        "Redis 连接使用", "消息队列生产者", "加密流操作", "压缩流处理",
        "音频流读取", "数据库游标使用", "对象池管理", "锁资源获取释放",
    ],
    "并发问题": [
        "计数器并发更新", "单例模式初始化", "缓存双检锁", "任务队列并发消费",
        "多协程共享状态", "并发写入日志", "定时任务并发控制", "分布式锁实现",
        "并发请求限流", "异步任务编排", "多线程文件写入", "协程池管理",
        "发布订阅模式实现", "事件总线并发", "连接池并发借还",
    ],
    "设计问题": [
        "用户信息处理类", "订单状态机", "通知发送模块", "数据格式转换工具",
        "配置管理器", "插件加载系统", "命令行参数解析", "缓存策略实现",
        "重试装饰器", "序列化工具类", "路由分发器", "中间件链",
        "观察者模式实现", "工厂方法模式", "策略模式实现",
    ],
    "代码正确": [
        "用户密码哈希存储", "HTTP 客户端封装", "配置文件读取", "日志模块初始化",
        "数据库连接管理", "JWT token 生成验证", "参数化 SQL 查询", "文件安全写入",
        "并发安全的计数器", "带重试的 API 调用", "资源上下文管理", "输入参数校验",
        "错误信息包装", "优雅关闭服务", "健康检查端点",
    ],
    "可读性": [
        "数学工具函数集合", "字符串处理函数", "日期时间格式化", "数组/列表操作",
        "数值类型转换", "正则表达式封装", "颜色格式转换", "单位换算工具",
        "文本分词处理", "数据校验函数", "树形数据处理", "图形算法实现",
    ],
}

# ══════════════════════════════════════════════
# 3. 指令池（60+ 种，覆盖不同语气和角色）
# ══════════════════════════════════════════════

INSTRUCTIONS = {
    # 风格1：直接审查请求（对应完整报告输出）
    "full_report": [
        "请对这段代码进行全面的代码评审。",
        "请审查并评估这段代码的质量。",
        "请像 Tech Lead 一样审查这段代码。",
        "你是一名资深工程师，请帮我审查这段代码。",
        "请对这段代码进行安全审计。",
        "这段代码准备上线，请帮忙做 Code Review。",
        "Please review the following code and identify potential issues.",
        "请审查以下代码，找出潜在的问题并给出改进建议。",
        "Perform a thorough code review on this snippet.",
        "请从工程质量角度对这段代码进行评审。",
    ],
    # 风格2：聚焦特定维度（对应聚焦型输出）
    "focused": [
        "请指出这段代码中的反模式（anti-pattern）。",
        "帮我找出这段代码中所有可能的 bug。",
        "这段代码有什么不符合最佳实践的地方？",
        "这段代码有哪些潜在的生产环境风险？",
        "从性能角度看这段代码有什么问题？",
        "请从代码可维护性角度审查以下代码。",
        "这段代码的安全性如何？有没有漏洞？",
        "请检查这段代码的错误处理是否完善。",
        "从代码可读性角度，这段代码有哪些需要改进的地方？",
        "请分析这段代码的时间复杂度和空间复杂度是否合理。",
    ],
    # 风格3：简短请求（对应简短输出）
    "brief": [
        "这段代码有问题吗？",
        "帮我看看这段代码。",
        "我是新手，这段代码有什么需要改进的吗？",
        "这段代码写得怎么样？",
        "审查这段代码，给出具体的改进方案。",
        "Quick code review?",
        "这段代码可以直接用吗？",
        "帮我 review 一下，谢谢。",
    ],
    # 风格4：上下文型（模拟真实场景）
    "contextual": [
        "我写了一个接口准备上线，帮忙看看有没有问题。",
        "这是我重构后的代码，和之前相比有改进吗？",
        "同事写的代码，我不确定有没有问题，帮忙看看。",
        "这段代码在生产跑了出了 bug，帮我找找原因。",
        "这是我参考 StackOverflow 写的，放心用吗？",
        "面试写的代码，请帮我评分并指出不足。",
        "PR 要合并了，最后帮我检查一遍。",
        "这是新来的实习生写的代码，帮忙做下 review。",
    ],
}

# 各指令风格对应的 output 格式说明（用于 prompt）
OUTPUT_FORMAT_SPECS = {
    "full_report": """用以下 Markdown 格式输出：
## 代码审查报告

### 问题分析
**[严重程度] 问题标题**
具体描述...

### 改进代码
```语言
改进后的代码
```

### 最佳实践
1. 要点1
2. 要点2""",

    "focused": """直接列出发现的问题，每个问题包含：
- 问题描述（1-2句话）
- 具体影响
- 改进方向
不需要完整的代码示例，保持简洁。""",

    "brief": """用2-4句话回答，语气要自然，像在和同事交流。
如果是新手请求，语气要友好鼓励。
不需要 Markdown 格式，直接说结论和最重要的改进点。""",

    "contextual": """根据上下文语气调整回复风格。
指出最重要的1-3个问题（不必穷举），给出明确的行动建议。
语气要像在做 code review，而不是写技术文档。""",
}

# ══════════════════════════════════════════════
# 4. Prompt 构建
# ══════════════════════════════════════════════

SYSTEM_PROMPT = """你是一个专业的代码生成助手，负责生成用于训练代码审查 AI 的数据。
每次生成一个完整的训练样本，严格以 JSON 格式返回，包含三个字段：
- instruction: 用户的提问
- input: 待审查的代码片段
- output: 代码审查的回答

要求：
1. input 中的代码必须是真实、可运行的代码片段（不要添加占位符注释）
2. 代码中的问题要自然融入，不能刻意标注（如"# bug here"）
3. output 必须专业准确，指出的问题要真实存在于代码中
4. 代码长度：简单10-20行，中等20-40行，困难30-60行
5. 只返回 JSON，不要任何额外解释"""


def build_prompt(
    language: str,
    scene: str,
    difficulty: str,
    context: str,
    instruction: str,
    output_format_spec: str,
) -> str:
    """构建生成 prompt，通过具体约束保证每次生成的代码唯一"""

    # 代码正确场景：单独处理
    if scene == "代码正确":
        return f"""生成一个代码审查训练样本：

约束条件：
- 编程语言：{language}
- 场景：{context}
- 代码质量：整体写得好，没有严重问题（可以有1-2个非常细微的改进建议，但不是必须）
- 难度：{difficulty}

用户指令（instruction 字段原文）：
"{instruction}"

output 格式要求：
{output_format_spec}

注意：output 要肯定代码的优点，如果有建议也要是锦上添花而非必要修改。
避免说"没有问题"这种空洞结论，而是说"整体设计合理，XXX处理得当..."

返回格式（严格 JSON）：
{{"instruction": "...", "input": "...", "output": "..."}}"""

    # 有问题的代码场景
    difficulty_desc = {
        "简单": "只包含1个显而易见的典型问题，代码10-20行",
        "中等": "包含2-3个问题（至少1个显式+1个隐式），代码20-40行",
        "困难": "问题比较细微（如边界条件、并发时序、性能陷阱），代码30-60行，表面上看不出大问题",
    }[difficulty]

    return f"""生成一个代码审查训练样本：

约束条件：
- 编程语言：{language}
- 问题类型：{scene}
- 具体场景：{context}（代码必须体现这个具体场景，不能写成通用代码）
- 难度：{difficulty}（{difficulty_desc}）

用户指令（instruction 字段原文）：
"{instruction}"

output 格式要求：
{output_format_spec}

重要：代码要真实体现"{context}"这个具体场景，有具体的变量名、函数名、业务逻辑，
不能是抽象的 foo/bar 代码。

返回格式（严格 JSON）：
{{"instruction": "...", "input": "...", "output": "..."}}"""


# ══════════════════════════════════════════════
# 5. API 调用
# ══════════════════════════════════════════════

def extract_json(text: str) -> Optional[dict]:
    """从模型输出中提取 JSON，兼容模型在 JSON 前后加文字的情况"""
    # 先尝试直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # 再尝试提取 ```json ... ``` 代码块
    match = __import__("re").search(r"```(?:json)?\s*(\{.*?\})\s*```", text, __import__("re").DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    # 最后尝试找第一个 { ... } 块
    match = __import__("re").search(r"(\{.*\})", text, __import__("re").DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    return None


def call_api(
    client: "OpenAI",
    model: str,
    prompt: str,
    use_json_mode: bool = True,
    max_retries: int = 3,
) -> Optional[dict]:
    """
    调用 API 生成一条样本，失败自动重试。

    use_json_mode=True : 使用 response_format json_object（OpenAI/DeepSeek/Qwen 支持）
    use_json_mode=False: 不使用，靠 prompt 约束输出格式（Claude API 等不支持该参数时用）
    """
    for attempt in range(1, max_retries + 1):
        try:
            kwargs = dict(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.9,
                max_tokens=2048,
            )
            if use_json_mode:
                kwargs["response_format"] = {"type": "json_object"}

            response = client.chat.completions.create(**kwargs)
            text = response.choices[0].message.content.strip()
            sample = extract_json(text)

            if sample is None:
                print(f"    [警告] 第{attempt}次：无法提取 JSON，重试...")
                continue

            # 基本结构校验
            if all(k in sample for k in ("instruction", "input", "output")):
                if len(sample["input"]) > 50 and len(sample["output"]) > 50:
                    return sample

            print(f"    [警告] 第{attempt}次：返回结构不完整，重试...")

        except Exception as e:
            err = str(e)
            # 如果是因为不支持 json_object 参数，自动降级
            if use_json_mode and ("response_format" in err or "json_object" in err):
                print(f"    [提示] 该模型不支持 json_object 模式，自动切换为 prompt 约束模式")
                use_json_mode = False
                continue
            print(f"    [警告] 第{attempt}次：API 错误 {e}，等待后重试...")
            time.sleep(2 ** attempt)

    return None


# ══════════════════════════════════════════════
# 6. 去重检测
# ══════════════════════════════════════════════

def input_fingerprint(code: str) -> str:
    """生成代码的去重指纹（去除空白后取哈希）"""
    normalized = " ".join(code.split())
    return hashlib.md5(normalized.encode()).hexdigest()


def is_duplicate(new_input: str, seen_fingerprints: set) -> bool:
    fp = input_fingerprint(new_input)
    return fp in seen_fingerprints


# ══════════════════════════════════════════════
# 7. 加权随机采样（按配比选取下一个格子）
# ══════════════════════════════════════════════

def weighted_choice(weights: dict) -> str:
    keys = list(weights.keys())
    probs = list(weights.values())
    return random.choices(keys, weights=probs, k=1)[0]


def pick_output_format(instruction: str) -> tuple[str, str]:
    """根据 instruction 文本选择合适的输出格式"""
    if any(kw in instruction for kw in ["全面", "评审", "review", "Review", "审查", "审计", "质量", "Tech Lead", "资深"]):
        fmt = "full_report"
    elif any(kw in instruction for kw in ["反模式", "bug", "Bug", "风险", "最佳实践", "性能", "安全", "维护", "可读"]):
        fmt = "focused"
    elif any(kw in instruction for kw in ["新手", "写得", "怎么样", "看看", "Quick", "谢谢", "可以用"]):
        fmt = "brief"
    else:
        fmt = "contextual"
    return fmt, OUTPUT_FORMAT_SPECS[fmt]


# ══════════════════════════════════════════════
# 8. 断点续跑支持
# ══════════════════════════════════════════════

def load_checkpoint(path: str) -> list[dict]:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"[恢复] 加载已有数据 {len(data)} 条，继续生成...")
        return data
    return []


def save_checkpoint(data: list[dict], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ══════════════════════════════════════════════
# 9. 主生成循环
# ══════════════════════════════════════════════

def generate(
    client: "OpenAI",
    model: str,
    target: int,
    out_path: str,
    use_json_mode: bool = True,
    scene_filter: Optional[str] = None,
    max_retry_per_slot: int = 5,
):
    # 加载已有数据（断点续跑）
    data = load_checkpoint(out_path)
    seen_fps = {input_fingerprint(item["input"]) for item in data}
    generated = len(data)

    print(f"\n目标：{target} 条 | 已有：{generated} 条 | 还需：{target - generated} 条\n")

    # 展开所有指令（打平成列表方便随机采样）
    all_instructions = [
        (style, inst)
        for style, insts in INSTRUCTIONS.items()
        for inst in insts
    ]

    consecutive_failures = 0  # 连续失败计数，防止死循环

    while generated < target:
        # 按权重随机选取一个格子（scene_filter 时锁定场景）
        language   = weighted_choice(LANGUAGE_WEIGHTS)
        scene      = scene_filter if scene_filter else weighted_choice(SCENE_WEIGHTS)
        difficulty = weighted_choice(DIFFICULTY_WEIGHTS)

        # 从场景约束池随机选 context
        contexts = SCENARIO_CONTEXTS.get(scene, ["通用场景"])
        context  = random.choice(contexts)

        # 随机选 instruction
        inst_style, instruction = random.choice(all_instructions)
        output_fmt, output_fmt_spec = pick_output_format(instruction)

        # 构建 prompt
        prompt = build_prompt(
            language, scene, difficulty,
            context, instruction, output_fmt_spec,
        )

        print(f"[{generated+1}/{target}] {language} | {scene} | {difficulty} | {context[:15]}...")

        # 生成（带重试去重）
        success = False
        for _ in range(max_retry_per_slot):
            sample = call_api(client, model, prompt, use_json_mode=use_json_mode)
            if sample is None:
                continue
            if is_duplicate(sample["input"], seen_fps):
                print(f"    [去重] 检测到重复 input，重新生成...")
                continue
            # 成功：记录并保存
            seen_fps.add(input_fingerprint(sample["input"]))
            data.append(sample)
            generated += 1
            consecutive_failures = 0
            success = True
            print(f"    ✓ 生成成功（input {len(sample['input'])}字符，output {len(sample['output'])}字符）")
            break

        if not success:
            consecutive_failures += 1
            print(f"    ✗ 该格子生成失败，跳过（连续失败 {consecutive_failures} 次）")
            if consecutive_failures >= 10:
                print("\n[错误] 连续失败 10 次，请检查 API 配置后重试")
                break

        # 每10条自动保存
        if generated % 10 == 0 and generated > 0:
            save_checkpoint(data, out_path)
            print(f"  → 已保存 {generated} 条到 {out_path}")

    # 最终保存
    save_checkpoint(data, out_path)

    # 打印最终统计
    print(f"\n{'='*50}")
    print(f"生成完成：{len(data)} 条")
    print(f"保存路径：{out_path}")
    lang_count = {}
    scene_count = {}
    for item in data:
        # 注：instruction 里不含语言/场景信息，这里只是简单统计
        pass
    print(f"{'='*50}\n")


# ══════════════════════════════════════════════
# 10. 入口
# ══════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="SFT 合成数据生成脚本")
    parser.add_argument("--api-key",       required=True,  help="API Key")
    parser.add_argument("--base-url",      default="https://api.openai.com/v1", help="API Base URL")
    parser.add_argument("--model",         default="gpt-4o-mini", help="模型名称")
    parser.add_argument("--target",        type=int, default=750,  help="目标生成条数")
    parser.add_argument("--out",           default="data/sft_generated.json", help="输出文件路径")
    parser.add_argument("--seed",          type=int, default=42,   help="随机种子（保证可复现）")
    parser.add_argument("--no-json-mode",  action="store_true",
                        help="禁用 response_format json_object（Claude API 等不支持时使用）")
    parser.add_argument("--scene",         default=None,
                        help="只生成指定场景，如 '代码正确'（不指定则按配比随机）")
    args = parser.parse_args()

    random.seed(args.seed)
    use_json_mode = not args.no_json_mode

    print(f"模型  : {args.model}")
    print(f"接口  : {args.base_url}")
    print(f"目标  : {args.target} 条")
    print(f"输出  : {args.out}")
    print(f"场景  : {args.scene if args.scene else '按配比随机'}")
    print(f"JSON模式: {'关闭（prompt约束）' if not use_json_mode else '开启'}")

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)

    generate(
        client=client,
        model=args.model,
        target=args.target,
        out_path=args.out,
        use_json_mode=use_json_mode,
        scene_filter=args.scene,
    )


if __name__ == "__main__":
    main()
