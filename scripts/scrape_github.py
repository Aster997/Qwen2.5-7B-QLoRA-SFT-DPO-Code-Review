"""
来源A：GitHub 真实代码爬虫

策略：
  1. 用 GitHub Search API 搜索 bug-fix 类 PR（安全修复、空指针、资源泄漏等）
  2. 获取 PR 的 diff，从中提取修复前的「有问题的代码」片段
  3. 用 LLM 对该片段生成代码审查 output（PR 原始评论质量不稳定，不直接用）
  4. 保存为 Alpaca 格式

用法：
  python scripts/scrape_github.py \\
      --github-token <YOUR_GITHUB_TOKEN> \\
      --api-key <YOUR_DEEPSEEK_API_KEY> \\
      --base-url https://api.deepseek.com/v1 \\
      --target 150 \\
      --out data/sft_source_a.json
"""

import json
import time
import re
import random
import argparse
import hashlib
from typing import Optional
from pathlib import Path

try:
    import requests
except ImportError:
    raise ImportError("请先安装 requests：pip install requests")

try:
    from openai import OpenAI
except ImportError:
    raise ImportError("请先安装 openai：pip install openai")


# ──────────────────────────────────────────────
# 1. 搜索查询配置
# ──────────────────────────────────────────────

# 每个 query 对应 (搜索词, 语言, 问题类型)
# 精准搜索 bug-fix PR，避免功能新增类 PR
SEARCH_QUERIES = [
    # Python 安全
    ("fix sql injection",           "python",     "安全漏洞"),
    ("fix xss vulnerability",       "javascript", "安全漏洞"),
    ("remove hardcoded password",   "python",     "安全漏洞"),
    ("fix path traversal",          "python",     "安全漏洞"),
    ("fix command injection",       "python",     "安全漏洞"),
    ("fix csrf",                    "javascript", "安全漏洞"),
    # Python 错误处理
    ("fix bare except",             "python",     "错误处理"),
    ("fix exception handling",      "python",     "错误处理"),
    ("fix silent exception",        "python",     "错误处理"),
    # Python 资源管理
    ("fix resource leak",           "python",     "资源管理"),
    ("fix file not closed",         "python",     "资源管理"),
    ("fix connection leak",         "python",     "资源管理"),
    # Go
    ("fix error ignored",           "go",         "错误处理"),
    ("fix goroutine leak",          "go",         "资源管理"),
    ("fix race condition",          "go",         "并发问题"),
    ("fix nil pointer",             "go",         "错误处理"),
    # Java
    ("fix null pointer exception",  "java",       "错误处理"),
    ("fix resource not closed",     "java",       "资源管理"),
    ("fix sql injection",           "java",       "安全漏洞"),
    # JavaScript/TypeScript
    ("fix memory leak",             "javascript", "性能问题"),
    ("fix prototype pollution",     "javascript", "安全漏洞"),
    ("fix async error handling",    "javascript", "错误处理"),
]

# 每种语言对应的文件扩展名
LANG_EXTENSIONS = {
    "python":     [".py"],
    "javascript": [".js", ".mjs"],
    "typescript": [".ts", ".tsx"],
    "go":         [".go"],
    "java":       [".java"],
    "sql":        [".sql"],
}

# 指令池（同 generate 脚本，保证多样性）
INSTRUCTIONS = [
    "请对这段代码进行全面的代码评审。",
    "请指出这段代码中的反模式（anti-pattern）。",
    "帮我找出这段代码中所有可能的 bug。",
    "请审查并评估这段代码的质量。",
    "请对这段代码进行安全审计。",
    "请像 Tech Lead 一样审查这段代码。",
    "这段代码准备上线，请帮忙做 Code Review。",
    "这段代码有哪些潜在的生产环境风险？",
    "Perform a thorough code review on this snippet.",
    "这段代码有什么不符合最佳实践的地方？",
    "请检查这段代码的错误处理是否完善。",
    "从安全角度审查这段代码，有没有漏洞？",
]


# ──────────────────────────────────────────────
# 2. GitHub API 封装
# ──────────────────────────────────────────────

class GitHubClient:
    BASE = "https://api.github.com"

    def __init__(self, token: str):
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        })
        self._last_request = 0.0

    def _get(self, url: str, params: dict = None) -> Optional[dict]:
        """带限速的 GET 请求（每秒最多1次）"""
        elapsed = time.time() - self._last_request
        if elapsed < 1.0:
            time.sleep(1.0 - elapsed)
        try:
            resp = self.session.get(url, params=params, timeout=15)
            self._last_request = time.time()
            if resp.status_code == 403:
                # Rate limit 触发，等待后重试
                reset = int(resp.headers.get("X-RateLimit-Reset", time.time() + 60))
                wait = max(reset - time.time() + 5, 10)
                print(f"    [限速] 等待 {wait:.0f} 秒...")
                time.sleep(wait)
                return self._get(url, params)
            if resp.status_code == 200:
                return resp.json()
            print(f"    [HTTP {resp.status_code}] {url}")
            return None
        except Exception as e:
            print(f"    [请求错误] {e}")
            return None

    def search_prs(self, query: str, language: str, page: int = 1) -> list[dict]:
        """搜索 merged PR"""
        q = f"{query} language:{language} type:pr is:merged is:public"
        data = self._get(f"{self.BASE}/search/issues", params={
            "q": q, "per_page": 15, "page": page, "sort": "updated",
        })
        if not data:
            return []
        return data.get("items", [])

    def get_pr_files(self, owner: str, repo: str, pr_number: int) -> list[dict]:
        """获取 PR 修改的文件列表（含 patch diff）"""
        data = self._get(f"{self.BASE}/repos/{owner}/{repo}/pulls/{pr_number}/files",
                         params={"per_page": 30})
        return data if isinstance(data, list) else []


# ──────────────────────────────────────────────
# 3. Diff 解析：提取「修复前」的代码片段
# ──────────────────────────────────────────────

def parse_pr_url(url: str) -> tuple[str, str, int]:
    """从 PR HTML URL 解析 owner/repo/number"""
    # https://github.com/owner/repo/pull/123
    m = re.match(r"https://github\.com/([^/]+)/([^/]+)/pull/(\d+)", url)
    if m:
        return m.group(1), m.group(2), int(m.group(3))
    return None, None, None


def extract_before_snippet(patch: str, max_lines: int = 60) -> Optional[str]:
    """
    从 unified diff patch 中提取「修复前」的代码。
    策略：取第一个 hunk，保留上下文行和删除行（-），跳过新增行（+）。
    """
    if not patch:
        return None

    lines = patch.split("\n")
    before_lines = []
    in_hunk = False
    hunk_count = 0

    for line in lines:
        if line.startswith("@@"):
            if hunk_count >= 1:
                break   # 只取第一个 hunk，避免代码太长
            in_hunk = True
            hunk_count += 1
            continue
        if not in_hunk:
            continue
        if line.startswith("+"):
            continue    # 跳过新增行（这是修复后的）
        if line.startswith("-"):
            before_lines.append(line[1:])   # 删除行就是原始有问题的代码
        elif line.startswith("\\"):
            continue    # 跳过 "\ No newline at end of file"
        else:
            before_lines.append(line[1:] if line.startswith(" ") else line)

    if len(before_lines) < 5:
        return None

    # 截断过长的代码
    snippet = "\n".join(before_lines[:max_lines])
    return snippet.strip()


def is_good_snippet(snippet: str, lang_exts: list[str], filename: str) -> bool:
    """判断这个代码片段是否值得用"""
    if not snippet:
        return False
    lines = snippet.split("\n")
    if len(lines) < 8 or len(lines) > 80:
        return False
    # 检查文件扩展名是否匹配
    if not any(filename.endswith(ext) for ext in lang_exts):
        return False
    # 过滤掉测试文件（测试文件的代码不适合做代码审查训练）
    low = filename.lower()
    if any(kw in low for kw in ["test", "spec", "mock", "fixture", "_test.", ".test."]):
        return False
    return True


# ──────────────────────────────────────────────
# 4. LLM 生成代码审查
# ──────────────────────────────────────────────

REVIEW_SYSTEM_PROMPT = """你是一个专业的代码审查专家。
给你一段真实的有问题的代码（来自 GitHub PR 修复前的版本），
请生成专业的代码审查意见。

严格以 JSON 格式返回：{"output": "审查内容"}
不要包含其他内容。"""


def generate_review(
    client: OpenAI,
    model: str,
    snippet: str,
    language: str,
    scene: str,
    use_json_mode: bool = True,
) -> Optional[str]:
    """用 LLM 为代码片段生成审查 output"""
    prompt = f"""以下是一段来自真实项目的 {language} 代码，存在{scene}问题，请审查：

```{language}
{snippet}
```

要求：
- 准确指出代码中实际存在的问题（不要编造不存在的问题）
- 解释问题的危害
- 给出改进建议或改进后的代码
- 格式：简洁的 Markdown，100-600字

返回 JSON：{{"output": "审查内容"}}"""

    try:
        kwargs = dict(
            model=model,
            messages=[
                {"role": "system", "content": REVIEW_SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.7,
            max_tokens=1200,
        )
        if use_json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        resp = client.chat.completions.create(**kwargs)
        text = resp.choices[0].message.content.strip()

        # 提取 JSON
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r'"output"\s*:\s*"(.*)"', text, re.DOTALL)
            if m:
                return m.group(1).replace("\\n", "\n")
            return None

        return data.get("output")
    except Exception as e:
        # 自动降级
        if use_json_mode and ("response_format" in str(e) or "json_object" in str(e)):
            return generate_review(client, model, snippet, language, scene, use_json_mode=False)
        print(f"    [LLM错误] {e}")
        return None


# ──────────────────────────────────────────────
# 5. 去重
# ──────────────────────────────────────────────

def fp(code: str) -> str:
    return hashlib.md5(" ".join(code.split()).encode()).hexdigest()


# ──────────────────────────────────────────────
# 6. 主流程
# ──────────────────────────────────────────────

def load_existing(path: str) -> tuple[list[dict], set]:
    if Path(path).exists():
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data, {fp(item["input"]) for item in data}
    return [], set()


def save(data: list[dict], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def scrape(
    gh: GitHubClient,
    llm: OpenAI,
    model: str,
    target: int,
    out_path: str,
    use_json_mode: bool = True,
):
    data, seen_fps = load_existing(out_path)
    collected = len(data)
    print(f"\n目标：{target} 条 | 已有：{collected} 条 | 还需：{target - collected} 条\n")

    queries = SEARCH_QUERIES.copy()
    random.shuffle(queries)

    for query_text, language, scene in queries:
        if collected >= target:
            break

        print(f"\n── 搜索：'{query_text}' ({language}) ──")
        lang_exts = LANG_EXTENSIONS.get(language, [f".{language}"])

        for page in range(1, 4):   # 最多搜3页
            if collected >= target:
                break

            prs = gh.search_prs(query_text, language, page=page)
            if not prs:
                break

            print(f"  第{page}页: 找到 {len(prs)} 个 PR")

            for pr in prs:
                if collected >= target:
                    break

                owner, repo, pr_num = parse_pr_url(pr.get("html_url", ""))
                if not owner:
                    continue

                print(f"  处理: {owner}/{repo}#{pr_num} - {pr.get('title','')[:50]}")

                # 获取 PR 修改的文件
                files = gh.get_pr_files(owner, repo, pr_num)
                if not files:
                    continue

                # 找到一个合适的文件
                for file_info in files:
                    filename = file_info.get("filename", "")
                    patch    = file_info.get("patch", "")
                    status   = file_info.get("status", "")

                    # 只处理修改的文件（不是新增/删除整个文件）
                    if status != "modified":
                        continue

                    if not is_good_snippet(patch, lang_exts, filename):
                        continue

                    snippet = extract_before_snippet(patch)
                    if not snippet:
                        continue

                    # 去重检查
                    snippet_fp = fp(snippet)
                    if snippet_fp in seen_fps:
                        print(f"    [跳过] 重复代码片段")
                        continue

                    # 生成审查
                    print(f"    生成审查: {filename} ({len(snippet.splitlines())}行)...")
                    review = generate_review(llm, model, snippet, language, scene, use_json_mode)
                    if not review or len(review) < 50:
                        print(f"    [跳过] 审查内容太短或生成失败")
                        continue

                    # 组装样本
                    instruction = random.choice(INSTRUCTIONS)
                    sample = {
                        "instruction": instruction,
                        "input":       snippet,
                        "output":      review,
                        "_source":     f"github:{owner}/{repo}#{pr_num}:{filename}",
                    }

                    data.append(sample)
                    seen_fps.add(snippet_fp)
                    collected += 1
                    print(f"    ✓ [{collected}/{target}] 收集成功")

                    # 每5条保存一次
                    if collected % 5 == 0:
                        save(data, out_path)
                        print(f"  → 已保存 {collected} 条")

                    break   # 每个 PR 只取一个文件片段

    # 清理 _source 字段（LLaMA-Factory 不需要）
    clean_data = [{k: v for k, v in item.items() if k != "_source"} for item in data]
    save(clean_data, out_path)

    print(f"\n{'='*50}")
    print(f"爬取完成：{len(clean_data)} 条")
    print(f"保存路径：{out_path}")
    print(f"{'='*50}\n")


# ──────────────────────────────────────────────
# 7. 入口
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GitHub 真实代码爬虫（来源A）")
    parser.add_argument("--github-token", required=True, help="GitHub Personal Access Token")
    parser.add_argument("--api-key",      required=True, help="LLM API Key（用于生成审查）")
    parser.add_argument("--base-url",     default="https://api.deepseek.com/v1")
    parser.add_argument("--model",        default="deepseek-chat")
    parser.add_argument("--target",       type=int, default=150)
    parser.add_argument("--out",          default="data/sft_source_a.json")
    parser.add_argument("--no-json-mode", action="store_true")
    parser.add_argument("--seed",         type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    print(f"目标条数 : {args.target}")
    print(f"输出路径 : {args.out}")
    print(f"LLM 模型 : {args.model}")

    gh  = GitHubClient(args.github_token)
    llm = OpenAI(api_key=args.api_key, base_url=args.base_url)

    scrape(
        gh=gh,
        llm=llm,
        model=args.model,
        target=args.target,
        out_path=args.out,
        use_json_mode=not args.no_json_mode,
    )


if __name__ == "__main__":
    main()
