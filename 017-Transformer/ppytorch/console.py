"""
console — 展示层 (不参与模型计算)。

  - 各模块 main() 返回 ModuleResult
  - python -m ppytorch.<module> → render_report
  - python -m ppytorch.verify   → render_summary
"""

from rich.console import Console
from rich.table import Table
from rich import box

console = Console()

_PASS = "[bold bright_green]✔ PASS[/]"
_FAIL = "[bold bright_red]✘ FAIL[/]"


class ModuleResult:
    """单个模块的对拍结果 (纯数据容器, 不做任何打印)。"""

    def __init__(self, title, subtitle=None, info=None, checks=None, extra=None):
        self.title = title
        self.subtitle = subtitle
        self.info = info or []
        self.checks = checks or []
        # extra: 额外 renderable 列表 (如掩码矩阵表), 仅在单模块详细视图展示
        self.extra = extra or []

    @property
    def passed(self) -> bool:
        return all(ok for _, ok in self.checks)

    def __bool__(self):
        # 让 verify.py 里 `sum(1 for r in results if r)` 能直接统计通过数
        return self.passed


def _mark(ok: bool) -> str:
    return _PASS if ok else _FAIL


def matrix_panel(mat, title, fmt="{:.0f}"):
    """把 2D 张量渲染成一张表 (用于展示掩码), -inf 显示为红色 -∞。返回 renderable。"""
    t = Table(box=box.MINIMAL, show_header=False, pad_edge=False,
              title=f"[dim]{title}[/]", title_justify="left", border_style="bright_black")
    for _ in range(mat.shape[1]):
        t.add_column(justify="right", min_width=4)
    for row in mat.tolist():
        cells = [("[bright_red]-∞[/]" if v == float("-inf")
                  else f"[bright_green]{fmt.format(v)}[/]") for v in row]
        t.add_row(*cells)
    return t


# ───── 单模块详细报告 (python3 xxx.py 时用) ─────────────────────────────────

def render_report(result: ModuleResult) -> ModuleResult:
    """单模块详细视图: 一张带边框的表 (标题+信息+校验), 无满宽横线。"""
    t = Table(box=box.ROUNDED, show_header=False, padding=(0, 1),
              title=f"[bold bright_cyan]{result.title}[/]", title_justify="left",
              border_style="bright_black")
    t.add_column(justify="left", style="bright_white", no_wrap=True)
    t.add_column(justify="left")
    if result.subtitle:
        t.add_row("[dim]说明[/]", f"[dim]{result.subtitle}[/]")
    if result.info:
        t.add_section()
        for label, val in result.info:
            t.add_row(label, f"[bright_yellow]{val}[/]")
    if result.checks:
        t.add_section()
        for desc, ok in result.checks:
            t.add_row(desc, _mark(ok))
    console.print(t)
    for extra in result.extra:
        console.print(extra)
    return result


# ───── 汇总视图 (verify.py 时用): 所有模块合并成一张统一表 ──────────────────

def render_summary(results):
    """
    把所有模块的校验合并到 **一张表**。列宽全局统一 -> 天然对齐;
    只有一条表头横线, 模块之间用轻分隔线区分, 不再满屏横线。
    """
    t = Table(box=box.SIMPLE_HEAD, pad_edge=False, border_style="bright_black",
              header_style="bold bright_cyan",
              title="[bold bright_white]各模块 vs 官方 PyTorch 对拍[/]", title_justify="left")
    t.add_column("模块", style="bright_cyan", no_wrap=True, min_width=20)
    t.add_column("校验项", style="bright_white", min_width=46, no_wrap=True)
    t.add_column("结果", justify="center", min_width=6)
    for i, r in enumerate(results):
        if i:
            t.add_section()  # 模块间轻分隔
        for j, (desc, ok) in enumerate(r.checks):
            t.add_row(r.title if j == 0 else "", desc, _mark(ok))
    console.print(t)

    passed = sum(1 for r in results if r.passed)
    total = len(results)
    ok = passed == total
    style = "bold bright_green" if ok else "bold bright_red"
    console.print(f"  合计: [{style}]{passed}/{total} 模块通过[/]\n")
    return passed, total


def section(title):
    """两大部分的标题 (verify.py)。用一条横线, 克制使用。"""
    console.print(f"\n[bold bright_magenta]▌ {title}[/]\n")


def train_table(history):
    """训练日志表: history = [(step, loss)]。"""
    t = Table(box=box.SIMPLE_HEAD, pad_edge=False, border_style="bright_black",
              header_style="bold bright_cyan",
              title="[bright_white]复制任务训练[/]", title_justify="left")
    t.add_column("step", justify="right", min_width=6)
    t.add_column("loss", justify="right", min_width=10)
    for step, loss in history:
        color = "bright_green" if loss < 0.5 else ("bright_yellow" if loss < 1.5 else "bright_white")
        t.add_row(str(step), f"[{color}]{loss:.4f}[/]")
    console.print(t)


def samples_table(rows):
    """
    生成样例表: rows = [(样本号, 源序列str, 生成结果str, 是否完全匹配)]。
    """
    t = Table(box=box.SIMPLE_HEAD, pad_edge=False, border_style="bright_black",
              header_style="bold bright_cyan",
              title="[bright_white]生成样例[/]", title_justify="left")
    t.add_column("样本", justify="right", min_width=4)
    t.add_column("源序列 (去 bos/eos)", style="bright_white", min_width=28)
    t.add_column("生成结果 (去 bos)", min_width=28)
    for idx, ref, out, match in rows:
        color = "bold bright_green" if match else "bright_yellow"
        t.add_row(str(idx), ref, f"[{color}]{out}[/]")
    console.print(t)
