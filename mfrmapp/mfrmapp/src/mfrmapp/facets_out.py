"""
MFRMSight v0.9.0 — Facets .out.txt 解析器
========================================
解析 Facets (Minifac) 输出文件，提取关键统计表格，
支持生成中文 Markdown 报告。
"""
import re
from typing import Optional
from pathlib import Path


def parse_facets_out(filepath: str) -> dict:
    """解析 Facets .out.txt 文件，返回结构化数据字典。

    提取内容:
      - header: 标题/Facets数/Model/Noncentered/Positive
      - table5: 可测量数据摘要 (ObsMean, ExpMean, VarExp, etc.)
      - students: 学生测量报告 [{label, total, count, obs_avg, fair_avg, meas, se, infit, outfit, ...}]
      - raters: 评分者测量报告
      - criteria: 标准测量报告
      - categories: 评分等级统计 [{score, count, pct, avg_meas, exp_meas, outfit, threshold, ...}]
      - unexpected: 异常反应 [{score, exp, resid, stres, student, criterion, ...}]
    """
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    result: dict = {
        "header": {},
        "table5": {},
        "students": {"rows": [], "separation": 0, "reliability": 0, "chi_sq": "", "rmse": 0},
        "raters": {"rows": [], "separation": 0, "reliability": 0, "chi_sq": "", "rmse": 0},
        "criteria": {"rows": [], "separation": 0, "reliability": 0, "chi_sq": "", "rmse": 0},
        "categories": [],
        "unexpected": [],
        "raw_lines": lines,
    }

    # ── 解析 Header ──
    for i, line in enumerate(lines[:50]):
        if line.startswith("Title ="):
            result["header"]["title"] = line.split("=", 1)[1].strip()
        elif "Facets =" in line and not line.strip().startswith(";"):
            m = re.search(r"Facets\s*=\s*(\d+)", line)
            if m:
                result["header"]["facets"] = int(m.group(1))
        elif "Non-centered" in line and "=" in line:
            m = re.search(r"Non-centered\s*=\s*(\d+)", line)
            if m:
                result["header"]["noncentered"] = int(m.group(1))
        elif "Positive =" in line:
            m = re.search(r"Positive\s*=\s*(\d+)", line)
            if m:
                result["header"]["positive"] = int(m.group(1))
        elif "Model =" in line and "?" in line:
            result["header"]["model"] = line.split("=", 1)[1].strip()

    # ── 定位各 Table ──
    table5_start = _find_section(lines, "Table 5.")
    table7_students = _find_section(lines, "Table 7.1.1")
    table7_raters = _find_section(lines, "Table 7.2.1")
    table7_criteria = _find_section(lines, "Table 7.3.1")
    table8_start = _find_section(lines, "Table 8.1  Category Statistics")
    table4_start = _find_section(lines, "Table 4.")

    # ── 解析 Table 5 (Measurable Data Summary) ──
    if table5_start >= 0:
        _parse_table5(lines, table5_start, result)

    # ── 解析 Table 7.1.1 (Students) ──
    if table7_students >= 0:
        _parse_table7(lines, table7_students, result["students"])

    # ── 解析 Table 7.2.1 (Raters) ──
    if table7_raters >= 0:
        _parse_table7(lines, table7_raters, result["raters"])

    # ── 解析 Table 7.3.1 (Criteria) ──
    if table7_criteria >= 0:
        _parse_table7(lines, table7_criteria, result["criteria"])

    # ── 解析 Table 8 (Category Statistics) ──
    if table8_start >= 0:
        _parse_table8(lines, table8_start, result)

    # ── 解析 Table 4 (Unexpected Responses) ──
    if table4_start >= 0:
        _parse_table4(lines, table4_start, result)

    return result


def _find_section(lines: list[str], marker: str) -> int:
    """定位包含 marker 的行索引"""
    for i, line in enumerate(lines):
        if marker in line:
            return i
    return -1


def _parse_table5(lines: list[str], start: int, result: dict):
    """解析 Table 5 — Measurable Data Summary"""
    t5 = result["table5"]
    for i in range(start, min(start + 50, len(lines))):
        line = lines[i]
        # 均值行: "|10.62 10.62 10.62  -.00  .00 | Mean (Count: 64)"
        if "Mean (Count:" in line:
            parts = [p for p in line.replace("|", " ").split() if p]
            if len(parts) >= 6:
                try:
                    t5["obs_mean"] = float(parts[0])
                    t5["exp_mean"] = float(parts[2])
                    t5["avg_resid"] = float(parts[3])
                    t5["stres_mean"] = float(parts[4])
                except (ValueError, IndexError):
                    pass
                try:
                    t5["n"] = int(re.search(r"Count:\s*(\d+)", line).group(1))
                except Exception:
                    t5["n"] = 0
        # S.D. 行: "| 5.26  5.26  4.98  .48  .98 | S.D. (Sample)"
        elif "S.D. (Sample)" in line or "S.D. (Population)" in line:
            parts = [p for p in line.replace("|", " ").split() if p]
            if len(parts) >= 6 and "stres_sd" not in t5:
                try:
                    t5["stres_sd"] = float(parts[4])
                except (ValueError, IndexError):
                    pass
        # 方差解释
        elif "Variance explained by Rasch measures" in line:
            m = re.search(r"=\s*([\d.]+)\s*([\d.]+)%", line)
            if m:
                t5["var_exp"] = float(m.group(2))
        elif "Raw-score variance" in line:
            m = re.search(r"=\s*([\d.]+)\s*([\d.]+)%", line)
            if m:
                t5["raw_var"] = float(m.group(2))
        elif "Variance of residuals" in line:
            m = re.search(r"=\s*([\d.]+)\s*([\d.]+)%", line)
            if m:
                t5["resid_var"] = float(m.group(2))
        elif "Global Pearson chi-squared" in line:
            m = re.search(r"=\s*([\d.]+).*prob.*=\s*([\d.]+)", line)
            if m:
                t5["chi_sq"] = m.group(1)
                t5["chi_prob"] = m.group(2)


def _parse_table7(lines: list[str], start: int, facet: dict):
    """解析 Table 7.x.x — Measurement Report"""
    data_start = -1

    for i in range(start, min(start + 15, len(lines))):
        line = lines[i]
        # 表头分隔线: "|---+---+---|" or "+---+---+"
        bare = line.strip()
        if (bare.startswith("|") and "---" in bare) or (bare.startswith("+") and "---" in bare):
            data_start = i + 1
            break

    if data_start < 0:
        return

    for i in range(data_start, min(data_start + 30, len(lines))):
        line = lines[i]
        bare = line.strip()
        # 统计行
        if bare.startswith("Model, Populn:") or bare.startswith("Model, Sample:"):
            m_sep = re.search(r'Separation\s+([\d.]+)', bare)
            m_rel = re.search(r'Reliability\s+([\d.]+)', bare)
            m_rmse = re.search(r'RMSE\s+([\d.]+)', bare)
            if m_sep and facet.get("separation", 0) == 0:
                facet["separation"] = float(m_sep.group(1))
            if m_rel:
                facet["reliability"] = float(m_rel.group(1))
            if m_rmse:
                facet["rmse"] = float(m_rmse.group(1))
            continue
        if "Fixed (all same) chi-squared" in bare:
            m = re.search(r'chi-squared:\s*([\d.]+).*d\.f\.:\s*(\d+).*significance.*:\s*([\d.]+)', bare)
            if m:
                facet["chi_sq"] = f"χ²({m.group(2)})={m.group(1)}, p={m.group(3)}"
            continue
        # 均值/SD 行 — 继续循环但不作为数据行
        if "Mean (Count:" in bare or "S.D." in bare:
            continue
        # 下一个 Table 编号出现时退出
        if bare.startswith("+"):
            continue
        if "Table 7." in bare and facet["rows"]:
            break
        if bare.startswith("|--"):
            continue
        # 数据行: "| ... |"
        if bare.startswith("|"):
            parts = [p.strip() for p in line.split("|")]
            parts = [p for p in parts if p]
            if len(parts) < 5:
                continue
            try:
                row_parts = parts[0].split()
                if len(row_parts) < 4:
                    continue
                total = int(float(row_parts[0]))
                count = int(float(row_parts[1]))
                obs_avg = float(row_parts[2])
                fair_avg = float(row_parts[3])
                meas_parts = parts[1].split()
                meas = float(meas_parts[0])
                se = float(meas_parts[1])
                fit_parts = parts[2].split()
                infit = float(fit_parts[0])
                infit_z = float(fit_parts[1]) if len(fit_parts) > 1 else 0.0
                outfit = float(fit_parts[2]) if len(fit_parts) > 2 else float(fit_parts[0])
                outfit_z = float(fit_parts[3]) if len(fit_parts) > 3 else 0.0
                label = parts[-1].strip() if len(parts) > 3 else ""
                facet["rows"].append({
                    "label": label,
                    "total": total, "count": count,
                    "obs_avg": obs_avg, "fair_avg": fair_avg,
                    "meas": meas, "se": se,
                    "infit": infit, "infit_z": infit_z,
                    "outfit": outfit, "outfit_z": outfit_z,
                })
            except (ValueError, IndexError):
                continue


def _parse_table8(lines: list[str], start: int, result: dict):
    """解析 Table 8.1 — Category Statistics"""
    in_data = False
    for i in range(start, min(start + 80, len(lines))):
        line = lines[i]
        if "Category" in line and "Counts" in line:
            in_data = True
            continue
        if not in_data:
            continue
        bare = line.strip()
        if bare.startswith("+") or bare.startswith("|-"):
            if bare.startswith("+"):
                break
            continue
        if "Scale structure" in line or "Probability Curves" in line:
            break
        parts = [p.strip() for p in line.split("|")]
        parts = [p for p in parts if p]
        if len(parts) < 5:
            continue
        try:
            # parts[0]: "2       3         3    1%   1%" → [score, total, used, pct, cum%]
            col1 = parts[0].split()
            if len(col1) < 3:
                continue
            score = int(col1[0])
            count = int(col1[1])
            pct = 0.0
            # Find the percentage column (contains %)
            for field in col1[2:]:
                if "%" in field:
                    try:
                        pct = float(field.rstrip("%"))
                    except ValueError:
                        pass
                    break
            # parts[1]: "-1.79  -2.35  1.6" → [avg_mea, exp_mea, outfit]
            col2 = parts[1].split()
            if len(col2) < 2:
                continue
            avg_str = col2[0].rstrip("*") if col2 else "0"
            avg_m = float(avg_str) if avg_str and avg_str.replace("-", "").replace(".", "").isdigit() else 0.0
            exp_m = float(col2[1]) if len(col2) > 1 and col2[1].replace("-", "").replace(".", "").isdigit() else 0.0
            outfit = float(col2[2]) if len(col2) > 2 and col2[2].replace("-", "").replace(".", "").isdigit() else 0.0
            # parts[3] or parts[9]: Threshold Measure
            threshold = -999.0
            thresh_se = 0.0
            # Threshold is in parts[9] if present (the "RASCH-ANDRICH Thresholds" column)
            for p_idx in [9, 3]:
                if len(parts) > p_idx and parts[p_idx]:
                    thr_raw = parts[p_idx].replace("(", "").replace(")", "").strip()
                    if thr_raw and thr_raw.replace("-", "").replace(".", "").isdigit():
                        threshold = float(thr_raw)
                        break
            result["categories"].append({
                "score": score, "count": count, "pct": pct,
                "avg_meas": avg_m, "exp_meas": exp_m,
                "outfit": outfit, "threshold": threshold, "thresh_se": thresh_se,
            })
        except (ValueError, IndexError):
            continue


def _parse_table4(lines: list[str], start: int, result: dict):
    """解析 Table 4 — Unexpected Responses"""
    in_data = False
    for i in range(start, min(start + 30, len(lines))):
        line = lines[i]
        if "Cat" in line and "Score" in line and "Exp." in line:
            in_data = True
            continue
        if not in_data:
            continue
        if "+" in line and "-" in line and len(line.strip()) < 5:
            continue
        parts = line.split("|")
        if len(parts) < 6:
            # Try space-separated
            parts_space = line.split()
            if len(parts_space) >= 10:
                try:
                    result["unexpected"].append({
                        "score": int(parts_space[0]),
                        "exp": float(parts_space[2]),
                        "resid": float(parts_space[3]),
                        "stres": float(parts_space[4]),
                        "student": parts_space[5],
                        "criterion": parts_space[-1],
                    })
                except (ValueError, IndexError):
                    continue
            continue
        try:
            result["unexpected"].append({
                "score": int(float(parts[1].strip())),
                "exp": float(parts[2].strip()),
                "resid": float(parts[3].strip()),
                "stres": float(parts[4].strip()),
                "student": parts[5].strip() if len(parts) > 5 else "",
                "criterion": parts[-1].strip() if len(parts) > 6 else "",
            })
        except (ValueError, IndexError):
            continue


# ════════════════════════════════════════════════════════════
# 中文报告生成
# ════════════════════════════════════════════════════════════

def generate_report(parsed: dict) -> str:
    """从解析结果生成中文 Markdown 报告"""
    h = parsed["header"]
    t5 = parsed["table5"]
    lines: list[str] = []

    title = h.get("title", "MFRM 分析报告")
    lines.append(f"# {title}")
    lines.append("")
    lines.append("## 一、报告摘要")
    lines.append("")

    n = t5.get("n", 0)
    obs = t5.get("obs_mean", 0)
    exp = t5.get("exp_mean", 0)
    ve = t5.get("var_exp", 0)
    rv = t5.get("resid_var", 0)
    lines.append(f"- 有效反应数: **{n}** 条")
    lines.append(f"- 观察均值: **{obs:.2f}**，期望均值: **{exp:.2f}**，差异: **{abs(obs-exp):.3f}**")
    lines.append(f"- Rasch 解释方差: **{ve:.2f}%**，残差方差: {rv:.2f}%")
    stress = t5.get("stres_sd", 0)
    lines.append(f"- 标准化残差 SD: **{stress:.2f}**")
    lines.append("")

    # ── 模型设定 ──
    lines.append("## 二、模型设定")
    lines.append("")
    lines.append(f"| 设定项 | 内容 | 解释 |")
    lines.append(f"|--------|------|------|")
    lines.append(f"| Facets | {h.get('facets', '?')} | 面向数 |")
    lines.append(f"| Model | `{h.get('model', '?')}` | 评分模型 |")
    lines.append(f"| Noncentered | {h.get('noncentered', '?')} | 约束面 |")
    lines.append(f"| Positive | {h.get('positive', '?')} | 高分代表高能力 |")
    lines.append("")

    # ── 各面向 ──
    for facet_key, title_str, emoji in [
        ("students", "学生面向", "🎓"),
        ("raters", "评分者面向", "👤"),
        ("criteria", "评分标准面向", "📋"),
    ]:
        fd = parsed.get(facet_key, {})
        if not fd.get("rows"):
            continue
        lines.append(f"## {emoji} {title_str}")
        lines.append("")
        lines.append(f"| 名称 | 总分 | ObsAvg | FairAvg | Measure | SE | Infit | Outfit |")
        lines.append(f"|------|------|--------|---------|---------|-----|-------|--------|")
        for r in fd["rows"]:
            lines.append(
                f"| {r['label']} | {r['total']} | {r['obs_avg']:.2f} | "
                f"{r['fair_avg']:.2f} | {r['meas']:.2f} | {r['se']:.2f} | "
                f"{r['infit']:.2f} | {r['outfit']:.2f} |"
            )
        lines.append("")
        sep = fd.get("separation", 0)
        rel = fd.get("reliability", 0)
        chi = fd.get("chi_sq", "")
        lines.append(f"- Separation = **{sep:.2f}**，Reliability = **{rel:.2f}**")
        if chi:
            lines.append(f"- {chi}")
        lines.append("")

    # ── 评分等级分析 ──
    cats = parsed.get("categories", [])
    if cats:
        lines.append("## 评分等级类别分析")
        lines.append("")
        lines.append(f"| 分数 | 次数 | % | Avg Meas | Exp Meas | Outfit | Threshold |")
        lines.append(f"|------|------|---|----------|----------|--------|-----------|")
        for c in cats:
            lines.append(
                f"| {c['score']} | {c['count']} | {c['pct']:.0f}% | "
                f"{c['avg_meas']:.2f} | {c['exp_meas']:.2f} | "
                f"{c['outfit']:.2f} | {c['threshold']:.2f} |"
            )

        # 诊断结论
        zero_cats = [c for c in cats if c["count"] == 0]
        low_cats = [c for c in cats if 0 < c["count"] <= 2]
        thresholds = [c["threshold"] for c in cats if c["count"] > 0]
        disordered = sum(1 for i in range(1, len(thresholds)) if thresholds[i] < thresholds[i-1])
        lines.append("")
        if zero_cats:
            lines.append(f"⚠️ **{len(zero_cats)} 个等级未使用** (分数: {', '.join(str(c['score']) for c in zero_cats)})")
        if low_cats:
            lines.append(f"⚠️ **{len(low_cats)} 个等级使用次数 ≤2**，建议考虑合并")
        if disordered > 0:
            lines.append(f"⚠️ **{disordered} 处阈值无序**，评分者无法稳定区分相邻等级")
        if not zero_cats and not low_cats and disordered == 0:
            lines.append("✅ 评分等级结构良好，无需合并。")
        lines.append("")

    # ── 异常反应 ──
    unexpected = parsed.get("unexpected", [])
    if unexpected:
        lines.append("## 异常反应记录")
        lines.append("")
        lines.append(f"| Score | Exp | Resid | StRes | 考生 | 标准 |")
        lines.append(f"|-------|-----|-------|-------|------|------|")
        for u in unexpected[:10]:
            lines.append(
                f"| {u['score']} | {u['exp']:.1f} | {u['resid']:.1f} | "
                f"{u['stres']:.1f} | {u.get('student', '')} | {u.get('criterion', '')} |"
            )
        lines.append("")

    # ── 结论 ──
    lines.append("## 综合结论")
    lines.append("")
    if ve >= 75:
        lines.append(f"- 模型整体拟合**良好**，Rasch 解释方差 {ve:.1f}%")
    elif ve >= 60:
        lines.append(f"- 模型整体拟合**可接受**，Rasch 解释方差 {ve:.1f}%")
    else:
        lines.append(f"- 模型整体拟合**需要关注**，Rasch 解释方差仅 {ve:.1f}%")

    if abs(obs - exp) < 0.1:
        lines.append("- 观察均值与期望均值**高度一致**，无系统性偏差")
    elif abs(obs - exp) < 0.5:
        lines.append("- 观察均值与期望均值**基本一致**")
    else:
        lines.append("- ⚠️ 观察均值与期望均值差异较大，请检查模型设定")

    zero_c = len([c for c in cats if c["count"] == 0])
    if zero_c > 2:
        lines.append(f"- ⚠️ 建议**合并评分等级**：当前 {zero_c} 个等级未使用，等级划分过细")
    lines.append("")

    return "\n".join(lines)
