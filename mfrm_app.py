#!/usr/bin/env python3
"""
MFRMSight v0.8.0 — 多面Rasch模型分析工具 (单文件版)
=====================================================
用法:
  python mfrm_app.py data.txt       命令行 (Minifac .txt)
  python mfrm_app.py data.xlsx      命令行 (Excel, 会询问面向数)
  python mfrm_app.py -w             启动 Gradio Web 界面
  python mfrm_app.py -s             启动纯 HTTP Web 界面

Minifac .txt 格式 (完全兼容):
  Facets=4
  Positive=1
  Noncentered=1
  Model=?,?,?,?,R23
  *
  Labels=
  1,Students
  1,Student1
  ...
  *
  2,Raters
  ...
  *
  Data=
  1,1,1,1,8
  ...

版本历史:
  0.6.0  增强 Minifac 解析: Model解析, Rating Scale块, Labels增强,
          Data索引展开, 扩展Header, Engine支持D/R双模型
  0.5.0  Minifac完整适配 + 交互式面向询问 (CLI & GUI)
  0.4.1  Excel解析增强, 纯HTTP Web版
  0.4.0  Gradio Web界面, 单文件重构, PROX U因子修复
  0.3.0  Fisher-scoring JMLE 参数稳定化
"""
import sys, os, re, tempfile
from pathlib import Path
import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════
# 1. Minifac .txt 解析器 (完全兼容)
# ═══════════════════════════════════════════════════════════════════════

def parse_minifac_txt(path: str) -> dict:
    """
    完整解析 Minifac .txt 格式，支持所有常见变体。

    增强功能 (v0.8.0):
      - Model 字符串结构化解析 → model_info
      - Rating scale 块解析 (命名量表 + 类别标签)
      - Labels: 范围语法 (1_17), 锚定面向 (,A), 锚定值 (,0)
      - Data: 索引展开 (2_4), R 复制前缀 (R3)
      - 扩展 Header 指令 (Convergence, Arrange 等)
      - 外部数据文件引用 (Data = filename.txt)

    返回 data dict，完全向后兼容 v0.8.0。
    """
    with open(path, encoding="utf-8") as f:
        raw_lines = f.readlines()

    # 保留空行用于段检测，但去掉每行首尾空白
    lines = [l.strip() for l in raw_lines]

    # ═══════════════════════════════════════════════════════════
    # Header 变量
    # ═══════════════════════════════════════════════════════════
    n_facets = 4
    model = ""
    title = ""
    positive = 1
    noncentered = 1
    convergence: list[float] | None = None
    inter_rater: int | None = None
    arrange: str = "mN"
    pt_biserial: str = "measure"
    umean: list[float] | None = None
    unexpected: float | None = None
    vertical: str = ""
    yard: str = ""
    delements: str = "N"
    score_files: str = ""
    zscore: list[float] | None = None
    data_file: str = ""          # 外部数据文件名
    xtreme: list[float] | None = None

    # ═══════════════════════════════════════════════════════════
    # Rating Scale 块
    # ═══════════════════════════════════════════════════════════
    rating_scale_info: dict | None = None

    # ═══════════════════════════════════════════════════════════
    # Labels 变量
    # ═══════════════════════════════════════════════════════════
    labels_info: list[list[tuple[int, str]]] = [[], [], [], []]  # facet 0-3
    declared_facets: set[int] = set()       # 已声明的面向编号，防止重复/回退
    facet_meta: list[dict] = []           # 面向元信息: {name, type: active|anchor|dummy}
    element_anchors: dict[int, list] = {} # facet_idx → [anchor_val or None, ...]
    element_groups: dict[int, list] = {}  # facet_idx → [group_num, ...]

    # ═══════════════════════════════════════════════════════════
    # Dvalues
    # ═══════════════════════════════════════════════════════════
    dvalues_info: list[dict] = []         # [{from_facet, from_field, to_facet, to_field}, ...]

    # ═══════════════════════════════════════════════════════════
    # 状态机解析
    # ═══════════════════════════════════════════════════════════
    state = "header"       # header | rating_scale | labels | dvalues | data
    current_facet = 0
    data_rows: list[list[int]] = []
    line_idx = 0
    header_finished = False  # Header 段是否已结束

    while line_idx < len(lines):
        line = lines[line_idx]

        # ── 跳过注释和空行 ──
        if not line or line.startswith(";"):
            line_idx += 1
            continue

        # ═══════════════════════════════════════════════════════
        # HEADER 状态
        # ═══════════════════════════════════════════════════════
        if state == "header":
            lower = line.lower()

            if lower.startswith("title=") or lower.startswith("title "):
                title = line.split("=", 1)[1].strip()
            elif lower.startswith("facets=") or lower.startswith("facets "):
                n_facets = int(line.split("=", 1)[1].strip().split()[0])
            elif lower.startswith("models=") or lower.startswith("model=") or lower.startswith("model "):
                model = line.split("=", 1)[1].strip()
            elif lower.startswith("positive=") or lower.startswith("positive "):
                try: positive = int(line.split("=", 1)[1].strip().split(",")[0])
                except: pass
            elif lower.startswith("noncent") and "=" in line:
                try: noncentered = int(line.split("=", 1)[1].strip())
                except: pass
            elif lower.startswith("inter-rater=") or lower.startswith("inter_rater="):
                try: inter_rater = int(line.split("=", 1)[1].strip())
                except: pass
            elif lower.startswith("converge"):
                try:
                    vals = line.split("=", 1)[1].strip()
                    convergence = [float(x.strip()) for x in vals.replace(",", " ").split() if x.strip()]
                except: pass
            elif lower.startswith("arrange"):
                arrange = line.split("=", 1)[1].strip()
            elif lower.startswith("pt-biserial") or lower.startswith("pt_biserial"):
                pt_biserial = line.split("=", 1)[1].strip().lower()
            elif lower.startswith("umean"):
                try:
                    vals = line.split("=", 1)[1].strip()
                    umean = [float(x.strip()) for x in vals.replace(",", " ").split() if x.strip()]
                except: pass
            elif lower.startswith("unexpected"):
                try: unexpected = float(line.split("=", 1)[1].strip())
                except: pass
            elif lower.startswith("vertical"):
                vertical = line.split("=", 1)[1].strip()
            elif lower.startswith("yard"):
                yard = line.split("=", 1)[1].strip()
            elif lower.startswith("delements"):
                delements = line.split("=", 1)[1].strip()
            elif lower.startswith("score file") or lower.startswith("scorefiles"):
                score_files = line.split("=", 1)[1].strip()
            elif lower.startswith("zscore"):
                try:
                    vals = line.split("=", 1)[1].strip()
                    zscore = [float(x.strip()) for x in vals.replace(",", " ").split() if x.strip()]
                except: pass
            elif lower.startswith("xtreme"):
                try:
                    vals = line.split("=", 1)[1].strip()
                    xtreme = [float(x.strip()) for x in vals.replace(",", " ").split() if x.strip()]
                except: pass
            elif lower.startswith("rating scale") or lower.startswith("rating scale="):
                # 进入 Rating Scale 子状态
                state = "rating_scale"
                # 回退一行，让 rating_scale 处理
                continue
            elif line == "*" or line.startswith("*"):
                # 第一个 * 分隔符 → Labels or Dvalues
                state = "labels"
                current_facet = 0
            # 其他 header 行（如 "Barchart = Yes"）静默忽略

            line_idx += 1
            continue

        # ═══════════════════════════════════════════════════════
        # RATING SCALE 子状态
        # ═══════════════════════════════════════════════════════
        if state == "rating_scale":
            rating_scale_info, line_idx = parse_rating_scale_block(lines, line_idx)
            state = "labels"
            current_facet = 0
            continue

        # ═══════════════════════════════════════════════════════
        # LABELS 状态
        # ═══════════════════════════════════════════════════════
        if state == "labels":
            lower = line.lower()

            # Rating scale 块可能出现在 * 之后、Labels 之间
            # (如 Pair.txt: Rating scale = reading,R9,K 在 Models 后)
            if lower.startswith("rating scale") and "=" in line:
                rating_scale_info, line_idx = parse_rating_scale_block(lines, line_idx)
                continue

            # Data 段开始
            if (lower.startswith("data=") or lower.startswith("data ")
                    or line.startswith("Data=")):
                state = "data"
                rhs = line.split("=", 1)[1].strip() if "=" in line else ""
                if rhs and not rhs.startswith(";") and not rhs.startswith("*"):
                    data_file = rhs.strip().strip('"')
                line_idx += 1
                if data_file and data_file.endswith('.txt'):
                    data_path = Path(path).parent / data_file
                    if data_path.exists():
                        with open(data_path, encoding="utf-8") as df:
                            for dline in df:
                                dline = dline.strip()
                                if dline and not dline.startswith(";"):
                                    expanded = _expand_data_line(
                                        [x.strip() for x in dline.split(",")], n_facets)
                                    if expanded:
                                        data_rows.extend(expanded)
                continue

            # Dvalues 段开始
            if lower.startswith("dvalues="):
                state = "dvalues"
                line_idx += 1
                continue

            # 面向分隔符 (在 Minifac 规范中用于分隔各面向)
            if line == "*":
                # 切换到下一个面向
                current_facet += 1
                line_idx += 1
                continue

            # 跳过 Labels= 行和纯数字声明行
            if lower.startswith("labels="):
                line_idx += 1
                continue
            if re.match(r'^\d+$', line):
                line_idx += 1
                continue

            # 面向声明检测
            # 用 declared_facets 跟踪已声明的面向
            # 策略：在当前面向范围内，以 "N,非数字名称" 开头的行是面声明
            next_facet = current_facet + 1
            is_facet_decl = False
            if line.startswith(f"{next_facet},") and next_facet not in declared_facets:
                rest = line.split(",", 1)[1].strip().rstrip(";").strip()
                # 分割逗号后，第一部分如果不是纯数字，就是面向名称
                first_part = rest.split(",")[0].strip() if rest else ""
                # 如果第一个部分是数字，这可能是元素标签（如 "1,Student1" → rest="Student1" → first_part="Student1"）
                # 如果第一个部分包含锚定标记，提取名称
                if first_part and not re.match(r'^\d+$', first_part):
                    current_facet = next_facet - 1
                    declared_facets.add(next_facet)
                    is_facet_decl = True

                    # 检测面向类型: ,A (锚定) 或 ,D (虚拟)
                    ftype = "active"
                    rest_parts = [p.strip() for p in rest.split(",")]
                    if len(rest_parts) > 1:
                        last = rest_parts[-1].upper()
                        if last == "A":
                            ftype = "anchor"
                        elif last == "D":
                            ftype = "dummy"

                    # 确保 facet_meta 有足够条目
                    while len(facet_meta) <= current_facet:
                        facet_meta.append({"name": "", "type": "active"})
                    facet_meta[current_facet] = {"name": first_part, "type": ftype}
            if is_facet_decl:
                line_idx += 1
                continue

            # 元素标签: "1, Student1" 或 "1=Student1,0" 或 "1_17=Boy,,1"
            if current_facet < 4 and ("," in line or "=" in line):
                # 用 = 优先分割
                sep = "=" if "=" in line else ","
                id_part, label_part = line.split(sep, 1)

                # id_part 必须是数字或范围
                if not re.match(r'^\d+$', id_part.strip()) and not re.match(r'^\d+[_-]\d+', id_part.strip()):
                    line_idx += 1
                    continue

                # 检查 id_part 是否为范围: "1_17"
                range_match = re.match(r'^(\d+)[_-](\d+)$', id_part.strip())
                if range_match:
                    # 范围标签: "1_17 = Boy,,1"
                    lo, hi = int(range_match.group(1)), int(range_match.group(2))
                    label_full = label_part.strip().rstrip(";").strip()
                    # 解析标签的各个字段
                    label_fields = [f.strip() for f in label_full.split(",")]
                    base_label = label_fields[0] if label_fields else ""
                    group_num = None
                    # 组号通常在最后，如 ",1" 或 ",,1"
                    if len(label_fields) > 1 and label_fields[-1].isdigit():
                        group_num = int(label_fields[-1])
                    elif len(label_fields) > 2 and label_fields[-1].isdigit():
                        group_num = int(label_fields[-1])

                    for eid in range(lo, hi + 1):
                        # 如果标签前缀是 "Boy" → 每个元素的标签为 "Boy" (共享)
                        lbl = base_label if base_label else ""
                        labels_info[current_facet].append((eid, lbl))
                        # 记录分组
                        if group_num is not None:
                            while len(element_groups.get(current_facet, [])) < eid:
                                element_groups.setdefault(current_facet, []).append(0)
                            # will pad later
                else:
                    # 单元素标签: "1, Student1" 或 "1=boys,0"
                    try:
                        elem_id = int(id_part.strip())
                    except ValueError:
                        line_idx += 1
                        continue

                    label_full = label_part.strip().rstrip(";").strip()
                    label_fields = [f.strip() for f in label_full.split(",")]

                    # 提取锚定值（如果有）
                    anchor_val = None
                    if len(label_fields) > 1:
                        # 检测最后一个字段是否为数字（锚定值）
                        try:
                            anchor_val = float(label_fields[-1].replace(",", "."))
                        except ValueError:
                            anchor_val = None

                    # 标签文本
                    label_text = label_fields[0] if label_fields else ""

                    # 组号
                    group_num = None
                    if len(label_fields) > 2:
                        try:
                            group_num = int(label_fields[-2])
                        except ValueError:
                            pass

                    labels_info[current_facet].append((elem_id, label_text))

                    # 存储锚定值
                    while len(element_anchors.get(current_facet, [])) < elem_id:
                        element_anchors.setdefault(current_facet, []).append(None)
                    if anchor_val is not None:
                        lst = element_anchors.setdefault(current_facet, [])
                        while len(lst) < elem_id:
                            lst.append(None)
                        lst[elem_id - 1] = anchor_val

            line_idx += 1
            continue

        # ═══════════════════════════════════════════════════════
        # DVALUES 子状态
        # ═══════════════════════════════════════════════════════
        if state == "dvalues":
            lower = line.lower()
            if lower.startswith("data="):
                state = "data"
                line_idx += 1
                continue
            if line == "*":
                state = "labels"
                line_idx += 1
                continue
            # 解析: "3, 1, 9, 1" → 面向3的元素从面向1标签的第9列提取
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                dvalues_info.append({
                    "to_facet": int(parts[0]) - 1,
                    "from_facet": int(parts[1]) - 1,
                    "from_field": int(parts[2]),
                    "count": int(parts[3]),
                })
            line_idx += 1
            continue

        # ═══════════════════════════════════════════════════════
        # DATA 状态
        # ═══════════════════════════════════════════════════════
        if state == "data":
            # 同时支持逗号和空格/制表符分隔
            items = [x.strip() for x in line.split(",")]
            if len(items) < n_facets + 1:
                # 尝试空格分隔
                items = [x.strip() for x in line.split()]
            if len(items) >= n_facets + 1:
                expanded_rows = _expand_data_line(items, n_facets)
                if expanded_rows:
                    for row in expanded_rows:
                        if len(row) >= n_facets + 1:
                            try:
                                data_rows.append([int(x) for x in row[:n_facets + 1]])
                            except ValueError:
                                pass
            line_idx += 1
            continue

    # ═══════════════════════════════════════════════════════════
    # 解析 Model 字符串
    # ═══════════════════════════════════════════════════════════
    model_info = parse_model_string(model, n_facets)

    # ═══════════════════════════════════════════════════════════
    # 构建 raw 数组
    # ═══════════════════════════════════════════════════════════
    raw = np.array(data_rows) if data_rows else np.empty((0, n_facets + 1), dtype=int)

    if len(raw) == 0:
        # 可能数据在外部文件中但路径无效，或者文件使用外部数据源
        if data_file:
            raise ValueError(
                f"未找到数据行。规范文件指向外部数据文件 \"{data_file}\"，"
                f"但该文件不存在或无法读取。"
            )
        raise ValueError("未找到有效数据行")

    n_s = int(raw[:, 0].max()) if raw.shape[1] >= 1 else 0
    n_r = int(raw[:, 1].max()) if raw.shape[1] >= 2 else 0
    n_c = int(raw[:, 2].max()) if raw.shape[1] >= 3 else 0
    n_i = int(raw[:, 3].max()) if n_facets >= 4 and raw.shape[1] >= 4 else 1

    # 从 model_info 覆盖分数范围（如果可用）
    # R23 → max_score=22，但 min_score 应信任实际数据
    mi = model_info
    if mi.get("max_score") is not None:
        mx = mi["max_score"]
        # 二分模型：完全使用 model 指定的范围
        # Rating Scale：max 来自 model，min 信任数据
        if mi.get("model_type") == "dichotomous":
            mn = mi.get("min_score", 0)
        else:
            mn = int(raw[:, -1].min())
    else:
        mx = int(raw[:, -1].max())
        mn = int(raw[:, -1].min())

    # ═══════════════════════════════════════════════════════════
    # 构建标签列表
    # ═══════════════════════════════════════════════════════════
    def _build_labels(lb_info: list, n_elem: int, prefix: str) -> list[str]:
        """从标签信息构建标签列表，缺位的用默认前缀填充"""
        result = [""] * n_elem
        for eid, label in lb_info:
            if 1 <= eid <= n_elem:
                result[eid - 1] = label
        for i in range(n_elem):
            if not result[i]:
                result[i] = f"{prefix}{i + 1}"
        return result

    s_labels = _build_labels(labels_info[0], n_s, "Student")
    r_labels = _build_labels(labels_info[1], n_r, "Rater")
    c_labels = _build_labels(labels_info[2], n_c, "Criterion")
    i_labels = _build_labels(labels_info[3], max(n_i, 1), "Item")

    # 降级标签处理：如果 labels_info 有标签但所有标签都是空字符串，
    # 说明解析产生了空条目，使用实际存在的标签
    for fi in range(4):
        existing = [l for _, l in labels_info[fi] if l and l.strip()]
        if existing:
            if fi == 0 and len(existing) >= n_s:
                s_labels = existing[:n_s]
            elif fi == 1 and len(existing) >= n_r:
                r_labels = existing[:n_r]
            elif fi == 2 and len(existing) >= n_c:
                c_labels = existing[:n_c]
            elif fi == 3 and len(existing) >= max(n_i, 1):
                i_labels = existing[:max(n_i, 1)]

    # 确保标签列表长度匹配
    s_labels = s_labels[:n_s] if len(s_labels) >= n_s else s_labels + [
        f"Student{i}" for i in range(len(s_labels) + 1, n_s + 1)]
    r_labels = r_labels[:n_r] if len(r_labels) >= n_r else r_labels + [
        f"Rater{i}" for i in range(len(r_labels) + 1, n_r + 1)]
    c_labels = c_labels[:n_c] if len(c_labels) >= n_c else c_labels + [
        f"Criterion{i}" for i in range(len(c_labels) + 1, n_c + 1)]
    i_labels = i_labels[:max(n_i, 1)] if len(i_labels) >= max(n_i, 1) else i_labels + [
        f"Item{i}" for i in range(len(i_labels) + 1, max(n_i, 1) + 1)]

    # ═══════════════════════════════════════════════════════════
    # 确保 facet_meta 完整
    # ═══════════════════════════════════════════════════════════
    default_names = [f"Facet{i+1}" for i in range(n_facets)]
    name_map = {0: "Students", 1: "Raters", 2: "Criteria", 3: "Items"}
    for idx, name in name_map.items():
        if idx < n_facets:
            default_names[idx] = name

    while len(facet_meta) < n_facets:
        idx = len(facet_meta)
        facet_meta.append({"name": default_names[idx], "type": "active"})

    # ═══════════════════════════════════════════════════════════
    # 返回完整 dict
    # ═══════════════════════════════════════════════════════════
    result = {
        # === 核心数据（现有，完全向后兼容）===
        "raw": raw, "nf": n_facets, "N": len(raw),
        "ns": n_s, "nr": n_r, "nc": n_c, "ni": max(n_i, 1),
        "mn": mn, "mx": mx,
        "sl": s_labels, "rl": r_labels,
        "cl": c_labels, "il": i_labels,
        "model": model, "title": title, "positive": positive,

        # === 新增：结构化模型信息 ===
        "model_info": model_info,

        # === 新增：Rating Scale 定义 ===
        "rating_scale": rating_scale_info or {
            "name": f"R{mx + 1}" if mx else "unknown",
            "categories": int(mx - mn + 1) if mx and mn else 0,
            "keep_unobserved": False,
            "category_labels": {},
            "renumbered": {},
        },

        # === 新增：扩展元数据 ===
        "facet_meta": facet_meta,
        "element_anchors": element_anchors,
        "element_groups": element_groups,
        "convergence": convergence,
        "inter_rater": inter_rater,
        "noncentered": noncentered,
        "arrange": arrange,
        "pt_biserial": pt_biserial,
        "umean": umean,
        "unexpected": unexpected,
        "vertical": vertical,
        "yard": yard,
        "delements": delements,
        "score_files": score_files,
        "zscore": zscore,
        "xtreme": xtreme,
        "dvalues": dvalues_info,
    }
    return result


# ═══════════════════════════════════════════════════════════════════════
# 1.5 Model 字符串解析 & 数据展开工具
# ═══════════════════════════════════════════════════════════════════════

def parse_model_string(model: str, n_facets: int) -> dict:
    """
    解析 Minifac Model 字符串为结构化信息。

    Model 语法:
      ?,?,D              → 二分模型
      ?,?,?,?,R23        → Rating Scale (分数 1~22)
      ?,?B,?B,?,R9       → Rating Scale + Bias 分析
      ?,?,?,?,Creativity → 命名量表 (配合 Rating Scale 块)

    返回:
      { "facets": [...], "model_type": "dichotomous"|"rating_scale"|"poisson",
        "score_model": "R23", "max_score": 22, "min_score": 1,
        "has_bias": [...], "scale_name": None, ... }
    """
    if not model:
        return {
            "facets": ["?"] * n_facets, "model_type": "rating_scale",
            "score_model": "", "max_score": None, "min_score": None,
            "has_bias": [False] * n_facets, "is_missing_model": False,
            "scale_name": None,
        }

    # 按逗号分割，最后一段是评分模型标识
    parts = [p.strip() for p in model.split(",")]
    score_model = parts[-1] if parts else ""

    # ── 判断模型类型 ──
    model_type = "rating_scale"
    max_score = None
    min_score = 1          # 默认最小分
    scale_name = None
    is_missing = False

    r_match = re.match(r'^R(\d+)', score_model, re.IGNORECASE)
    if r_match:
        model_type = "rating_scale"
        max_score = int(r_match.group(1)) - 1   # R23 → max=22
    elif re.match(r'^D\b', score_model, re.IGNORECASE):
        model_type = "dichotomous"
        max_score = 1
        min_score = 0
    elif re.match(r'^P\b', score_model, re.IGNORECASE):
        model_type = "poisson"
    elif re.match(r'^M\b', score_model, re.IGNORECASE):
        is_missing = True
    elif score_model and not score_model[0].isdigit():
        # 非数字开头 → 命名量表 (如 "Creativity", "reading")
        # 只取第一个词
        scale_name = score_model.split()[0]

    # ── 解析各面向列的规范 ──
    facet_parts = parts[:-1] if len(parts) > 1 else []
    has_bias: list[bool] = []
    facet_specs: list[str] = []

    for fp in facet_parts:
        fp = fp.strip()
        has_bias.append("B" in fp.upper() and "?" in fp)  # ?B → bias
        facet_specs.append(fp)

    # 补齐到 n_facets
    while len(facet_specs) < n_facets:
        facet_specs.append("?")
        has_bias.append(False)

    return {
        "facets": facet_specs[:n_facets],
        "model_type": model_type,
        "score_model": score_model,
        "max_score": max_score,
        "min_score": min_score,
        "has_bias": has_bias[:n_facets],
        "is_missing_model": is_missing,
        "scale_name": scale_name,
    }


def parse_rating_scale_block(lines: list[str], start_idx: int) -> tuple[dict, int]:
    """
    解析 Rating scale = ... 定义块。
    返回 (info_dict, 下一个索引)。

    支持的格式:
      Rating scale = Creativity,R9          → 9类别，命名 "Creativity"
      Rating scale = reading,R9,K           → Keep 未出现的中间类别
      Rating scale = DoublePoints,R20,Keep  → 20类别 + Keep
      0 = 0.0                               → 类别重编号
      10 = 5.0
      1 = lowest                            → 类别标签
    """
    info: dict = {
        "name": "", "categories": 0, "keep_unobserved": False,
        "category_labels": {}, "renumbered": {},
    }

    line = lines[start_idx].strip()
    rhs = line.split("=", 1)[1].strip()
    r_parts = [p.strip() for p in rhs.split(",")]

    if r_parts:
        info["name"] = r_parts[0]
    if len(r_parts) > 1:
        r_match = re.match(r'R(\d+)', r_parts[1], re.IGNORECASE)
        if r_match:
            info["categories"] = int(r_match.group(1))
    if len(r_parts) > 2:
        # "K" 或 "Keep" → 保留未出现类别
        third = r_parts[2].strip()
        info["keep_unobserved"] = third.upper().startswith("K")

    # 读取类别标签，直到遇到单独的 * 行
    idx = start_idx + 1
    while idx < len(lines):
        cur = lines[idx].strip()
        idx += 1
        if cur == "*":
            break
        if "=" in cur:
            cat_str, label = cur.split("=", 1)
            cat_str = cat_str.strip()
            label = label.strip().rstrip(";")
            try:
                cat_num = int(cat_str)
                # 检测是重编号 (数字) 还是标签 (文本)
                try:
                    float(label.replace(",", "."))
                    info["renumbered"][cat_num] = float(label.replace(",", "."))
                except ValueError:
                    info["category_labels"][cat_num] = label
            except ValueError:
                pass

    return info, idx


def _expand_range_token(token: str) -> list[int] | None:
    """展开范围 token: '2_4' → [2,3,4], '1-3' → [1,2,3], 纯数字 → [num]。无法解析返回 None。"""
    token = token.strip()
    if re.match(r'^\d+$', token):
        return [int(token)]
    m = re.match(r'^(\d+)[_-](\d+)$', token)
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
        return list(range(lo, hi + 1))
    return None


def _expand_data_line(tokens: list[str], n_facets: int) -> list[list[int]] | None:
    """
    将含范围 token 的数据行展开为标准 [f1, f2, ..., fN, score] 列表。

    处理:
      1. "1,2_4,1,1,1,0" → 范围 2_4 = items 2-4，后面 1,1,0 是对应分数
      2. "R3,1,1,1,5"    → R3 = 复制 3 次
      3. 标准行直接返回

    返回 None 表示该行无法解析（应跳过）。
    """
    if not tokens:
        return None

    # ── 检测 R 复制前缀 ──
    reps = 1
    start_idx = 0
    r_match = re.match(r'^R(\d+)$', tokens[0].strip(), re.IGNORECASE)
    if r_match:
        reps = int(r_match.group(1))
        start_idx = 1

    # 提取剩余的 token
    remaining = [t.strip() for t in tokens[start_idx:]]

    # ── 找到所有范围 token 及其位置 ──
    # 思路：前 n_facets 个位置是面向元素 ID，之后的是分数
    # 如果某个 ID 位置是范围，则后续分数需要被分配

    # 先扫描所有 token，找到范围的位置
    id_tokens = []         # 面向 ID 部分
    score_tokens = []      # 分数部分
    range_positions = []   # (id_index, expanded_values)
    in_scores = False

    # 启发式：遍历 token，如果遇到范围或数字个数超过 n_facets，切换模式
    for i, t in enumerate(remaining):
        expanded = _expand_range_token(t)
        if expanded is None:
            # 可能是空 token 或格式错误
            if t == "" or t == "-":
                continue
            return None  # 无法解析

        if len(expanded) == 1 and not in_scores:
            id_tokens.append(expanded[0])
        elif len(expanded) > 1 and not in_scores:
            range_positions.append((len(id_tokens), expanded))
            id_tokens.append(expanded[0])  # 占位
        else:
            # 已进入分数区域
            score_tokens.extend(expanded)
            in_scores = True

    # 如果还没有确定分数区域，则根据 n_facets 判断
    if not in_scores and len(id_tokens) > n_facets:
        # 多出来的 token 是分数
        score_tokens = id_tokens[n_facets:]
        id_tokens = id_tokens[:n_facets]

    # 补齐 ID tokens 到 n_facets（如果不够）
    while len(id_tokens) < n_facets:
        id_tokens.append(1)  # 默认第一个元素

    # ── 展开为完整行 ──
    rows = []
    if not range_positions:
        # 没有范围 → 标准 1 行模式
        if len(score_tokens) == 1:
            rows.append(id_tokens[:n_facets] + [score_tokens[0]])
        elif len(score_tokens) == 0 and len(id_tokens) == n_facets + 1:
            # id_tokens 中最后一个就是分数
            rows.append(id_tokens[:n_facets] + [id_tokens[-1]])
        else:
            # 可能是多分数行
            for si, s_val in enumerate(score_tokens):
                row = id_tokens[:]
                # 替换范围位置的值为当前子值
                rows.append(row[:n_facets] + [s_val])
    else:
        # 有范围 → 需要展开
        # 确定总行数：range 的长度 = score_tokens 的长度
        n_scores = len(score_tokens)
        if n_scores == 0:
            n_scores = 1

        # 处理单个范围的情况
        if len(range_positions) == 1:
            pos, expanded = range_positions[0]
            for ei, elem_val in enumerate(expanded):
                row = id_tokens[:n_facets]
                row[pos] = elem_val
                s_val = score_tokens[ei] if ei < n_scores else score_tokens[-1]
                rows.append(row + [s_val])
        else:
            # 多个范围 → 笛卡尔积（复杂情况，先不处理）
            pass

    # ── 复制 ──
    if reps > 1:
        rows = rows * reps

    return rows if rows else None

def _find_header_row(df: pd.DataFrame) -> int:
    """自动检测表头行 (BUG-008 增强: 支持多行标题自动合并)"""
    for i in range(min(5, len(df))):
        vals = [str(v) for v in df.iloc[i] if pd.notna(v)]
        if any("编号" in s or "评分人" in s or "题目" in s for s in vals):
            # 检测 N+1 行是否也是表头 (如 "行号/评分人/criterion1/criterion1/..." 在第一行,
            # 真正的 "编号/评分人/沟通能力-1号/沟通能力-2号/..." 在第二行)
            if (i + 1 < len(df) and
                any("编号" in str(v) or "评分" in str(v) or "-" in str(v) or "能力" in str(v)
                    for v in df.iloc[i + 1] if pd.notna(v)) and
                len([v for v in df.iloc[i + 1] if pd.notna(v)]) >
                len([v for v in df.iloc[i] if pd.notna(v)])):
                return i + 1  # 下一行更具体
            return i
    return 1


def _merge_header_rows(df: pd.DataFrame, hr: int) -> int:
    """BUG-008: 检测并合并多行标题。

    如果标题有两行: 第一行是分组标题 (如 "沟通能力", "逻辑思维"),
    第二行是具体列名 (如 "1号", "2号"), 则合并为 "沟通能力-1号" 等。
    返回合并后的表头行索引 (通常是最后一行)。
    """
    if hr == 0 or len(df) <= hr:
        return hr

    row_above = [str(v) for v in df.iloc[hr - 1] if pd.notna(v)]
    row_header = [str(v) for v in df.iloc[hr] if pd.notna(v)]
    # 如果上一行也有含义丰富的文本 (不含数字/元信息)
    if (len(row_above) >= 2 and
        not any("编号" in s or "评分" in s for s in row_above)):
        # 上一行可能是分组标题 → 合并到当前行
        for j in range(df.shape[1]):
            above_val = df.iloc[hr - 1, j]
            header_val = df.iloc[hr, j]
            if pd.notna(above_val) and pd.notna(header_val):
                above_s = str(above_val).strip()
                header_s = str(header_val).strip()
                # 只合并有意义的不同字符串, 避免重复
                if above_s and above_s != header_s and not above_s.isdigit():
                    df.iloc[hr, j] = f"{above_s}-{header_s}"
    return hr


def _guess_facet_from_name(name: str) -> str:
    """从列名猜测是哪个面向 (BUG-002 增强版)。

    增强: "综合分析-1号" 中 "1号" 可能=Student 或 Item。
    双模式: 综合后缀关键词 + 连字符结构分析。
    """
    name_lower = name.lower()

    # 强信号关键词 → 直接判定
    keywords = {
        "student": ["学生", "student", "考生", "被试", "面试者", "examinee", "学员", "受试"],
        "rater":   ["评分人", "评分者", "评委", "rater", "考官", "评价者", "judge", "examiner"],
        "criterion": ["标准", "维度", "criterion", "方面", "competency", "domain"],
        "item": ["题目", "item", "题号", "项目", "task", "评分标准", "rubric"],
    }
    scores = {k: 0 for k in keywords}
    for cat, terms in keywords.items():
        for t in terms:
            if t in name_lower:
                scores[cat] += 1

    # 弱信号: 连字符右侧后缀
    if "-" in name or "—" in name:
        sep = "-" if "-" in name else "—"
        parts = name.rsplit(sep, 1) if name.count(sep) == 1 else (name.split(sep)[0], name.split(sep)[-1])
        suffix = parts[-1]
        # 后缀含"题"→item, 含"号"/"人"/"生"→student
        if any(kw in suffix for kw in ["题", "item"]):
            scores["item"] += 1
        elif any(kw in suffix for kw in ["号", "人", "生"]):
            scores["student"] += 1
        elif re.search(r'\d+', suffix):
            scores["student"] += 0.5  # 弱信号
        # 前缀含"能力"/"综合"/"沟通"等→criterion
        prefix = parts[0]
        if any(kw in prefix for kw in ["综合", "沟通", "分析", "人际", "表达", "逻辑", "创新", "协作", "能力"]):
            scores["criterion"] += 1

    # 纯数字列名 → student (不含强信号时)
    if re.match(r'^\d+$', name.strip()) and scores["student"] == 0:
        scores["student"] += 0.3

    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "unknown"


def _extract_num(s: str) -> int:
    nums = re.findall(r'\d+', str(s))
    return int(nums[-1]) if nums else 1


def _guess_excel_facets_auto(df: pd.DataFrame, hr: int) -> int:
    """自动猜测面向数"""
    n_data_cols = df.shape[1] - 2  # 去掉编号列和评分人列
    if n_data_cols <= 4:
        return 3  # 可能是 Student × Rater × Criterion (无 Items)
    elif n_data_cols <= 8:
        return 3
    else:
        return 4


def parse_xlsx_interactive(path: str, n_facets: int,
                           facet_col_map: dict = None) -> dict:
    """
    交互式解析 Excel。

    n_facets: 用户指定的面向数 (2/3/4)
    facet_col_map: 用户指定的列映射, 如 {"col_idx": "student"}

    如果 facet_col_map 为空, 则使用自动检测。
    """
    df = pd.read_excel(path, header=None)
    hr = _find_header_row(df)
    # BUG-008: 合并多行标题
    hr = _merge_header_rows(df, hr)
    cns = [str(df.iloc[hr, j]) if pd.notna(df.iloc[hr, j]) else f"Col{j}"
           for j in range(df.shape[1])]

    # 构建列信息
    for j, cn in enumerate(cns):
        guess = "unknown"
        if j >= 2:  # 跳过编号和评分人列
            guess = _guess_facet_from_name(cn)
        col_info.append({"idx": j, "name": cn, "guess": guess})

    if facet_col_map is None:
        # 自动映射
        facet_col_map = {}
        for ci in col_info:
            if ci["idx"] < 2:
                if ci["idx"] == 0: facet_col_map[ci["idx"]] = "row_id"
                elif ci["idx"] == 1: facet_col_map[ci["idx"]] = "rater"
            else:
                # 用 guess
                facet_col_map[ci["idx"]] = ci["guess"]

    # 提取核心数据列（索引 >= 2）的面向映射
    data_col_maps = {k: v for k, v in facet_col_map.items()
                     if k >= 2 and v not in ("row_id", "rater")}

    # 按面向分组
    students = sorted(set(v for k, v in data_col_maps.items() if v == "student"))
    criteria = sorted(set(v for k, v in data_col_maps.items() if v == "criterion"))
    items = sorted(set(v for k, v in data_col_maps.items() if v == "item"))
    unknowns = [k for k, v in data_col_maps.items() if v == "unknown"]

    n_s = len(students) if students else (len(data_col_maps) if unknowns else 4)
    n_c = len(criteria) if criteria else 2
    n_i = len(items) if items else (1 if n_facets < 4 else 4)

    # 修正自动检测偏差
    if n_s > 8 and not students:
        n_s = 4
    if n_c > 4 and not criteria:
        n_c = 2

    # 根据 n_facets 自动推算缺失的维度
    # 3 面: ns × nr × nc = total_cols_per_row
    #      ns × nc = n_data_cols_per_row
    n_data_cols = df.shape[1] - 2  # 去掉编号和评分人
    if n_facets == 3 and n_data_cols > 0:
        # ns × nc = n_data_cols
        # 已知 ns (from column grouping) 和 n_data_cols → nc = n_data_cols / ns
        calc_nc = n_data_cols / max(n_s, 1)
        if calc_nc == int(calc_nc) and 1 <= calc_nc <= 4:
            n_c = int(calc_nc)
    elif n_facets == 4 and n_data_cols > 0:
        # ns × nc × ni = n_data_cols
        calc = n_data_cols / max(n_s * n_c, 1)
        if calc == int(calc) and 1 <= calc <= 10:
            n_i = int(calc)

    # 构建数据行
    data_rows, rater_labels = [], []
    for ri in range(hr + 1, len(df)):
        row = df.iloc[ri]
        if pd.isna(row.iloc[0]): continue
        r_label = str(row.iloc[1])
        r_num = _extract_num(r_label)
        rater_labels.append(r_label)

        # 简化：按列顺序解析
        # 每一行: Student × Criterion 矩阵（如果有 Items 再加一维）
        col_offset = 2
        if n_facets >= 4:
            # 4 面向: Student × Rater × Criterion × Item
            for si in range(n_s):
                for ci in range(n_c):
                    for ii in range(n_i):
                        col = col_offset + ci * n_s * n_i + si * n_i + ii
                        if col < df.shape[1]:
                            v = row.iloc[col]
                            if pd.notna(v):
                                data_rows.append([si + 1, r_num, ci + 1, ii + 1, int(v)])
        elif n_facets == 3:
            # 3 面向: Student × Rater × Criterion
            for si in range(n_s):
                for ci in range(n_c):
                    col = col_offset + ci * n_s + si
                    if col < df.shape[1]:
                        v = row.iloc[col]
                        if pd.notna(v):
                            data_rows.append([si + 1, r_num, ci + 1, 1, int(v)])
        elif n_facets == 2:
            # 2 面向: Student × Rater
            for si in range(n_s):
                col = col_offset + si
                if col < df.shape[1]:
                    v = row.iloc[col]
                    if pd.notna(v):
                        data_rows.append([si + 1, r_num, 1, 1, int(v)])

    raw = np.array(data_rows) if data_rows else np.array([[1, 1, 1, 1, 1]])
    return {
        "raw": raw, "nf": n_facets, "N": len(raw),
        "ns": n_s, "nr": len(rater_labels) if rater_labels else int(raw[:, 1].max()),
        "nc": n_c, "ni": max(n_i, 1),
        "mn": int(raw[:, -1].min()), "mx": int(raw[:, -1].max()),
        "sl": [f"S{i+1}" for i in range(n_s)],
        "rl": rater_labels if rater_labels else [f"R{i+1}" for i in range(int(raw[:, 1].max()))],
        "cl": [f"C{i+1}" for i in range(n_c)] if n_c > 1 else ["Score"],
        "il": [f"Item{i+1}" for i in range(max(n_i, 1))],
        "model": "", "title": Path(path).stem, "positive": 1,
        "excel_col_info": col_info,
    }


def describe_excel(path: str) -> dict:
    """预读 Excel，返回列信息供用户参考"""
    df = pd.read_excel(path, header=None)
    hr = _find_header_row(df)
    # BUG-008: 合并多行标题
    hr = _merge_header_rows(df, hr)
    cns = [str(df.iloc[hr, j]) if pd.notna(df.iloc[hr, j]) else f"Col{j}"
           for j in range(df.shape[1])]
    n_rows = len(df) - hr - 1
    n_cols = len(cns)
    col_info = []
    for j, cn in enumerate(cns):
        guess = "row_id" if j == 0 else ("rater" if j == 1 else _guess_facet_from_name(cn))
        sample = ""
        if hr + 1 < len(df) and j < df.shape[1]:
            v = df.iloc[hr + 1, j]
            sample = str(v) if pd.notna(v) else "(空)"
        col_info.append({"idx": j, "name": cn, "guess": guess, "sample": sample})
    return {
        "header_row": hr, "n_data_rows": n_rows, "n_cols": n_cols,
        "columns": col_info,
        "n_facets_auto": _guess_excel_facets_auto(df, hr),
    }


# ═══════════════════════════════════════════════════════════════════════
# 3. MFRM 引擎 (不变，只优化细节)
# ═══════════════════════════════════════════════════════════════════════

class Engine:
    """MFRM 参数估计引擎。

    支持模型类型:
      - rating_scale: Andrich Rating Scale 模型 (默认)
      - dichotomous:  标准 Rasch 二分模型 (0/1)
    """

    def __init__(self, d: dict):
        for k, v in d.items():
            setattr(self, k, v)
        self.noncentered = d.get("noncentered", 1)  # 默认约束第1面
        self._idx()

    def _idx(self):
        self.si, self.ri = self.raw[:, 0] - 1, self.raw[:, 1] - 1
        self.ci = np.clip(self.raw[:, 2] - 1, 0, self.nc - 1)
        self.ii = self.raw[:, 3] - 1 if self.nf >= 4 else np.zeros(self.N, dtype=int)
        self.sc = self.raw[:, -1].astype(float)

        # 从 model_info 检测模型类型
        mi = getattr(self, 'model_info', {})
        self._model_type = mi.get('model_type', 'rating_scale') if mi else 'rating_scale'

        if self._model_type == 'dichotomous':
            # 二分模型: 分数范围 0-1
            self.mn = 0
            self.mx = 1
            self.K = 1
            self.x = self.sc.astype(float)
        else:
            # Rating Scale: 使用数据中的分数范围（或 model_info 的）
            self.mn = int(self.raw[:, -1].min())
            self.mx = int(self.raw[:, -1].max())
            # 如果 model_info 提供了分数范围，使用它
            if mi.get('max_score') is not None:
                self.mx = mi['max_score']
                self.mn = mi.get('min_score', 1)
            self.x = self.sc - self.mn
            self.K = int(self.mx - self.mn)

        self.ct = np.arange(self.K + 1, dtype=float)
        self.os = [np.where(self.si == s)[0] for s in range(self.ns)]
        self.or_ = [np.where(self.ri == r)[0] for r in range(self.nr)]
        self.oc = [np.where(self.ci == c)[0] for c in range(self.nc)]
        self.oi = [np.where(self.ii == it)[0] for it in range(max(self.ni, 1))]

    def _pb(self):
        """计算概率矩阵。二分模型和 Rating Scale 模型使用不同公式。"""
        linear = self.th[self.si] - self.dl[self.ri] - self.al[self.ci] - self.bt[self.ii]

        if self._model_type == 'dichotomous':
            # 标准 Rasch 二分模型: P(x=1) = 1 / (1 + exp(-logit))
            p1 = 1.0 / (1.0 + np.exp(-linear))
            return np.column_stack([1.0 - p1, p1])

        # Rating Scale 模型: 累积 logit 参数化
        lg = np.zeros((self.N, self.K + 1))
        cu = np.zeros(self.N)
        for k in range(self.K):
            cu += linear - self.tu[k]
            lg[:, k + 1] = cu
        lg -= lg.max(axis=1, keepdims=True)
        e = np.exp(lg)
        return e / e.sum(axis=1, keepdims=True)

    def _s2l(self, scores_list):
        """Score→logit Newton-Raphson 反演（替代粗糙的 mean/M→logit）。

        Andrich Rating Scale 模型中，E[score|θ,τ] 是 θ 的单调函数。
        给定固定 τ，用 N-R 自洽求解 θ 使期望分等于观测均值。
        二分模型退回简单 logit 公式。
        """
        if self._model_type == 'dichotomous':
            result = []
            for sc in scores_list:
                if len(sc) == 0:
                    result.append(0.0)
                else:
                    p = np.clip(sc.mean(), 0.02, 0.98)
                    result.append(np.log(p / (1.0 - p)))
            return np.array(result)

        n = len(scores_list)
        th = np.zeros(n)
        tu = self.tu
        K = self.K

        for i in range(n):
            sc = scores_list[i]
            nobs = len(sc)
            if nobs == 0:
                th[i] = 0.0
                continue

            target = sc.mean() - self.mn  # 0-based

            # 边界截断：极端分数直接给边界值，避免 N-R 跑飞
            if target <= 0.05 * K:
                th[i] = -5.0
                continue
            if target >= 0.95 * K:
                th[i] = 5.0
                continue

            t = 0.0
            for _ in range(25):
                lin = t - tu               # (K,)
                cum = np.zeros(K + 1)
                for k in range(K):
                    cum[k + 1] = cum[k] + lin[k]
                cum -= cum.max()
                e = np.exp(cum)
                p = e / e.sum()
                exp_s = p @ self.ct
                exp_sq = p @ (self.ct ** 2)
                var = max(exp_sq - exp_s ** 2, 0.001)
                delta = (target - exp_s) / var
                t += delta
                if abs(delta) < 1e-6:
                    break

            th[i] = np.clip(t, -10, 10)

        return th

    def _px(self):
        """PROX 初始化 — 两阶段 N-R 反演（Facets 风格）。

        阶段1: 全局分布估计初始 τ → N-R 反解各元素参数
        阶段2: 用新参数 refine τ → 再跑 N-R（解决先有鸡还是先有蛋问题）
        """
        # ── 初始 τ（从全局分数分布）──
        if self._model_type == 'dichotomous':
            self.tu = np.array([0.0])
        else:
            self.tu = np.array([-np.log(
                np.clip((self.sc >= self.mn + k + 1).mean(), 0.02, 0.98) /
                np.clip((self.sc < self.mn + k + 1).mean(), 0.02, 0.98))
                for k in range(self.K)])
            self.tu -= self.tu[0]

        # ── 阶段1: N-R 反解各面参数（基于初始 τ）──
        self.th = self._s2l([self.sc[self.os[s]] for s in range(self.ns)])
        self.dl = -self._s2l([self.sc[self.or_[r]] for r in range(self.nr)])
        self.al = -self._s2l([self.sc[self.oc[c]] for c in range(self.nc)])
        if self.ni > 1:
            self.bt = -self._s2l([self.sc[self.oi[it]] for it in range(self.ni)])
        else:
            self.bt = np.zeros(1)
        self._center_prox()

        # ── 阶段2: refine τ 并重新 N-R ──
        if self._model_type != 'dichotomous' and self.K > 0:
            p = self._pb()
            # 用当前参数估计重新估计 τ（面向残差）
            for k in range(self.K):
                o = (self.x >= k + 1).sum()
                e = p[:, k + 1:].sum()
                pg = p[:, k + 1:].sum(axis=1)
                info = (pg * (1 - pg)).sum() + 1.0
                self.tu[k] += 0.5 * (e - o) / info
            self.tu -= self.tu[0]

            # 重新 N-R
            self.th = self._s2l([self.sc[self.os[s]] for s in range(self.ns)])
            self.dl = -self._s2l([self.sc[self.or_[r]] for r in range(self.nr)])
            self.al = -self._s2l([self.sc[self.oc[c]] for c in range(self.nc)])
            if self.ni > 1:
                self.bt = -self._s2l([self.sc[self.oi[it]] for it in range(self.ni)])
            self._center_prox()

        # ── U 调整（温和校正量尺）──
        p = self._pb()
        se = self.mn + p @ self.ct
        sd_o, sd_e = np.std(self.sc), np.std(se)
        U = np.clip(1.0 + 0.5 * (sd_o / sd_e - 1.0) if sd_e > 0.01 else 1.0, 0.8, 2.0)
        self.th *= U; self.dl *= U; self.al *= U; self.bt *= U
        if self._model_type != 'dichotomous':
            self.tu *= U

    def _center_prox(self):
        """仅对 noncentered 指定的面居中 (Rasch 识别约束: 仅需1个sum-to-zero)"""
        if self.noncentered != 1:
            self.th -= self.th.mean()
        if self.noncentered != 2:
            self.dl -= self.dl.mean()
        if self.noncentered != 3:
            self.al -= self.al.mean()
        if self.noncentered != 4 and self.ni > 1:
            self.bt -= self.bt.mean()

    def fit(self, p1: int = 200, p2: int = 300):
        """两阶段 JMLE: 高阻尼+强ridge → 低阻尼+弱ridge。二分模型自动跳过 tau 更新。"""
        # BUG-004: 稀疏数据检测与警告
        if self._model_type != 'dichotomous' and self.K > 0:
            per_cat = np.bincount(self.x.astype(int), minlength=self.K + 1)
            sparse_cats = [str(self.mn + i) for i, c in enumerate(per_cat) if c < 8]
            if sparse_cats:
                import warnings
                warnings.warn(
                    f"⚠️ 稀疏数据警告 (每个等级建议 >= 8 观测): "
                    f"分数 {','.join(sparse_cats)} 观测不足。参数可能不稳定，建议合并评分等级。"
                )
        self._px()
        best = (-1e100, None)
        is_dich = self._model_type == 'dichotomous'

        for ph, (ni, rs, rd, dp) in enumerate([(p1, 0.3, 0.99, 0.2), (p2, 0.03, 0.9995, 0.08)]):
            for it in range(1, ni + 1):
                rid = max(rs * (rd ** it), 0.001)
                self._ut(dp, rid); self._ud(dp, rid)
                self._ua(dp, rid); self._ub(dp, rid)
                if not is_dich:
                    self._uu(dp * 0.3, rid * 5)
                if it % 10 == 0:
                    p = self._pb()
                    ll = sum(np.log(max(p[i, int(self.x[i])], 1e-300)) for i in range(self.N))
                    cur = (ll, (self.th.copy(), self.dl.copy(),
                               self.al.copy(), self.bt.copy(),
                               self.tu.copy() if not is_dich else self.tu))
                    if ll > best[0]:
                        best = cur
        if best[1]:
            params = best[1]
            self.th, self.dl, self.al, self.bt = params[:4]
            if is_dich:
                self.tu = np.array([0.0])
            else:
                self.tu = params[4]
        self._fn()
        return self

    def _up(self, a, o, n, d, r, s, center=False):
        """Fisher scoring 更新 — sequential update（BUG-013 修复）。

        旧版: 循环外一次 _pb() → 批量更新所有元素（batch mode）
        新版: 循环内每更新一个元素后立即 _pb()（sequential/JMLE mode）

        数学: JMLE 等价于对每个参数做 Newton step，
              需要"其他参数固定"的假设才成立。
              Sequential 保持了该假设。
        """
        for i in range(n):
            p = self._pb(); e = p @ self.ct
            v = np.clip((p @ (self.ct ** 2)) - e ** 2, 0.001, None)
            idx = o[i]
            a[i] += d * (s * self.x[idx].sum() - s * e[idx].sum()) / (v[idx].sum() + r)
        if center:
            a -= a.mean()
        a[:] = np.clip(a, -20, 20)

    def _ut(self, d, r): self._up(self.th, self.os, self.ns, d, r, 1, center=(self.noncentered != 1))
    def _ud(self, d, r): self._up(self.dl, self.or_, self.nr, d, r, -1, center=(self.noncentered != 2))
    def _ua(self, d, r): self._up(self.al, self.oc, self.nc, d, r, -1, center=(self.noncentered != 3))
    def _ub(self, d, r):
        if self.ni > 1: self._up(self.bt, self.oi, self.ni, d, r, -1, center=(self.nf >= 4 and self.noncentered != 4))

    def _uu(self, d, r):
        """Tau 更新 — sequential update（BUG-013 修复）。

        每个 τ_k 更新后立即重算概率，
        因为 τ_k 的变化影响所有 k+1...K 的累积概率。
        """
        if self._model_type == 'dichotomous' or self.K < 1:
            return
        for k in range(self.K):
            p = self._pb()
            o = (self.x >= k + 1).sum()
            e = p[:, k + 1:].sum()
            pg = p[:, k + 1:].sum(axis=1)
            info = (pg * (1 - pg)).sum() + r
            self.tu[k] += d * (e - o) / info
        self.tu -= self.tu[0]; self.tu[:] = np.clip(self.tu, -15, 25)

    def _fn(self):
        p = self._pb()
        e = p @ self.ct; self.vo = np.clip((p @ (self.ct ** 2)) - e ** 2, 0.001, None)
        self.es = self.mn + e; self.rs = self.sc - self.es
        self.zs = self.rs / np.sqrt(self.vo)
        self.ve = max(0, (np.var(self.sc, ddof=1) - np.var(self.rs, ddof=1))
                      / np.var(self.sc, ddof=1) * 100)

    def _fc(self, o, pm, lb, n):
        rs, ms, ss = [], [], []
        for p in range(n):
            idx = o[p]
            if len(idx) == 0: continue
            t, nm = self.sc[idx].sum(), len(idx)
            zz, vv = self.zs[idx], self.vo[idx]; w = vv.sum()
            inf = (zz ** 2 * vv).sum() / max(w, 1e-10)
            otf = (zz ** 2).sum() / max(nm, 1)
            se = 1.0 / np.sqrt(max(vv.sum(), 1e-10))
            rs.append({"l": lb[p], "t": int(t),
                       "oa": round(self.sc[idx].mean(), 2),
                       "fa": round(self.es[idx].mean(), 2),
                       "m": round(pm[p], 3), "se": round(se, 3),
                       "inf": round(inf, 3), "otf": round(otf, 3)})
            ms.append(pm[p]); ss.append(se)
        ma, sa = np.array(ms), np.array(ss)
        vo = np.var(ma, ddof=1) if len(ma) > 1 else 0.001
        me = np.mean(sa ** 2); vt = max(vo - me, 0.001)
        return rs, np.sqrt(vt / me) if me > 0 else 0, vt / (vt + me)

    def report(self):
        if not hasattr(self, 'es'): raise RuntimeError("请先运行 fit()")
        r = {"summary": {
            "N": self.N, "ns": self.ns, "nr": self.nr, "nc": self.nc, "ni": self.ni,
            "nf": self.nf, "sc": f"{self.mn}-{self.mx}",
            "om": round(float(self.sc.mean()), 3),
            "em": round(float(self.es.mean()), 3),
            "rs": round(float(self.rs.std(ddof=1)), 4),
            "zs": round(float(self.zs.std(ddof=1)), 4),
            "ve": round(self.ve, 2),
        }, "facets": {}}
        for n, o, pm, lb, nn in [("students", self.os, self.th, self.sl[:self.ns], self.ns),
                                  ("raters", self.or_, self.dl, self.rl[:self.nr], self.nr),
                                  ("criteria", self.oc, self.al, self.cl[:self.nc], self.nc)]:
            rows, sep, rel = self._fc(o, pm, lb, nn)
            r["facets"][n] = {"rows": rows, "sep": round(sep, 2), "rel": round(rel, 3)}
        if self.ni > 1:
            rows, sep, rel = self._fc(self.oi, self.bt, self.il[:self.ni], self.ni)
            r["facets"]["items"] = {"rows": rows, "sep": round(sep, 2), "rel": round(rel, 3)}
        return r


# ═══════════════════════════════════════════════════════════════════════
# 4. CLI 交互式入口 (询问面向数)
# ═══════════════════════════════════════════════════════════════════════

def cli_interactive():
    """CLI 交互模式 — 询问用户面向数和列映射"""
    path = input("输入数据文件路径: ").strip().strip('"')
    p = Path(path)
    if not p.exists():
        print(f"文件不存在: {p}")
        return

    if p.suffix == ".txt":
        d = parse_minifac_txt(str(p))
        print(f"\n已从 .txt 文件中读取:")
        print(f"  标题: {d.get('title', '(无)')}")
        print(f"  面向: {d['nf']} 面 ({d['ns']}学生 × {d['nr']}评分者 × {d['nc']}标准"
              + (f" × {d['ni']}题目" if d['ni'] > 1 else "") + ")")
        print(f"  模型: {d.get('model', '(无)')}")
        print(f"  反应数: {d['N']} 条")
        eng = Engine(d).fit()
        eng._print_report()
        return

    # Excel: 交互式询问
    info = describe_excel(str(p))
    print(f"\n=== 数据预览 ===")
    print(f"行数: {info['n_data_rows']} | 列数: {info['n_cols']}")
    print(f"\n列信息:")
    for ci in info["columns"]:
        print(f"  列{ci['idx']}: {ci['name']:<20} "
              f"(猜测: {ci['guess']:<12}) 样例: {ci['sample']}")

    print(f"\n自动检测: {info['n_facets_auto']} 面设计")

    answer = input(f"\n这是几面数据? (2/3/4, 回车=自动 {info['n_facets_auto']}): ").strip()
    nf = int(answer) if answer.isdigit() else info["n_facets_auto"]

    print(f"使用 {nf} 面设计")
    d = parse_xlsx_interactive(str(p), nf)
    eng = Engine(d).fit()
    eng._print_report()


def cli_direct(path: str) -> Engine:
    """CLI 直接模式 — 已知文件类型，直接分析"""
    p = Path(path)
    if p.suffix == ".txt":
        d = parse_minifac_txt(str(p))
    else:
        info = describe_excel(str(p))
        nf = info["n_facets_auto"]
        d = parse_xlsx_interactive(str(p), nf)
    eng = Engine(d).fit()
    eng._print_report()
    return eng


# 将 _print_report 挂到 Engine 上
def _print_report(self):
    r = self.report(); s = r["summary"]
    mt = getattr(self, '_model_type', 'rating_scale')
    mt_label = {"rating_scale": "Rating Scale", "dichotomous": "Dichotomous",
                "poisson": "Poisson"}.get(mt, mt)
    print(f"\n{'='*60}\nMFRMSight v0.8.0 — {mt_label}\n{'='*60}")
    print(f"{s['N']}条 | {s['nf']}面 | {s['ns']}S×{s['nr']}R×{s['nc']}C"
          + (f"×{s['ni']}I" if s['ni'] > 1 else "") + f" | {s['sc']}分 | Var={s['ve']}%")
    print(f"ObsMean={s['om']:.2f} ExpMean={s['em']:.2f} ResidSD={s['rs']:.4f} StResSD={s['zs']:.4f}")
    for fn, fd in r["facets"].items():
        if not fd["rows"]: continue
        print(f"\n{'─'*60}\n{fn} — Sep={fd['sep']:.2f} Rel={fd['rel']:.3f}\n{'─'*60}")
        print(f"{'':<16}{'Total':>7}{'ObsAvg':>7}{'FairAvg':>7}{'Meas':>7}{'SE':>6}{'Infit':>6}{'Outfit':>6}")
        print("-" * 60)
        for x in fd["rows"]:
            print(f"{x['l']:<16}{x['t']:>7}{x['oa']:>7}{x['fa']:>7}{x['m']:>7}{x['se']:>6}{x['inf']:>6}{x['otf']:>6}")

Engine._print_report = _print_report


# ═══════════════════════════════════════════════════════════════════════
# 5. GUI (Gradio, 带交互式询问)
# ═══════════════════════════════════════════════════════════════════════

def gui():
    import gradio as gr

    def pre_read_excel(file):
        """预读 Excel，返回列信息供用户选择"""
        if file is None:
            return "请先上传文件", gr.update(visible=False), gr.update(choices=[])
        sf = Path(file.name).suffix
        if sf == ".txt":
            return ("✅ .txt 文件 — 将自动识别面向数\n(无需手动选择)", gr.update(visible=False), gr.update(choices=[]))

        path = file.name if hasattr(file, 'name') else file
        info = describe_excel(path)
        lines = [
            f"### Excel 数据预览",
            f"数据行数: {info['n_data_rows']} | 列数: {info['n_cols']}",
            f"自动检测: **{info['n_facets_auto']} 面设计**",
            "",
            "| 列索引 | 列名 | 猜测类型 | 样例 |",
            "|--------|------|---------|------|",
        ]
        choices = []
        for ci in info["columns"]:
            lines.append(f"| {ci['idx']} | {ci['name']} | {ci['guess']} | {ci['sample']} |")
            if ci['idx'] >= 2:  # 只让用户选择数据列
                choices.append(f"列{ci['idx']}: {ci['name']} ({ci['guess']})")
        return ("\n".join(lines), gr.update(visible=True, value=info['n_facets_auto']),
                gr.update(choices=choices, value=[c for c in choices if "student" in c.lower()]))

    def run_analysis(file, n_facets, selected_cols):
        """执行分析"""
        if file is None:
            return "请先上传文件", None, None, None, None
        sf = Path(file.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=sf) as f:
            f.write(open(file.name, "rb").read() if not isinstance(file, bytes) else file)
            tmp = f.name
        try:
            if sf == ".txt":
                d = parse_minifac_txt(tmp)
            else:
                d = parse_xlsx_interactive(tmp, int(n_facets))
            e = Engine(d).fit()
        finally:
            os.unlink(tmp)
        r = e.report(); s = r["summary"]
        parse_info = f"({s['nf']}面: {s['ns']}学生×{s['nr']}评分者×{s['nc']}标准"
        if s['ni'] > 1: parse_info += f"×{s['ni']}题目"
        parse_info += ")"
        txt = (f"### 分析结果 {parse_info}\n"
               f"{s['N']}条 | {s['sc']}分 | 方差解释: **{s['ve']}%**\n"
               f"ObsMean={s['om']:.2f} | ExpMean={s['em']:.2f} | "
               f"ResidSD={s['rs']:.4f} | StResSD={s['zs']:.4f}")
        dfs = []
        for key in ["students", "raters", "criteria", "items"]:
            fd = r["facets"].get(key)
            if fd and fd["rows"]:
                df = pd.DataFrame(fd["rows"])[["l", "t", "oa", "m", "se", "inf", "otf"]]
                df.columns = ["名称", "总分", "ObsAvg", "Meas", "SE", "Infit", "Outfit"]
                df["总分"] = df["总分"].astype(int)
                dfs.append(df)
            else:
                dfs.append(None)
        return (txt,) + tuple(dfs[i] if i < len(dfs) else None for i in range(4))

    with gr.Blocks(title="MFRMSight v0.8.0") as app:
        gr.Markdown("# 📊 MFRMSight v0.8.0 — 多面Rasch模型分析")
        gr.Markdown("**上传数据 → 确认面向数 → 自动分析**")

        fi = gr.File(label="📂 上传数据文件", file_types=[".txt", ".xlsx", ".xls"])
        preview = gr.Markdown()

        with gr.Row(visible=False) as facet_row:
            nf_input = gr.Number(label="面向数 (Facets)", value=3, precision=0, minimum=2, maximum=4)
            col_select = gr.CheckboxGroup(label="Student 对应的列 (可多选)", choices=[])

        status = gr.Markdown()
        btn = gr.Button("🚀 开始分析", variant="primary")

        with gr.Row():
            with gr.Column(): t1 = gr.DataFrame(label="🎓 学生")
            with gr.Column(): t2 = gr.DataFrame(label="👤 评分者")
        with gr.Row():
            with gr.Column(): t3 = gr.DataFrame(label="📋 标准")
            with gr.Column(): t4 = gr.DataFrame(label="📝 题目")

        fi.change(fn=pre_read_excel, inputs=[fi],
                  outputs=[preview, facet_row, col_select])

        btn.click(fn=run_analysis, inputs=[fi, nf_input, col_select],
                  outputs=[status, t1, t2, t3, t4])

        gr.Markdown("MFRMSight v0.8.0 · Minifac 完全适配 · 交互式面向配置")

    import socket, signal, subprocess, platform, time

    # BUG-001 修复: 自动清理旧进程 + 端口重试
    def _find_and_kill_old(port: int) -> bool:
        """在 Windows/Linux 上查找占用指定端口的进程并终止。"""
        killed = False
        if platform.system() == "Windows":
            try:
                result = subprocess.run(
                    ["netstat", "-ano"], capture_output=True, text=True, timeout=10
                )
                for line in result.stdout.splitlines():
                    if f":{port}" in line and "LISTENING" in line:
                        parts = line.split()
                        pid = parts[-1]
                        if pid.isdigit():
                            print(f"[!] 端口 {port} 被 PID={pid} 占用，正在终止...")
                            subprocess.run(["taskkill", "/F", "/PID", pid],
                                           capture_output=True)
                            killed = True
                            time.sleep(0.5)
            except Exception as e:
                print(f"[!] 端口检查失败: {e}")
        else:
            try:
                result = subprocess.run(
                    ["lsof", "-ti", f":{port}"], capture_output=True, text=True
                )
                pids = result.stdout.strip().splitlines()
                for pid in pids:
                    if pid.isdigit():
                        print(f"[!] 端口 {port} 被 PID={pid} 占用，正在终止...")
                        os.kill(int(pid), signal.SIGTERM)
                        killed = True
                        time.sleep(0.3)
            except Exception as e:
                print(f"[!] 端口检查失败: {e}")
        return killed

    # 自动选择可用端口
    max_retries = 10
    for attempt in range(max_retries):
        try:
            port = 7870 + attempt
            _find_and_kill_old(port)
            print(f"🚀 启动 Gradio 服务: http://localhost:{port}")
            app.launch(server_name="0.0.0.0", server_port=port, share=False)
            break
        except OSError as e:
            if "10048" in str(e) or "address already in use" in str(e).lower():
                print(f"[!] 端口 {port} 仍被占用，尝试下一个...")
                continue
            raise
    else:
        print("[✗] 无法绑定端口 (7870-7879 均被占用)，请手动检查")


# ═══════════════════════════════════════════════════════════════════════
# 6. 入口
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("-w", "--web"):
        print("启动 Gradio Web 界面...")
        gui()
    elif len(sys.argv) > 1 and sys.argv[1] in ("-i", "--interactive"):
        cli_interactive()
    elif len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help"):
        print("MFRMSight v0.8.0 — 多面Rasch模型分析工具")
        print()
        print("用法:")
        print("  python mfrm_app.py data.txt       # Minifac .txt 直接分析")
        print("  python mfrm_app.py data.xlsx      # Excel 直接分析 (自动检测)")
        print("  python mfrm_app.py -i             # 交互模式 (手动指定面向数)")
        print("  python mfrm_app.py -w             # Gradio Web 界面 (推荐)")
        print()
        print("Minifac .txt 格式: Facets=N, Model=?,?,?,?,R23, Labels=..., Data=...")
    elif len(sys.argv) > 1:
        cli_direct(sys.argv[1])
    else:
        print("MFRMSight v0.8.0")
        print("用法: python mfrm_app.py <文件> | -w | -i | -h")
