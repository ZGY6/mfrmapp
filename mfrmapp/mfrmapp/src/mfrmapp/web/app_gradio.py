"""
MFRMSight Web — Gradio 版 v1.0.12
渐进四步: 上传选面 → 分析结果 → 交互分析 → 报告+图表
Gradio 6 兼容: 不用 visible 切换 / 不用 .then() / 所有组件始终可见
图表: 保存PNG→文件路径传gr.Image (不用base64 URI)
"""
import gradio as gr
import pandas as pd
import numpy as np
import tempfile, os, sys, io, base64
from pathlib import Path

_parent = Path(__file__).parent.parent
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))
from engine import (parse_facets_txt, parse_excel, MFRMEngine,
                    extract_dimensions, generate_word_report,
                    chart_ruler_map, chart_category_curves, chart_fit_distribution)

_SESSION = {}


def _gen_interactions(dims: dict) -> list[str]:
    keys = [f["key"] for f in dims["facets"]]
    labels = {f["key"]: f["label"] for f in dims["facets"]}
    return [f"{labels[keys[i]]} x {labels[keys[j]]}"
            for i in range(len(keys)) for j in range(i+1, len(keys))]


# ═══════ Step 1: Upload + Facet Selection ═══════

def step1_upload(file):
    if file is None:
        return "👆 请先上传文件", gr.update(choices=[], value=[])
    suffix = Path(file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(file.read() if hasattr(file, 'read') else open(file.name, "rb").read())
        tmp = f.name
    try:
        data = parse_facets_txt(tmp) if suffix == ".txt" else parse_excel(tmp)
    finally:
        os.unlink(tmp)
    _SESSION["data"] = data; _SESSION["engine"] = None; _SESSION["bias_results"] = []
    dims = extract_dimensions(data)
    _SESSION["dims"] = dims; _SESSION["all_interactions"] = _gen_interactions(dims)
    choices = [f"{fi['label']} ({fi['key']})" for fi in dims["facets"]]
    lines = [f"### ✅ 检测到 {dims['n_facets']} 面\n勾选要分析的面 (最少 Students + Raters):"]
    for fi in dims["facets"]:
        elems = fi["elements"][:fi["n"]]
        zh = fi["label"]
        orig = f" ({fi['original']})" if fi['original'] != fi['key'] else ""
        lines.append(f"- **{zh}**{orig}: {fi['n']} 个元素 — {', '.join(elems[:8])}")
    return "\n".join(lines), gr.update(choices=choices, value=choices)


# ═══════ Step 2: Analysis (16 outputs) ═══════

def _none16():
    return (gr.update(),) * 16

def step2_analyze(selected_facets):
    data = _SESSION.get("data")
    if data is None:
        return ("### ❌ 请先上传文件", *_none16()[1:])
    if not selected_facets:
        return ("### ❌ 请至少选择一个面", *_none16()[1:])

    selected_keys = set()
    for item in selected_facets:
        selected_keys.add(item.split("(")[-1].rstrip(")"))
    if not {"students", "raters"}.issubset(selected_keys):
        return ("### ❌ 必须包含 Students + Raters", *_none16()[1:])

    raw = data["raw"].copy()
    nf_orig = data.get("n_facets", data.get("nf", 4))
    has_c = "criteria" in selected_keys; has_i = "items" in selected_keys
    nf_new = 4 if has_i else (3 if has_c else 2)

    # 去掉 Items 时需要聚合: 同一 (student,rater,criterion) 下的多个 Item 取均值
    if nf_orig >= 4 and not has_i:
        # 按 (student, rater, criterion) 分组，score 取平均
        df_raw = pd.DataFrame(raw, columns=["s", "r", "c", "i", "score"])
        grouped = df_raw.groupby(["s", "r", "c"])["score"].mean().round().astype(int).reset_index()
        raw = grouped[["s", "r", "c", "score"]].values
    # 去掉 Criteria 时需要聚合: 同一 (student,rater) 下的多个 Criteria 取均值
    elif nf_orig >= 3 and not has_c and has_i:
        df_raw = pd.DataFrame(raw, columns=["s", "r", "c", "i", "score"])
        grouped = df_raw.groupby(["s", "r", "i"])["score"].mean().round().astype(int).reset_index()
        raw = grouped[["s", "r", "i", "score"]].values

    cols = [raw[:, 0], raw[:, 1]]
    if has_c: cols.append(raw[:, 2])
    if has_i: cols.append(raw[:, 3])
    cols.append(raw[:, -1])
    data_tmp = dict(data)
    data_tmp["raw"] = np.column_stack(cols)
    data_tmp["N"] = len(data_tmp["raw"]); data_tmp["n_facets"] = nf_new
    if not has_i: data_tmp["n_i"] = 1
    if not has_c: data_tmp["n_c"] = 1

    try:
        eng = MFRMEngine(data_tmp).fit(p1=150, p2=80)
    except Exception as e:
        import traceback
        return (f"### ❌ 分析失败: {e}\n```\n{traceback.format_exc()[:300]}\n```", *_none16()[1:])

    _SESSION["engine"] = eng
    r = eng.report(); s = r["summary"]

    bias_count = len(r.get("bias", []))
    bias_text = ""
    if bias_count > 0:
        tags = set()
        for b in r["bias"]:
            for t, _ in b["flags"]: tags.add(t)
        bias_text = f"\n\n> ⚠️ 检出 {bias_count} 位偏差: {', '.join(tags)}"

    summary = (
        f"### ✅ Step 2 完成 — 分析结果\n"
        f"N={s['N']} | {nf_new}面 | {s['score_range']}分 | 方差解释 **{s['var_exp']}%**\n"
        f"ObsMean={s['obs_mean']:.2f} | ExpMean={s['exp_mean']:.2f}" + bias_text
    )

    bias_map = {b["rater"]: [t for t, _ in b["flags"]] for b in r.get("bias", [])}
    emo = {"students": "🎓", "raters": "👤", "criteria": "📋", "items": "📝"}
    tables = {}
    for key in ["students", "raters", "criteria", "items"]:
        fd = r["facets"].get(key)
        if not fd or not fd["rows"]: tables[key] = gr.update(value=None); continue
        df = pd.DataFrame(fd["rows"])
        cols = ["label", "total", "obs_avg", "meas", "se", "infit", "outfit"]
        cn = ["名称", "总分", "ObsAvg", "Meas", "SE", "Infit", "Outfit"]
        if key == "raters" and bias_map:
            df["bias_flag"] = df["label"].apply(lambda x: ", ".join(bias_map.get(x, [])))
            cols.append("bias_flag"); cn.append("偏差")
        df_d = df[cols].copy(); df_d.columns = cn
        if "总分" in df_d.columns: df_d["总分"] = df_d["总分"].astype(int)
        sep = fd.get("separation", 0); rel = fd.get("reliability", 0)
        tables[key] = gr.update(value=df_d,
                                label=f"{emo.get(key,'')} {key} (Sep={sep:.2f} Rel={rel:.3f})")

    rank = r.get("rank_compare", [])
    rank_df = pd.DataFrame(rank) if rank else pd.DataFrame()
    if len(rank_df) > 0:
        rank_df.columns = ["考生", "原始总分", "FairAvg", "Meas", "原始排名", "校正排名", "排名变化"]
    rank_out = gr.update(value=rank_df if len(rank_df) > 0 else None)

    interactions = _SESSION.get("all_interactions", [])

    return (summary,
            tables["students"], tables["raters"], tables["criteria"], tables["items"],
            gr.update(), gr.update(), gr.update(),
            rank_out,
            gr.update(value="### Step 3: 偏差交互分析\n请勾选要分析的交互对"),
            gr.update(choices=interactions, value=[]),
            gr.update(),
            gr.update(value=""), gr.update(value=""),
            gr.update(value=""),
            gr.update())


# ═══════ Step 3: Bias Interaction (6 outputs) ═══════

def step3_bias(selected_interactions):
    eng = _SESSION.get("engine"); dims = _SESSION.get("dims")
    if eng is None:
        return ("### ⏳ 请先完成 Step 2 分析", gr.update(value=None),
                gr.update(), gr.update(), gr.update(), gr.update())
    if not selected_interactions:
        return ("### ⏳ 请勾选至少一个交互对", gr.update(value=None),
                gr.update(), gr.update(), gr.update(), gr.update())

    key_map = {f["label"]: f["key"] for f in dims["facets"]}
    bias_results = []; all_rows = []
    for pair_name in selected_interactions:
        parts = pair_name.split(" x ")
        if len(parts) != 2: continue
        ka = key_map.get(parts[0]); kb = key_map.get(parts[1])
        if not ka or not kb: continue
        rows = eng.bias_interaction(ka, kb)
        bias_results.append({"pair": pair_name, "rows": rows})
        for r in rows: r["pair"] = pair_name
        all_rows.extend(rows)

    _SESSION["bias_results"] = bias_results
    sig_count = sum(1 for r in all_rows if r.get("significant"))
    summary = f"### ✅ Step 3 完成 — 偏差交互分析\n{len(all_rows)} 对交互, **{sig_count} 对显著** (|z| >= 2)"

    df = pd.DataFrame(all_rows) if all_rows else pd.DataFrame()
    if len(df) > 0:
        cols_ok = [c for c in ["pair", "a", "b", "obs_avg", "exp_avg", "bias", "z", "significant"] if c in df.columns]
        df_d = df[cols_ok].copy()
        df_d.columns = ["交互对", "A", "B", "ObsAvg", "ExpAvg", "Bias", "z", "显著"][:len(cols_ok)]
        bias_df = gr.update(value=df_d)
    else:
        bias_df = gr.update(value=None)

    return (summary, bias_df,
            gr.update(value="### Step 4: 专业报告 + 统计图\n点击下方按钮生成"),
            gr.update(value=""),
            gr.update(),
            gr.update())


# ═══════ Step 4: Report + Charts (8 outputs) ═══════

def _path_to_numpy(img_path: str) -> "np.ndarray | None":
    """将 PNG 文件路径转为 Gradio Image (type=numpy) 可用的 numpy 数组"""
    if not img_path or not os.path.exists(img_path):
        return None
    try:
        from PIL import Image as PILImage
        return np.array(PILImage.open(img_path))
    except Exception:
        return None


def step4_generate():
    eng = _SESSION.get("engine"); data = _SESSION.get("data")
    bias_results = _SESSION.get("bias_results", [])
    _n8 = (gr.update(),) * 8

    if eng is None or data is None:
        return (gr.update(value="### ❌ 请先完成 Step 2 和 Step 3"),
                gr.update(value=f"eng={eng is not None} data={data is not None}"),
                gr.update(value=""),
                *_n8[3:])

    # 1. Word 报告
    word_path = os.path.join(tempfile.gettempdir(), "mfrm_report.docx")
    try:
        generate_word_report(eng, data, bias_results, word_path)
        word_size = os.path.getsize(word_path)
    except Exception as e:
        import traceback
        return (gr.update(value="### ❌ Word生成失败"),
                gr.update(value=f"错误: {e}"),
                gr.update(value=traceback.format_exc()),
                *_n8[3:])

    # 2. 统计图 → 保存 PNG → 转 numpy 数组 (Gradio 6 gr.Image 默认 type=numpy)
    chart_arrays = []
    for chart_name, chart_fn in [("ruler_map", chart_ruler_map),
                                  ("fit_distribution", chart_fit_distribution),
                                  ("category_curves", chart_category_curves)]:
        try:
            b64 = chart_fn(eng, data) if chart_name == "ruler_map" else chart_fn(eng)
            img_path = os.path.join(tempfile.gettempdir(), f"mfrm_{chart_name}.png")
            with open(img_path, "wb") as f:
                f.write(base64.b64decode(b64))
            chart_arrays.append(_path_to_numpy(img_path))
            os.unlink(img_path)
        except Exception as e:
            chart_arrays.append(None)

    n_ok = sum(1 for c in chart_arrays if c is not None)
    status = f"### ✅ 报告 + 图表生成完毕\n\n📊 图表: {n_ok}/3 张 | 📥 Word: {word_size/1024:.0f} KB"
    log = f"Word: {word_path} ({word_size} bytes) | Charts: {n_ok}/3"

    return (gr.update(value="### Step 4: 统计图 + Word 报告"),
            gr.update(value=status),
            gr.update(value=log),
            gr.update(value=word_path),
            gr.update(),
            gr.update(value=chart_arrays[0]),
            gr.update(value=chart_arrays[1]),
            gr.update(value=chart_arrays[2]))


# ═══════ UI ═══════

def build_interface():
    with gr.Blocks(title="MFRMSight v1.0.12") as app:
        gr.Markdown("# 📊 MFRMSight v1.0.12 — 多面 Rasch 模型分析")

        # Step 1
        gr.Markdown("### Step 1: 上传 + 选面")
        file_input = gr.File(label="上传 .txt 或 .xlsx", file_types=[".txt", ".xlsx", ".xls"])
        dim_preview = gr.Markdown("👆 请先上传文件")
        facet_select = gr.CheckboxGroup(label="勾选要分析的面", choices=[], interactive=True)
        btn1 = gr.Button("🚀 开始分析", variant="primary")

        # Step 2
        step2_header = gr.Markdown("")
        with gr.Row():
            t1 = gr.DataFrame(); t2 = gr.DataFrame()
        with gr.Row():
            t3 = gr.DataFrame(); t4 = gr.DataFrame()
        rank_table = gr.DataFrame(label="📊 排名对比")

        # Placeholders for Step 4 charts
        _ph1 = gr.Textbox(visible=False); _ph2 = gr.Textbox(visible=False); _ph3 = gr.Textbox(visible=False)

        # Step 3
        step3_header = gr.Markdown("### Step 3: 偏差交互分析")
        inter_select = gr.CheckboxGroup(label="勾选交互对", choices=[])
        btn3 = gr.Button("🔍 分析交互项", variant="primary")
        bias_summary = gr.Markdown("")
        bias_table = gr.DataFrame(label="偏差交互分析结果")

        # Step 4
        step4_header = gr.Markdown("### Step 4: 专业报告 + 统计图")
        step4_status = gr.Markdown("")
        report_md = gr.Markdown("")
        btn4 = gr.Button("📄 生成 Word 报告 + 图表", variant="secondary")
        with gr.Row():
            chart_img1 = gr.Image(label="垂直标尺图")
            chart_img2 = gr.Image(label="拟合统计量分布图")
        chart_img3 = gr.Image(label="等级概率曲线与 ICC")
        report_dl = gr.File(label="📥 下载 Word 报告")

        # Events
        S2_OUT = [step2_header, t1, t2, t3, t4, _ph1, _ph2, _ph3, rank_table,
                  step3_header, inter_select, btn3,
                  step4_header, step4_status, report_md, report_dl]

        file_input.change(fn=step1_upload, inputs=[file_input],
                          outputs=[dim_preview, facet_select])

        btn1.click(fn=step2_analyze, inputs=[facet_select], outputs=S2_OUT)

        S3_OUT = [bias_summary, bias_table, step4_header, step4_status, report_dl, btn4]
        btn3.click(fn=step3_bias, inputs=[inter_select], outputs=S3_OUT)

        S4_OUT = [step4_header, step4_status, report_md, report_dl, btn4,
                  chart_img1, chart_img2, chart_img3]
        btn4.click(fn=step4_generate, outputs=S4_OUT)

        gr.Markdown("MFRMSight v1.0.12 · Andrich Rating Scale · Fisher-scoring JMLE")

    return app


def main():
    app = build_interface()
    port = int(os.environ.get("PORT", "7870"))
    print(f"MFRMSight v1.0.12 — port={port}, PID={os.getpid()}")
    app.launch(server_name="0.0.0.0", server_port=port, share=False, theme=gr.themes.Soft())


if __name__ == "__main__":
    main()
