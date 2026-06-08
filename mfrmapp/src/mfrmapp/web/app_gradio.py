"""
MFRMSight Web — Gradio 版 v1.0.0
四步递进: 选面分析 → 结果展示 → 交互分析 → 报告+图表下载
"""
import gradio as gr
import pandas as pd
import numpy as np
import tempfile, os, sys, io, base64, time
from pathlib import Path

_parent = Path(__file__).parent.parent
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))
from engine import (parse_facets_txt, MFRMEngine,
                    extract_dimensions, generate_report, generate_word_report,
                    chart_ruler_map, chart_category_curves, chart_fit_distribution)

_SESSION = {}
_HIDE4 = (gr.update(visible=False),) * 4


def _gen_interaction_choices(dims):
    keys = [f["key"] for f in dims["facets"]]
    labels = {f["key"]: f["label"] for f in dims["facets"]}
    return [f"{labels[keys[i]]} x {labels[keys[j]]}"
            for i in range(len(keys)) for j in range(i+1, len(keys))]


# ════════════════════════════════════════════
# Step 1: 上传 + 选面
# ════════════════════════════════════════════

def step1_upload(file):
    if file is None:
        return "👆 请先上传文件", gr.update(choices=[], value=[])
    suffix = Path(file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(file.read() if hasattr(file, 'read') else open(file.name, "rb").read())
        tmp = f.name
    try:
        if suffix == ".txt": data = parse_facets_txt(tmp)
        else:
            from engine import parse_excel
            data = parse_excel(tmp)
    finally: os.unlink(tmp)
    _SESSION["data"] = data; _SESSION["engine"] = None; _SESSION["bias_results"] = []
    dims = extract_dimensions(data)
    _SESSION["dims"] = dims; _SESSION["all_interactions"] = _gen_interaction_choices(dims)
    choices = [f"{fi['label']} ({fi['key']})" for fi in dims["facets"]]
    lines = [f"### ✅ 检测到 {dims['n_facets']} 面\n勾选要分析的面:"]
    for fi in dims["facets"]:
        elems = fi["elements"][:fi["n"]]
        lines.append(f"- **{fi['label']}**: {', '.join(elems[:8])}")
    return "\n".join(lines), gr.update(choices=choices, value=choices)


# ════════════════════════════════════════════
# Step 2: 分析
# ════════════════════════════════════════════

def _err17(msg):
    # 16 outputs: msg + 4 tables + 3 charts + rank + step3_hdr + inter_select + btn3 + step4_hdr + status + report_md + report_dl
    return (msg, *[None]*4, None, None, None, None,  # 1+4+3+1=9
            gr.update(visible=False), gr.update(choices=[], value=[]), gr.update(visible=False),  # +3=12
            gr.update(visible=False), gr.update(visible=False, value=""), gr.update(visible=False, value=""), gr.update(visible=False))  # +4=16

def step2_analyze(selected_facets):
    data = _SESSION.get("data")
    if data is None: return _err17("### ❌ 请先上传文件")
    if not selected_facets: return _err17("### ❌ 请至少选择一个面")
    selected_keys = set()
    for item in selected_facets: selected_keys.add(item.split("(")[-1].rstrip(")"))
    if not {"students", "raters"}.issubset(selected_keys):
        return _err17("### ❌ 必须包含 Students + Raters")

    raw = data["raw"].copy()
    has_c = "criteria" in selected_keys; has_i = "items" in selected_keys
    nf_new = 4 if has_i else (3 if has_c else 2)
    cols = [raw[:, 0], raw[:, 1]]
    if has_c: cols.append(raw[:, 2])
    if has_i: cols.append(raw[:, 3])
    cols.append(raw[:, -1])
    data_tmp = dict(data); data_tmp["raw"] = np.column_stack(cols)
    data_tmp["N"] = len(data_tmp["raw"]); data_tmp["n_facets"] = nf_new
    if not has_i: data_tmp["n_i"] = 1
    if not has_c: data_tmp["n_c"] = 1

    try: eng = MFRMEngine(data_tmp).fit()
    except Exception as e: return _err17(f"### ❌ 分析失败: {e}")
    _SESSION["engine"] = eng
    r = eng.report(); s = r["summary"]

    # 偏差摘要
    bias_count = len(r.get("bias", []))
    bias_text = ""
    if bias_count > 0:
        tags = set()
        for b in r["bias"]:
            for t, _ in b["flags"]: tags.add(t)
        bias_text = f"\n\n> ⚠️ 检出 {bias_count} 位评分者偏差: {', '.join(tags)}"

    summary_text = (
        f"### ✅ Step 2 完成 — 分析结果\n"
        f"N={s['N']} | {nf_new}面 | {s['score_range']}分 | 方差解释 **{s['var_exp']}%**\n"
        f"ObsMean={s['obs_mean']:.2f} | ExpMean={s['exp_mean']:.2f}" + bias_text
    )

    bias_map = {b["rater"]: [t for t, _ in b["flags"]] for b in r.get("bias", [])}
    emo = {"students": "🎓", "raters": "👤", "criteria": "📋", "items": "📝"}
    tables = {}
    for key in ["students", "raters", "criteria", "items"]:
        fd = r["facets"].get(key)
        if not fd or not fd["rows"]: tables[key] = gr.update(visible=False, value=None); continue
        df = pd.DataFrame(fd["rows"])
        cols = ["label", "total", "obs_avg", "meas", "se", "infit", "outfit"]
        cn = ["名称", "总分", "ObsAvg", "Meas", "SE", "Infit", "Outfit"]
        if key == "raters" and bias_map:
            df["bias_flag"] = df["label"].apply(lambda x: ", ".join(bias_map.get(x, [])))
            cols.append("bias_flag"); cn.append("偏差")
        df_d = df[cols].copy(); df_d.columns = cn
        if "总分" in df_d.columns: df_d["总分"] = df_d["总分"].astype(int)
        sep = fd.get("separation", 0); rel = fd.get("reliability", 0)
        tables[key] = gr.update(value=df_d, visible=True,
                                label=f"{emo.get(key,'')} {key} (Sep={sep:.2f} Rel={rel:.3f})")

    rank = r.get("rank_compare", [])
    rank_df = pd.DataFrame(rank) if rank else None
    if rank_df is not None and len(rank_df) > 0:
        rank_df.columns = ["考生", "原始总分", "FairAvg", "Meas", "原始排名", "校正排名", "排名变化"]
        rank_out = gr.update(value=rank_df, visible=True)
    else: rank_out = gr.update(visible=False, value=None)

    interactions = _SESSION.get("all_interactions", [])
    # 17 outputs: msg, 4 tables, 3 charts(empty), rank, step3_hdr, inter_select*2, btn3*2, report_md+status*3
    # Actually: msg + t1,t2,t3,t4 + chart_ruler,fit,cat(empty) + rank + step3_header,inter_msg,inter_select,btn3
    #          + step4_header,step4_status,report_md,report_dl
    return (summary_text,
            tables["students"], tables["raters"], tables["criteria"], tables["items"],
            None, None, None,            # charts (empty, generated in step 4)
            rank_out,
            gr.update(visible=True, value="### Step 3: 偏差交互分析\n请勾选要分析的交互对"),  # step3_header
            gr.update(visible=True, choices=interactions, value=[]),  # inter_select
            gr.update(visible=True),      # btn3
            gr.update(visible=False),     # step4_header
            gr.update(visible=False, value=""),  # step4_status
            gr.update(visible=False, value=""),  # report_md
            gr.update(visible=False))     # report_dl


# ════════════════════════════════════════════
# Step 3: 偏差交互
# ════════════════════════════════════════════

def step3_bias(selected_interactions):
    eng = _SESSION.get("engine"); dims = _SESSION.get("dims")
    if eng is None or not selected_interactions:
        return ("### ⏳ 等待选择交互对...", None,
                gr.update(visible=False), gr.update(value="", visible=False),
                gr.update(visible=False), gr.update(visible=False))

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
    sig = [r for r in all_rows if r.get("significant")]
    summary = f"### ✅ Step 3 完成 — 偏差交互分析\n{len(all_rows)} 对交互, **{len(sig)} 对显著** (|z| >= 2)"

    if sig:
        df = pd.DataFrame(sig[:30])
        cols_ok = [c for c in ["pair", "a", "b", "obs_avg", "exp_avg", "bias", "z"] if c in df.columns]
        df_d = df[cols_ok].copy()
        df_d.columns = ["交互对", "A", "B", "ObsAvg", "ExpAvg", "Bias", "z"]
        bias_df = gr.update(value=df_d, visible=True)
    else:
        bias_df = gr.update(value=None, visible=True)

    return (summary, bias_df,
            gr.update(visible=True, value="### Step 4: 生成专业报告 + 统计图\n点击下方按钮生成报告和图表"),  # step4_header
            gr.update(visible=True, value=""),   # step4_status
            gr.update(visible=True),             # report_dl
            gr.update(visible=True))             # generate_btn


# ════════════════════════════════════════════
# Step 4: 图表展示 + Word 报告下载
# ════════════════════════════════════════════

def step4_generate():
    """生成 Word 报告 + 3 张 PNG 统计图，显示在网页中并提供下载"""
    eng = _SESSION.get("engine"); data = _SESSION.get("data")
    bias_results = _SESSION.get("bias_results", [])
    if eng is None:
        return ("### ❌ 请先完成前面的步骤", None, None, None, None, None, None)

    # 1. 生成 Word 报告
    word_path = os.path.join(tempfile.gettempdir(), "mfrm_report.docx")
    try:
        generate_word_report(eng, data, bias_results, word_path)
    except Exception as e:
        return (f"### ❌ Word 报告生成失败: {e}", None, None, None, None, None, None)

    # 2. 生成 3 张统计图 PNG
    chart_paths = []
    chart_images = []  # PIL Image objects for gr.Image
    for chart_name, chart_fn in [("ruler_map", chart_ruler_map),
                                  ("fit_distribution", chart_fit_distribution),
                                  ("category_curves", chart_category_curves)]:
        try:
            b64 = chart_fn(eng, data) if chart_name == "ruler_map" else chart_fn(eng)
            img_data = base64.b64decode(b64)
            img_path = os.path.join(tempfile.gettempdir(), f"mfrm_{chart_name}.png")
            with open(img_path, "wb") as f:
                f.write(img_data)
            chart_paths.append(img_path)
            chart_images.append(img_path)
        except Exception:
            chart_paths.append(None)
            chart_images.append(None)

    status = ("### ✅ 报告 + 图表生成完毕\n\n"
              f"📥 点击下方按钮下载 **Word 报告**")

    # outputs: header, status, report_md(hidden), word_dl, btn4, chart1, chart2, chart3
    return (gr.update(value="### Step 4: 统计图 + Word 报告", visible=True),
            gr.update(value=status, visible=True),
            gr.update(value="", visible=False),         # report_md hidden
            gr.update(value=word_path, visible=True),    # word download
            gr.update(visible=False),                    # btn4 hidden
            *[gr.update(value=ci, visible=True) if ci else gr.update(visible=False) for ci in chart_images])


# ════════════════════════════════════════════
# UI
# ════════════════════════════════════════════

def build_interface():
    with gr.Blocks(title="MFRMSight v1.0", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 📊 MFRMSight v1.0.0 — 多面 Rasch 模型分析")

        # Step 1
        gr.Markdown("### Step 1: 上传 + 选面")
        file_input = gr.File(label="上传 .txt 或 .xlsx", file_types=[".txt", ".xlsx", ".xls"])
        dim_preview = gr.Markdown("👆 请先上传文件")
        facet_select = gr.CheckboxGroup(label="勾选要分析的面", choices=[], interactive=True)
        btn1 = gr.Button("🚀 开始分析", variant="primary")
        step1_status = gr.Markdown("")

        # Step 2
        step2_header = gr.Markdown("", visible=True)
        with gr.Row():
            t1 = gr.DataFrame(visible=False); t2 = gr.DataFrame(visible=False)
        with gr.Row():
            t3 = gr.DataFrame(visible=False); t4 = gr.DataFrame(visible=False)
        rank_table = gr.DataFrame(label="📊 排名对比", visible=False)

        # Step 3 (hidden until step2 done)
        step3_header = gr.Markdown(visible=False)
        inter_select = gr.CheckboxGroup(label="勾选交互对", choices=[], visible=False)
        btn3 = gr.Button("🔍 分析交互项", variant="primary", visible=False)
        bias_summary = gr.Markdown(visible=False)
        bias_table = gr.DataFrame(label="显著偏差交互 (|z| >= 2)", visible=False)

        # Step 4 (hidden until step3 done)
        step4_header = gr.Markdown(visible=False)
        step4_status = gr.Markdown(visible=False)

        with gr.Row(visible=True):
            btn4 = gr.Button("📄 生成 Word 报告 + 图表", variant="secondary", visible=False)

        with gr.Row():
            chart_img1 = gr.Image(label="垂直标尺图", visible=False, show_download_button=True, show_fullscreen_button=True)
            chart_img2 = gr.Image(label="拟合统计量分布图", visible=False, show_download_button=True, show_fullscreen_button=True)
        chart_img3 = gr.Image(label="等级概率曲线与 ICC", visible=False, show_download_button=True, show_fullscreen_button=True)

        report_dl = gr.File(label="📥 下载 Word 报告", visible=False)

        # ── Events ──
        analysis_outputs = [step2_header, t1, t2, t3, t4,
                           gr.Textbox(visible=False), gr.Textbox(visible=False), gr.Textbox(visible=False),  # 3 chart placeholders
                           rank_table,
                           step3_header, inter_select, btn3,
                           step4_header, step4_status, report_md, report_dl]

        file_input.change(fn=step1_upload, inputs=[file_input],
                          outputs=[dim_preview, facet_select])

        btn1.click(fn=lambda: "⏳ 分析中，请稍候...", outputs=[step1_status])\
             .then(fn=step2_analyze, inputs=[facet_select], outputs=analysis_outputs)\
             .then(fn=lambda: "", outputs=[step1_status])

        btn3.click(fn=step3_bias, inputs=[inter_select],
                   outputs=[bias_summary, bias_table, step4_header, step4_status, report_dl, btn4])

        btn4.click(fn=step4_generate,
                   outputs=[step4_header, step4_status, report_md, report_dl, btn4,
                            chart_img1, chart_img2, chart_img3])

        gr.Markdown("MFRMSight v1.0.0 · Andrich Rating Scale · Fisher-scoring JMLE")

    return app


def main():
    app = build_interface()
    port = int(os.environ.get("PORT", "7870"))
    print(f"MFRMSight v1.0.0 — port={port}, PID={os.getpid()}")
    app.launch(server_name="0.0.0.0", server_port=port, share=False)


if __name__ == "__main__":
    main()
