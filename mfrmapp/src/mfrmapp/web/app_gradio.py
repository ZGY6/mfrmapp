"""
MFRMSight Web — Gradio 版 v0.8.0
纯 HTTP，不需要 WebSocket，代理环境下也能正常打开。
运行: python app_gradio.py
支持 Render.com 环境变量 PORT 自动适配端口。
"""
import gradio as gr
import pandas as pd
import numpy as np
import tempfile, os, sys
from pathlib import Path

_parent = Path(__file__).parent.parent
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))
from engine import parse_facets_txt, MFRMEngine


def analyze(file):
    """分析上传的文件，返回结果"""
    if file is None:
        return "请先上传文件", None, None, None, None

    suffix = Path(file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        # 兼容 Gradio 传 bytes 和传文件路径两种情况
        if isinstance(file, bytes):
            f.write(file)
        else:
            # Gradio 可能传 file-like 对象，优先用 .read()，失败则用路径
            try:
                f.write(file.read())
            except Exception:
                f.write(open(file.name, "rb").read())
        tmp = f.name
    try:
        if suffix == ".txt":
            data = parse_facets_txt(tmp)
        else:
            from engine import parse_excel
            data = parse_excel(tmp)
        eng = MFRMEngine(data)
        eng.fit()
    finally:
        os.unlink(tmp)

    r = eng.report()
    s = r["summary"]

    # 摘要文本
    summary_text = (
        f"### 分析结果\n"
        f"反应数: {s['N']} | 面向: {s['n_s']}学生×{s['n_r']}评分者×{s['n_c']}标准"
        + (f"×{s['n_i']}题目" if s['n_i'] > 1 else "") + f"\n"
        f"分数范围: {s['score_range']} | 方差解释: {s['var_exp']}%\n"
        f"ObsMean={s['obs_mean']:.2f} | ExpMean={s['exp_mean']:.2f} | "
        f"ResidSD={s['resid_sd']:.4f} | StResSD={s['stres_sd']:.4f}\n"
    )

    # 各面向的 DataFrames
    dfs = {}
    seps = {}
    for key, emoji in [("students", "🎓"), ("raters", "👤"), ("criteria", "📋"), ("items", "📝")]:
        fd = r["facets"].get(key)
        if fd and fd["rows"]:
            df = pd.DataFrame(fd["rows"])
            df_display = df[["label", "total", "obs_avg", "meas", "se", "infit", "outfit"]].copy()
            df_display.columns = ["名称", "总分", "ObsAvg", "Meas", "SE", "Infit", "Outfit"]
            df_display["总分"] = df_display["总分"].astype(int)
            dfs[f"{emoji} {key} (Sep={fd['separation']:.2f} Rel={fd['reliability']:.3f})"] = df_display
            seps[key] = f"Sep={fd['separation']:.2f} Rel={fd['reliability']:.3f}"

    # Excel 导出
    buf = pd.ExcelWriter(os.path.join(tempfile.gettempdir(), "mfrm_output.xlsx"), engine="openpyxl")
    for k, fd in r["facets"].items():
        if fd["rows"]:
            pd.DataFrame(fd["rows"]).to_excel(buf, sheet_name=k, index=False)
    pd.DataFrame([s]).to_excel(buf, sheet_name="summary", index=False)
    buf.close()

    return summary_text, dfs.get(list(dfs.keys())[0] if dfs else None), dfs.get(list(dfs.keys())[1] if len(dfs) > 1 else None), dfs.get(list(dfs.keys())[2] if len(dfs) > 2 else None), dfs.get(list(dfs.keys())[3] if len(dfs) > 3 else None)


def build_interface():
    with gr.Blocks(title="MFRMSight", theme=gr.themes.Soft()) as app:
        gr.Markdown(
            """
            # 📊 MFRMSight — 多面Rasch模型分析
            **上传 Facets .txt 或 Excel .xlsx 文件，自动运行 MFRM 分析**
            """
        )

        file_input = gr.File(label="上传数据", file_types=[".txt", ".xlsx", ".xls"])

        with gr.Row():
            run_btn = gr.Button("🚀 开始分析", variant="primary")

        summary = gr.Markdown()

        with gr.Row():
            with gr.Column(scale=1):
                table1 = gr.DataFrame(label="🎓 学生")
            with gr.Column(scale=1):
                table2 = gr.DataFrame(label="👤 评分者")

        with gr.Row():
            with gr.Column(scale=1):
                table3 = gr.DataFrame(label="📋 标准")
            with gr.Column(scale=1):
                table4 = gr.DataFrame(label="📝 题目")

        run_btn.click(
            fn=analyze,
            inputs=[file_input],
            outputs=[summary, table1, table2, table3, table4],
        )

        gr.Markdown("MFRMSight v0.8.0 · Andrich Rating Scale Model · Fisher-scoring JMLE")

    return app


def main():
    """入口点: mfrmapp-gradio (适配 Render.com PORT 环境变量)"""
    app = build_interface()

    port = int(os.environ.get("PORT", "7870"))
    # BUG-006: PID 追踪
    print(f"🚀 MFRMSight v0.8.0 — 启动 Gradio 服务: port={port}, PID={os.getpid()}")
    app.launch(server_name="0.0.0.0", server_port=port, share=False)


if __name__ == "__main__":
    main()
