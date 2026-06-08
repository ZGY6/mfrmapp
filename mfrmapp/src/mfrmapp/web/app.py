"""
MFRMSight Web — 多面Rasch分析应用 (手机版)
streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import tempfile, os, sys
from pathlib import Path
from io import BytesIO

_parent = Path(__file__).parent.parent
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))
from engine import parse_facets_txt, MFRMEngine

st.set_page_config(
    page_title="MFRMSight",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
.main-title { font-size: 1.8rem; font-weight: 700; text-align: center; margin-bottom: 0.2rem; }
.subtitle { font-size: 0.85rem; color: #888; text-align: center; margin-bottom: 1rem; }
.mcard { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
         border-radius: 12px; padding: 0.8rem; text-align: center; color: #fff; }
.mcard .v { font-size: 1.5rem; font-weight: 700; }
.mcard .l { font-size: 0.7rem; opacity: 0.9; }
.ftitle { font-size: 1rem; font-weight: 600; color: #1a56db; margin-top: 1rem; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">📊 MFRMSight</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">多面Rasch模型 · 手机版</div>', unsafe_allow_html=True)

uploaded = st.file_uploader("📂 上传数据", type=["txt", "xlsx", "xls"],
                            help="Facets .txt 或 Excel .xlsx")

with st.expander("📋 数据格式说明"):
    st.markdown("""
**Facets .txt:**
```
Facets=4
Positive=1
*
Labels=
1,Students
1,S1
*
Data=
1,1,1,1,8
```

**Excel:**
| 编号 | 评分人 | 维度A-1号 | 维度B-1号 | ... |
|------|--------|----------|----------|-----|
""")

if uploaded:
    suffix = Path(uploaded.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(uploaded.read())
        tmp = f.name
    try:
        with st.spinner("⏳ 分析中..."):
            if suffix == ".txt":
                data = parse_facets_txt(tmp)
            else:
                from engine import parse_excel
                data = parse_excel(tmp)
            engine = MFRMEngine(data)
            engine.fit()
        r = engine.report()
        s = r["summary"]

        # 指标卡
        st.markdown("### 📈 结果")
        cols = st.columns(5)
        for col, (label, value) in zip(cols, [
            ("反应数", s["N"]),
            ("面向", f"{s['n_s']}×{s['n_r']}×{s['n_c']}" + (f"×{s['n_i']}" if s['n_i'] > 1 else "")),
            ("分数", s["score_range"]),
            ("方差解释", f"{s['var_exp']}%"),
            ("残差SD", s["resid_sd"]),
        ]):
            with col:
                st.markdown(f'<div class="mcard"><div class="v">{value}</div><div class="l">{label}</div></div>',
                            unsafe_allow_html=True)

        st.caption(f"ObsMean={s['obs_mean']:.2f} | ExpMean={s['exp_mean']:.2f} | "
                   f"StResSD={s['stres_sd']:.4f} | LL={s['ll']:.0f}")

        # 各面向
        for key, emoji, title, cols_d in [
            ("students", "🎓", "学生面向", ["名称", "总分", "ObsAvg", "FairAvg", "Meas", "SE", "Infit", "Outfit"]),
            ("raters", "🧑‍⚖️", "评分者面向", ["名称", "总分", "ObsAvg", "Meas", "SE", "Infit", "Outfit"]),
            ("criteria", "📋", "标准面向", ["名称", "总分", "ObsAvg", "Meas", "SE", "Infit", "Outfit"]),
            ("items", "📝", "题目面向", ["名称", "总分", "ObsAvg", "Meas", "SE", "Infit", "Outfit"]),
        ]:
            fd = r["facets"].get(key)
            if not fd or not fd["rows"]:
                continue
            if key == "raters":
                st.image("mfrmapp/src/mfrmapp/web/static/rater_icon.jpg", width=40)
            st.markdown(f'<div class="ftitle">{emoji} {title} — Sep={fd["separation"]:.2f} | Rel={fd["reliability"]:.3f}</div>',
                        unsafe_allow_html=True)
            df = pd.DataFrame(fd["rows"])
            cmap = {"名称": "label", "总分": "total", "ObsAvg": "obs_avg",
                    "FairAvg": "fair_avg", "Meas": "meas", "SE": "se",
                    "Infit": "infit", "Outfit": "outfit"}
            df_d = df[[cmap[c] for c in cols_d]].copy()
            df_d.columns = cols_d
            for c in ["总分"]:
                if c in df_d.columns:
                    df_d[c] = df_d[c].astype(int)
            st.dataframe(df_d, use_container_width=True, hide_index=True)

        # 下载
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            buf = BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as w:
                for k, fd in r["facets"].items():
                    if fd["rows"]:
                        pd.DataFrame(fd["rows"]).to_excel(w, sheet_name=k, index=False)
                pd.DataFrame([s]).to_excel(w, sheet_name="summary", index=False)
            st.download_button("📥 Excel", buf.getvalue(), "mfrm_result.xlsx",
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        with col2:
            csv_buf = BytesIO()
            all_df = pd.concat([
                pd.DataFrame(fd["rows"]).assign(facet=k)
                for k, fd in r["facets"].items() if fd["rows"]
            ])
            csv_buf.write(all_df.to_csv(index=False).encode("utf-8-sig"))
            st.download_button("📥 CSV", csv_buf.getvalue(), "mfrm_result.csv", "text/csv")

        st.caption("MFRMSight v0.8.0 · Andrich Rating Scale Model · Fisher-scoring JMLE")

    except Exception as e:
        st.error(f"分析失败: {e}")
        import traceback
        st.code(traceback.format_exc())
    finally:
        os.unlink(tmp)
else:
    st.info("👆 上传 .txt 或 .xlsx 文件开始分析")
