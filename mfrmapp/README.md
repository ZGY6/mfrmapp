# MFRMSight — 多面Rasch模型分析工具 v0.8.0

基于 Andrich Rating Scale Model (1978)，使用 Fisher-scoring JMLE 进行参数估计。

## 使用

### 🌐 Web 界面（推荐）

```bash
# Gradio Web（支持文件上传、在线分析、结果导出）
mfrmapp-gradio

# 浏览器打开 http://localhost:7870
# 或访问在线版: https://mfrmsight.onrender.com
```

在线版无需安装，上传数据即可分析。

### 命令行

```bash
# 分析 Facets 格式数据
mfrmapp data.txt

# 分析 Excel 数据
mfrmapp data.xlsx

# 导出结果
mfrmapp data.txt -o result.xlsx
mfrmapp data.xlsx -o report.docx
```

### Python API

```python
from mfrmapp import parse_facets_txt, MFRMEngine

# 加载数据
data = parse_facets_txt("data.txt")
engine = MFRMEngine(data)
engine.fit()

# 打印报告
engine.print()

# 导出
engine.to_excel("result.xlsx")
engine.to_word("report.docx")

# 获取结构化数据
report = engine.report()
print(report["summary"]["var_exp"])  # 方差解释率
print(report["facets"]["students"]["rows"])  # 学生详情
```

## 安装

```bash
# 从源码安装
pip install .

# 安装 Word 输出支持（可选）
pip install ".[word]"

# 安装 Streamlit 界面（可选）
pip install ".[web]"
```

## 数据格式

### Facets .txt 格式

```
Facets=4
Noncentered=1
Model=?,?,?,?,R23
*
Labels=
1,Students
1,Student1
...
*
Data=
1,1,1,1,8
```

### Excel 格式

| 编号 | 评分人 | 综合分析-1号 | 综合分析-2号 | ... |
|------|--------|------------|------------|-----|
| 1 | Rater1 | 8 | 5 | ... |

支持多行标题自动合并（如第一行 "沟通能力" + 第二行 "1号" → "沟通能力-1号"）。

## 部署

本项目已部署到 Render.com：<https://mfrmsight.onrender.com>

如需自行部署，使用仓库中的 `render.yaml` (Blueprint) 或手动配置：
- Build: `pip install uv && uv sync --no-dev`
- Start: `uv run mfrmapp-gradio`

## 测试

```bash
# 运行单文件版测试（32 用例）
uv run python -m pytest tests/ -v

# 运行包版测试（16 用例）
cd mfrmapp && uv run python -m pytest tests/ -v
```

## 依赖

- Python >= 3.10
- numpy, pandas, scipy, openpyxl, gradio
- (可选) python-docx → Word 输出
- (可选) streamlit → Streamlit Web 界面

## 许可

MIT License
