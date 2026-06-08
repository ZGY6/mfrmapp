# MFRMSight — 多面Rasch模型分析工具 v0.8.0

基于Andrich Rating Scale Model (1978)，使用Fisher-scoring JMLE进行参数估计。

## 安装

```bash
# 从源码安装
pip install .

# 安装Word输出支持
pip install ".[word]"
```

## 使用

### 命令行

```bash
# 分析Facets格式数据
mfrmapp data.txt

# 分析Excel数据
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

## 数据格式

### Facets .txt 格式

```
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
1,Rater1
...
*
3,Criteria
1,comp
2,inte
*
4,Items
1,Item1
...
*
Data=
1,1,1,1,8
1,1,1,2,5
...
```

### Excel 格式

| 编号 | 评分人 | 综合分析-1号 | 综合分析-2号 | ... | 人际沟通-4号 |
|------|--------|------------|------------|-----|------------|
| 1 | Rater1 | 8 | 5 | ... | 20 |

## 依赖

- Python >= 3.10
- numpy, pandas, scipy, openpyxl
- (可选) python-docx → Word输出

## 许可

MIT License
