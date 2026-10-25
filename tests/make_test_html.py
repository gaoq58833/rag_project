"""生成一份结构化 HTML 测试文档，覆盖标题/表格/图片/列表/超长段落"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "test_file", "rag_test_doc.html")

long_para = (
    "这是一段用来测试超长段落降级切分的文字。" * 60
)

html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="utf-8"><title>RAG 测试文档</title></head>
<body>
<h1>第一章 RAG 文档处理测试</h1>
<p>这是一份用于测试 HTML 解析与分块的结构化文档。</p>

<h2>1.1 背景介绍</h2>
<p>RAG（检索增强生成）通过从外部知识库检索相关内容来辅助大模型回答，文档解析与分块的质量直接影响检索效果。</p>
<ul>
<li>支持 PDF、Word、Markdown、HTML 四种输入</li>
<li>统一输出 markdown 格式</li>
<li>分块保留章节路径与图片归属</li>
</ul>

<h2>1.2 核心模块</h2>
<p>下表列出各个解析器及其原理：</p>
<table>
<thead><tr><th>解析器</th><th>原理</th><th>是否用模型</th></tr></thead>
<tbody>
<tr><td>PDFParser</td><td>docling 版式模型 + TableFormer</td><td>是</td></tr>
<tr><td>WordParser</td><td>python-docx 读 OOXML</td><td>否</td></tr>
<tr><td>MarkdownParser</td><td>直接读取</td><td>否</td></tr>
<tr><td>HTMLParser</td><td>markdownify 转 DOM</td><td>否</td></tr>
</tbody>
</table>

<h3>1.2.1 分块策略</h3>
<p>段落优先，超长自动降级到句子级、字符级。段间不做 overlap，段内保留 overlap 保证语义连贯。</p>

<h2>1.3 超长段落测试</h2>
<p>{long_para}</p>

<h2>1.4 图片测试</h2>
<p>下面是一张示例图片：</p>
<img src="_fig1.png" alt="架构图">

<h2>1.5 大表格测试</h2>
<p>下表有很多行，用于测试表格按行窗口切分：</p>
<table>
<thead><tr><th>序号</th><th>名称</th><th>描述</th></tr></thead>
<tbody>
{''.join(f'<tr><td>{i}</td><td>项目{i}</td><td>这是第{i}行的描述内容，用于填充表格使其超过最大块长度限制</td></tr>' for i in range(1, 31))}
</tbody>
</table>

<h2>1.6 嵌套标题</h2>
<h3>1.6.1 子节 A</h3>
<p>子节 A 的内容。</p>
<h4>1.6.1.1 孙节</h4>
<p>孙节内容，用于测试四级标题路径。</p>
</body>
</html>
"""

open(OUT, "w", encoding="utf-8").write(html)
print(f"已生成：{OUT}")
