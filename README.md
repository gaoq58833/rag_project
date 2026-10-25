# 📄 RAG 文档处理管线

> 把 PDF / Word / Markdown / HTML / TXT 转成结构化 markdown，切成语义完整的检索块，向量化后存入 Qdrant，业务数据入 PostgreSQL，提供 **入库 → 检索 → 重排 → 问答** 全链路 REST API。

[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.162%2B-009688)](https://fastapi.tiangolo.com/)
[![Formats](https://img.shields.io/badge/formats-PDF%20%7C%20DOCX%20%7C%20MD%20%7C%20HTML%20%7C%20TXT-orange)]()
[![Vector DB](https://img.shields.io/badge/vector-Qdrant-purple)]()
[![DB](https://img.shields.io/badge/db-PostgreSQL-blue)]()
[![License](https://img.shields.io/badge/license-Unlicense-lightgrey)]()

---

## ✨ 功能特性

- **多格式解析**：PDF（docling 版式模型）、Word（python-docx）、Markdown、HTML、TXT 一站式覆盖
- **智能分块**：段落优先的级联降级策略（段落 → 句子 → 字符），表格原子化、图片归属、章节路径溯源
- **解析器工厂**：按扩展名自动路由 + 单例缓存，PDF 模型只加载一次
- **RAG 全链路**：入库（解析→分块→向量化）→ 向量召回（Qdrant HNSW）→ 重排（bge-reranker-v2-m3）→ LLM 问答（带引用来源）
- **业务库**：完整 chunk 数据入 PostgreSQL（JSONB），Qdrant 只存向量 + 最小 payload，检索后回查 PG
- **REST API**：FastAPI + uvicorn，`/ingest`、`/retrieve`、`/chat` 等，Swagger 文档自动生成
- **量化评测**：百分制打分（完整性 / 块大小 / 表格 / 粒度 / 可追溯），结果可复现
- **容器化部署**：Dockerfile 就绪，CPU / GPU 双版本支持

---

## 📊 量化指标（100+ 文档实测）

> 覆盖 PDF / Word / Markdown / HTML / TXT 多格式样本，100+ 文档实测结论。

### 解析准确率

| 指标 | 数值 | 适用格式 | 说明 |
| --- | --- | --- | --- |
| 整篇文档字符 F1 | **98.94%** | PDF | 准确率与召回率调和平均（17946/18277 字） |
| 表格抽取准确率 | **96.13%** | PDF | 逐格比对，忽略空白与 markdown 强调符（149/155） |

### 分块质量评分（满分 100）

| 格式 | 总分 | 完整性 | 表格完整性 | 结构可追溯 |
| --- | --- | --- | --- | --- |
| **HTML** | **100.0** | 100% | 100% | 100% |
| **PDF** | **95.2** | 100% | 100% | 100% |
| **Markdown** | **89.1** | 100% | 100% | 100% |
| **自造 Word** | **88.0** | 100% | 100% | 100% |
| **真实 Word 报告** | **70.0** | 100% | 100% | 50% |

- **分块内容完整性 100%**：全格式字符丢失 0 字，表格/代码/图片引用全覆盖
- 表格块 100% 保留表头 + 行号前缀，正文块 100% 携带章节路径（真实 Word 报告因未用 Heading 样式除外）
- 图片全部落盘 `data/images/` 并正确归属到所在块的 `images` 字段

---

## 🚀 快速开始

### 前置要求

- Python ≥ 3.12
- [uv](https://docs.astral.sh/uv/) 包管理器（推荐）
- PostgreSQL 14+（业务库，存完整 chunk 数据）
- OpenAI 兼容 API Key（Embedding + LLM + Rerank，如阿里云百炼、火山方舟）
- PDF 解析建议 ≥ 8GB 空闲内存

### 1. 安装依赖

```bash
git clone <仓库地址>
cd rag_project
uv sync
```

### 2. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env`，填写以下关键项：

| 变量 | 说明 |
| --- | --- |
| `LLM_API_KEY` | 你的 API 密钥（Embedding/LLM/Rerank 共用或分别填） |
| `LLM_BASE_URL` | OpenAI 兼容端点，如 `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `EMBEDDING_MODEL` | 向量化模型，默认 `text-embedding-flash`（1024 维） |
| `LLM_MODEL` | 问答模型，默认 `qwen3.7-plus` |
| `RERANKER_MODEL` | 重排模型，默认 `bge-reranker-v2-m3` |
| `PG_*` | PostgreSQL 连接信息（host/port/db/user/password） |
| `QDRANT_PATH` | 本地模式路径，默认 `./data/qdrant`（无需启 Qdrant 服务） |

### 3. 准备 PostgreSQL

创建数据库（表结构由服务启动时自动创建，无需手动执行 SQL）：

```sql
CREATE DATABASE rag;
```

### 4. 启动服务

```bash
uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

打开 http://localhost:8000/docs 查看交互式 API 文档。

### 三行代码调用

```python
from src.parsers.factory import parse

# 自动识别格式，一步解析并分块
result = parse("实验报告.pdf")
print(result.chunks)        # 分块结果列表
print(result.raw_text)      # 完整 markdown 文本
```

---

## 📑 目录

- [处理流程](#-处理流程)
- [核心模块](#-核心模块)
- [RAG 数据链路](#-rag-数据链路)
- [HTTP API](#-http-apifastapi)
- [Docker 部署](#-docker-部署)
- [测试与评测](#-测试与评测)
- [目录结构](#-目录结构)
- [已知限制](#-已知限制)
- [环境变量](#-环境变量)

---

## 🔄 处理流程

```
原始文件 ──> 解析器 ──> markdown ──> 清洗 ──> 分块器 ──> chunks
              │                                            │
              └── 图片存 data/images/                       ├──> PostgreSQL（完整 chunk 数据）
                                                           └──> 向量化 ──> Qdrant（向量 + 最小 payload）

检索： query ──> 向量化 ──> Qdrant 召回 ──> 回查 PG 取完整数据 ──> 重排 ──> 相关块
问答： 相关块 ──> 构造上下文 ──> LLM 生成回答 + 引用来源
```

| 环节      | 模块                                         | 说明                                                               |
| --------- | -------------------------------------------- | ------------------------------------------------------------------ |
| 工厂      | `src/parsers/factory.py`                     | 按扩展名自动选解析器 + 单例缓存，支持 `parse()` 一步到位            |
| PDF 解析  | `src/parsers/pdf_parser.py`                  | docling：版式模型定位段落/表格，TableFormer 还原表格，图片另存本地 |
| Word 解析 | `src/parsers/word_parser.py`                 | python-docx 直接读 OOXML，不用模型、无 OCR 噪音                    |
| MD 解析   | `src/parsers/markdown_parser.py`             | 直接读 .md，复用清洗与分块逻辑                                     |
| HTML 解析 | `src/parsers/html_parser.py`                 | 去除 `<script>`/`<style>`，正文转 markdown，编码自动回退           |
| TXT 解析  | `src/parsers/txt_parser.py`                  | 纯文本按段落切分，不做标题入栈，编码自动回退                       |
| 表格处理  | `src/parsers/table_parser.py`                | 跨页表合并、markdown ↔ DataFrame、表格转文本                       |
| 清洗      | `src/parsers/cleaner.py`                     | 去页码/分隔线/零宽字符                                             |
| 分块      | `src/parsers/chunker.py`                     | 段落优先的降级分块 + 章节路径 + 表格原子单元 + 图片归属            |
| LLM 客户端| `src/llm/client.py`                          | OpenAI 兼容：embed / chat / rerank，统一走一个 SDK                  |
| 向量库    | `src/vectorstore/qdrant_store.py`            | Qdrant 本地/服务双模式，HNSW 索引，payload 过滤                    |
| 业务库    | `src/db/database.py` / `repository.py`       | PostgreSQL 连接池 + documents/chunks CRUD，启动自动建表            |
| RAG 管线  | `src/rag/pipeline.py`                        | ingest_file / retrieve / chat 三函数，串联全链路                    |
| 评测      | `量化/eval_chunks.py`、`量化/eval_tables.py` | 百分制打分 / 表格准确率比对，结果写入 `量化/指标记录.md`           |

---

## 🧩 核心模块

### ParserFactory（推荐入口）

无需手动判断格式，工厂根据扩展名自动选择解析器，并复用单例实例（PDFParser 的 docling 模型只加载一次）。

```python
from src.parsers.factory import get_parser, parse, parse_from_bytes

# 方式一：拿到解析器再调用
parser = get_parser("上传文件.pdf")
result = parser.parse("上传文件.pdf")

# 方式二：一步到位
result = parse("上传文件.docx")          # 自动用 WordParser
result = parse("笔记.txt")              # 自动用 TXTParser

# 从字节流解析（Web 上传场景）
result = parse_from_bytes(file_bytes, "report.html")
```

- 默认参数走**单例缓存**：`get_parser("a.pdf") is get_parser("b.pdf")`
- 传自定义参数时每次新建：`get_parser("a.pdf", do_ocr=False)` 不缓存
- 不支持的格式抛 `ValueError`，用 `supported_extensions()` 查看支持列表
- 支持动态注册：`register_parser(".foo", SomeParser)`

### PDFParser

```python
from src.parsers.pdf_parser import PDFParser

parser = PDFParser(do_ocr=False)          # 文字版 PDF 建议关 OCR：更稳、更快、表格不丢
result = parser.parse("test_file/xxx.pdf")

result.raw_text    # markdown，图片位置是 ![图片1](data/images/xxx_1.png)
result.tables      # 表格列表（已合并跨页表）
result.total_pages # 页数
```

| 参数           | 默认          | 说明                                           |
| -------------- | ------------- | ---------------------------------------------- |
| `do_ocr`       | `True`        | 文字版 PDF 建议 `False`；开启 OCR 会随机丢表格 |
| `ocr_scale`    | `1.5`         | OCR 区域放大倍数，默认 3.0 内存占用过大        |
| `images_scale` | `1.0`         | 导出图片分辨率                                 |
| `image_dir`    | `data/images` | 图片落盘目录                                   |

### WordParser

```python
from src.parsers.word_parser import WordParser

result = WordParser().parse("test_file/xxx.docx")
```

- 只支持 `.docx`，`.doc` 老二进制格式需先转换（见[已知限制](#-已知限制)）
- 标题样式 `Heading 1-6` / `标题 1-6` 映射成对应数量的 `#`（最多 6 级）
- 遍历 body 子节点保留表格在文档中的真实顺序
- 兼容新格式 `a:blip` 和老式 VML `v:imagedata`，表格单元格内图片也能提取

### MarkdownParser

```python
from src.parsers.markdown_parser import MarkdownParser

result = MarkdownParser().parse("test_file/xxx.md")
```

直接读取 `.md` / `.markdown`，表格（`|...|` 行）抽取并复用 `TableProcessor` 规范化，标题沿用原生 `#` 语法。

### HTMLParser

```python
from src.parsers.html_parser import HTMLParser

result = HTMLParser().parse("test_file/xxx.html")
```

去除 `<script>`、`<style>`、注释等噪音，编码按 `utf-8 → utf-8-sig → gb18030` 回退，表格/列表/标题/代码块结构尽量保留。

### TXTParser

```python
from src.parsers.txt_parser import TXTParser

result = TXTParser().parse("test_file/xxx.txt")
```

TXT 结构化弱，**不做标题入栈**，分块走段落策略（空行分段 → 句子 → 字符）。编码自动回退，无表格/图片。

### TextChunker

```python
from src.parsers.chunker import TextChunker

chunker = TextChunker(max_chunk_size=1000, min_chunk_size=100, chunk_overlap=200)
chunks = chunker.split(markdown_text)
```

**级联降级策略：**

1. **段落优先**：按空行切段，逐段累加，超 `max_chunk_size` 则输出
2. **段落超长 → 句子级**：按 `。！？；` 切句重组，保留 `chunk_overlap`
3. **单句仍超长 → 字符级**：固定长度硬切，残余并入上一块，不丢字
4. **段落间不重叠**：相邻段落主题可能不同，重叠会污染检索

**特殊规则：**

- **章节路径**：标题栈实时推导，如 `三、双向 LSTM > 3.2 取最后隐状态`；跨章节用 `；` 并列
- **表格原子单元**：整表放得下就一块；超限按行窗口切，每块重复表头 + `【表格 · 第x-y行/共N行】` 前缀
- **图片不单独成块**：引用行并入当前块，路径记入 `images` 字段

**块数据结构：**

```json
{
  "content": "……正文……",
  "index": 0,
  "length": 856,
  "type": "text",
  "section": "第二章 分块策略 > 2.1 长段落处理",
  "images": ["data/images/xxx_1.png"]
}
```

---

## 🔗 RAG 数据链路

### 入库（ingest）

```
文件 → 解析 → 分块 → 向量化(content) → 同时写入：
  ├─ PostgreSQL.documents  （文档元数据）
  ├─ PostgreSQL.chunks     （完整 chunk：content/section/type/images...）
  └─ Qdrant                （向量 + 最小 payload：qdrant_point_id/filename/section）
```

- **同名文件覆盖**：入库前按 filename 删旧记录（PG 级联删 chunks，Qdrant 按 payload 删）
- **Point ID 用 UUID**：Qdrant 本地模式要求，同时存入 PG 的 `chunks.qdrant_point_id` 用于回查

### 检索（retrieve）

```
query → embed → Qdrant.query_points(top_k=10) → 拿到 qdrant_point_id 列表
      → 回 PostgreSQL 查完整 chunk → bge-reranker 重排(top_n=5) → 返回
```

### 问答（chat）

```
retrieve 结果 → 拼接上下文 → system prompt + user query → LLM 生成 → answer + sources
```

### PostgreSQL 表结构

| 表 | 关键字段 | 说明 |
| --- | --- | --- |
| `documents` | id, filename, file_ext, total_pages, table_count, chunk_count, raw_text, uploaded_at | 每个上传文件一条 |
| `chunks` | id, document_id(FK), qdrant_point_id, chunk_index, content, length, chunk_type, section, images(JSONB) | 每个分块一条，通过 qdrant_point_id 关联 Qdrant |

索引：`documents(filename)`、`chunks(document_id)`、`chunks(qdrant_point_id)`、`chunks(chunk_type)`、`chunks(section)`、`chunks(images)` GIN。

> 表结构定义在 `src/db/schema.sql`，服务启动时自动执行建表（幂等）。

---

## 🌐 HTTP API（FastAPI）

### 启动

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

启动时预热 PDFParser（加载 docling 模型），首次请求不卡顿。API 文档在 `/docs`（Swagger UI）和 `/redoc`。

### 接口列表

| 方法 | 路径        | 说明                                                       |
| ---- | ----------- | ---------------------------------------------------------- |
| GET  | `/health`   | 健康检查                                                   |
| GET  | `/formats`  | 返回支持的扩展名列表                                       |
| POST | `/parse`    | 上传文件 → 返回 `raw_text`、`tables`、`metadata`（不分块） |
| POST | `/chunk`    | 上传文件 → 解析并分块，返回 `chunks` 列表                  |
| POST | `/ingest`   | 上传文档 → 解析分块 → 向量化 → 写入 PostgreSQL + Qdrant    |
| POST | `/retrieve` | `query` → 向量召回 → 重排 → 返回相关块（含来源）           |
| POST | `/chat`     | `query` → 检索 → LLM 生成回答 + 引用来源                   |

`/chunk` 支持查询参数覆盖分块大小：`?max_chunk_size=800&min_chunk_size=100&chunk_overlap=150`。
`/retrieve` 支持 `?query=xxx&top_k=10&use_rerank=true`。

### 调用示例

```python
import requests

# 解析
with open("报告.pdf", "rb") as f:
    r = requests.post("http://localhost:8000/parse", files={"file": f})
print(r.json()["raw_text"])

# 解析并分块
with open("笔记.txt", "rb") as f:
    r = requests.post("http://localhost:8000/chunk", files={"file": f})
for c in r.json()["chunks"]:
    print(c["type"], c["length"], c["content"][:50])

# 入库到知识库
with open("实验报告.docx", "rb") as f:
    r = requests.post("http://localhost:8000/ingest", files={"file": f})
print(r.json())  # {filename, chunk_count, stored_count}

# 检索
r = requests.post("http://localhost:8000/retrieve", params={"query": "循环神经网络原理"})
for c in r.json()["results"]:
    print(f"[{c['score']:.3f}] {c['section']} | {c['content'][:60]}")

# 问答
r = requests.post("http://localhost:8000/chat", json={"query": "RNN 为什么能处理序列？", "use_rerank": True})
print(r.json()["answer"])
for s in r.json()["sources"]:
    print(f"  - 来源: {s['filename']} > {s['section']}")
```

```bash
# 入库
curl -X POST http://localhost:8000/ingest -F "file=@test_file/sample.docx"

# 检索
curl -X POST "http://localhost:8000/retrieve?query=RNN"

# 问答
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" \
  -d '{"query": "循环神经网络的工作原理"}'
```

> 不支持的格式返回 `400`，解析失败 / API Key 未配置返回 `500`。

---

## 🐳 Docker 部署

### CPU 版本

```bash
docker build -t rag-project:cpu .
docker run -p 8000:8000 rag-project:cpu
```

### GPU 版本

需要宿主机安装 NVIDIA 驱动 + `nvidia-container-toolkit`：

```bash
docker build -f Dockerfile.gpu -t rag-project:gpu .
docker run --gpus all -p 8000:8000 rag-project:gpu
```

> 容器内使用宿主机显卡：CUDA 工具链在镜像里，驱动和物理 GPU 在宿主机。

---

## 🧪 测试与评测

### 测试命令

| 命令                                          | 作用                                             |
| --------------------------------------------- | ------------------------------------------------ |
| `python tests/test_pdf.py`                    | 解析 PDF，输出 markdown                          |
| `python tests/test_cleaner.py`                | 验证清洗规则                                     |
| `python tests/test_table.py`                  | 表格解析、跨页合并、转 DataFrame                 |
| `python tests/test_chunker.py`                | PDF 全流程 → 分块 → 写 `data/chunk_test/`        |
| `python tests/test_word.py`                   | Word 解析 → 分块 → 写 `data/word_test/`          |
| `python tests/test_markdown.py`               | Markdown 解析 → 分块 → 写 `data/md_test/`        |
| `python tests/test_html.py`                   | HTML 解析 → 分块 → 写 `data/html_test/`          |
| `python tests/test_txt.py`                    | TXT 解析 → 分块 → 写 `data/txt_test/`            |
| `python tests/test_factory.py`                | 解析器工厂：路由、单例、自定义参数、错误处理      |
| `python tests/test_api.py`                    | FastAPI 接口：health/parse/chunk/错误处理        |

`test_*.py` 都会报告：解析覆盖率、块数、超限/碎片块、丢失字符、重复字符、章节路径覆盖、图片归属。

### 量化评测

```bash
# 参数：数据目录 [章节号] [章节标题]
python 量化/eval_chunks.py data/word_test_gen 5 "自造 Word 测试文档"
python 量化/eval_chunks.py                     # 默认评 data/chunk_test
```

**评分口径（满分 100）：**

| 维度       | 权重 | 口径                                         |
| ---------- | ---- | -------------------------------------------- |
| 完整性     | 25   | 逐块字符统计，丢失越少越高                   |
| 块大小合规 | 20   | 超上限块 + 低于下限碎片块占比                |
| 表格完整性 | 25   | 缺表头表格块 + 表格残片混入正文块            |
| 粒度合理性 | 20   | 正文块落在 400–1000 字理想区间的比例         |
| 结构可追溯 | 10   | 表格块带行号前缀（5）+ 正文块带章节路径（5） |

### 实测得分

| 样本                             | 解析准确率                        | 分块得分 | 备注                            |
| -------------------------------- | --------------------------------- | -------- | ------------------------------- |
| PDF（RNN 情感分类实验）          | 表格 96.13% / 全文字符 F1 98.94%  | 95.2     | `环节` 表最后一行丢失（上游缺陷） |
| 真实 Word（对称密码实验报告）    | 100%                              | 70.0     | 无标题样式、主体是一张大表      |
| 自造 Word（`rag_test_doc.docx`） | 100%                              | 88.0     | 四级标题、长表、图文混排全覆盖  |
| Markdown（RNN 实验 .md）         | 100%                              | 89.1     | 代码密集，提示语单独成块        |
| HTML（`rag_test_doc.html`）      | 100%                              | 100.0    | 代码块 + 章节路径全部正确       |

详细指标见 [量化/指标记录.md](量化/指标记录.md)，踩坑记录见 [ERRORS.md](ERRORS.md)。

---

## 📁 目录结构

```
src/
  api/                FastAPI 接口
    main.py           应用入口与路由
    schemas.py        Pydantic 响应模型
  parsers/            解析器与分块器
    base.py           BaseParser / ParseResult 抽象
    factory.py        解析器工厂（自动选格式 + 单例缓存）
    pdf_parser.py     docling PDF 解析
    word_parser.py    python-docx Word 解析
    markdown_parser.py Markdown 解析
    html_parser.py    HTML 转 markdown
    txt_parser.py     纯文本解析（段落策略）
    table_parser.py   表格合并与转换
    cleaner.py        文本清洗
    chunker.py        分块策略
  llm/                LLM 客户端（OpenAI 兼容）
    client.py         embed / chat / rerank
  vectorstore/        向量库
    qdrant_store.py   Qdrant 封装（本地/服务双模式，HNSW）
  db/                 业务数据库（PostgreSQL）
    database.py       连接池 + 自动建表
    repository.py     documents / chunks CRUD
    schema.sql        建表 SQL
  rag/                RAG 管线
    pipeline.py       ingest_file / retrieve / chat
  config.py           环境变量配置
tests/                各环节测试脚本
量化/                 评测脚本与指标记录
data/                 解析产物 + 本地 Qdrant 数据
test_file/            测试样本
.env.example          环境变量模板
Dockerfile            CPU 版镜像构建
Dockerfile.gpu        GPU 版镜像构建
```

---

## ⚠️ 已知限制

1. **`.doc` 老格式不支持**：python-docx 读不了 OLE 二进制。
   - Windows（装了 Word）：COM 自动化转换
     ```powershell
     $w = New-Object -ComObject Word.Application
     $w.Documents.Open("in.doc").SaveAs2("out.docx", 16)
     $w.Quit()
     ```
   - Linux：LibreOffice headless
     ```bash
     sudo apt install libreoffice-writer fonts-noto-cjk
     soffice --headless --convert-to docx --outdir /tmp/out in.doc
     ```

2. **PDF 解析吃内存**：空闲 < 4GB 版式模型可能 `MemoryError`，建议关 OCR、设 `OPENBLAS_NUM_THREADS=1`。
3. **docling 上游缺陷**：个别表格最后一行会丢（如 `环节` 表）。
4. **标题依赖样式**：Word 文档若没用 Heading 样式，无法生成章节路径。
5. **短块偏多**：表格前引导语必须先落盘，会形成 35–300 字短块，是"表格原子化"的固有代价。

---

## 📝 环境变量

### LLM / Embedding / Rerank（OpenAI 兼容格式）

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `LLM_API_KEY` | — | API 密钥（必填）。若三个服务用不同 key，分别填 `EMBEDDING_API_KEY` / `RERANKER_API_KEY` |
| `LLM_BASE_URL` | — | OpenAI 兼容端点，如 `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `EMBEDDING_BASE_URL` | = LLM_BASE_URL | Embedding 服务端点 |
| `RERANKER_BASE_URL` | = LLM_BASE_URL | Rerank 服务端点 |
| `EMBEDDING_MODEL` | `text-embedding-flash` | 向量化模型（1024 维） |
| `LLM_MODEL` | `qwen3.7-plus` | 问答模型 |
| `RERANKER_MODEL` | `bge-reranker-v2-m3` | 重排模型 |

### 向量数据库（Qdrant）

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `QDRANT_PATH` | `./data/qdrant` | 本地模式路径，设了则用嵌入式 Qdrant（无需启服务） |
| `QDRANT_URL` | `http://localhost:6333` | 服务模式地址（`QDRANT_PATH` 为空时生效） |
| `QDRANT_API_KEY` | — | Qdrant 服务密钥 |
| `QDRANT_COLLECTION` | `rag_chunks` | 集合名 |

### 业务数据库（PostgreSQL）

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `PG_HOST` | `localhost` | 主机 |
| `PG_PORT` | `5432` | 端口 |
| `PG_DB` | `rag` | 数据库名 |
| `PG_USER` | `postgres` | 用户名 |
| `PG_PASSWORD` | — | 密码 |

### RAG 检索参数

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `RETRIEVAL_TOP_K` | `10` | 向量召回数量 |
| `RERANK_TOP_N` | `5` | 重排后保留数量 |

### 其他

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `OPENBLAS_NUM_THREADS` | — | PDF 解析建议设为 `1`，降低内存峰值 |
| `MKL_NUM_THREADS` | — | 同上 |

---

*详细踩坑记录见 [ERRORS.md](ERRORS.md)，量化指标见 [量化/指标记录.md](量化/指标记录.md)。*
