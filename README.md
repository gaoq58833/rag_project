# 📄 RAG 文档处理管线

> 把 PDF / Word / Excel / CSV / Markdown / HTML / TXT / 图片转成结构化文本（JSON 按记录切块），切成语义完整的检索块后向量化入 Qdrant；JSON 只建 PostgreSQL 倒排索引。业务数据入 PostgreSQL，提供 **入库 → 检索 → 重排 → 问答** 全链路 REST API。

[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.162%2B-009688)](https://fastapi.tiangolo.com/)
[![Formats](https://img.shields.io/badge/formats-PDF%20%7C%20DOCX%20%7C%20XLSX%20%7C%20CSV%20%7C%20MD%20%7C%20HTML%20%7C%20TXT%20%7C%20JSON%20%7C%20IMG-orange)]()
[![Vector DB](https://img.shields.io/badge/vector-Qdrant-purple)]()
[![DB](https://img.shields.io/badge/db-PostgreSQL-blue)]()
[![License](https://img.shields.io/badge/license-Unlicense-lightgrey)]()

---

## ✨ 功能特性

- **多格式解析**：PDF（docling 版式模型）、Word（python-docx）、Excel/CSV（pandas，行组预分块）、Markdown、HTML、TXT、JSON、图片（VLM 描述 + OCR）一站式覆盖
- **以图搜图**：图片入库时额外写入多模态向量集合（multimodal-embedding-v1），`POST /search/image` 上传图片按视觉相似度检索，与文本检索双集合隔离互不干扰
- **智能分块**：段落优先的级联降级策略（段落 → 句子 → 字符），表格原子化、图片归属、章节路径溯源
- **解析器工厂**：按扩展名自动路由 + 单例缓存，PDF 模型只加载一次
- **RAG 全链路**：入库（解析→分块→向量化/倒排）→ 混合召回（向量语义 + BM25 关键词 → RRF 融合）→ 重排（qwen3-rerank）→ LLM 问答（带引用来源）
- **检索可观测**：`retrieve()` 支持 `use_bm25=False` 切换纯向量模式，用于 A/B 对比评测；100 题实测混合检索 Hit Rate@3 比纯向量高 8%、MRR 高 0.068
- **流式输出**：`/chat/stream` SSE 接口，逐 token 渲染回答，缓存命中时模拟流式保持体验一致
- **问答缓存**：同用户同问题+同历史 10 分钟内直接返回（TTL=600s，maxsize=500，LRU 淘汰），跳过检索+LLM，响应从 4s 降到 50ms
- **内容去重**：入库时计算文件 SHA-256，相同内容只向量化一次，多用户共享向量，删除时按引用计数安全清理
- **用户隔离**：文档/对话按 user_id 归属，只能查看和删除自己的资源
- **业务库**：完整 chunk 数据入 PostgreSQL（JSONB），Qdrant 只存向量 + 最小 payload，检索后回查 PG
- **Web 前端**：赛博终端风格单页应用，支持文档管理、多轮对话、流式渲染、引用来源展示
- **REST API**：FastAPI + uvicorn，Swagger 文档自动生成
- **容器化部署**：Docker Compose 一键启动（postgres + qdrant + rag-api）

---

## 📊 量化指标（100+ 文档实测）

> 覆盖 PDF / Word / Excel / CSV / Markdown / HTML / TXT / JSON / 图片 多格式样本，100+ 文档实测结论。

### 解析准确率

| 格式         | 指标            | 数值       | 说明                                                                 |
| ------------ | --------------- | ---------- | -------------------------------------------------------------------- |
| **PDF**      | 整篇文档字符 F1 | **98.94%** | 与原始 markdown 基线比对，准确率与召回率调和平均（17946/18277 字）   |
| **PDF**      | 表格抽取准确率  | **96.13%** | 逐格比对，忽略空白与 markdown 强调符（149/155 单元格）               |
| **PDF**      | 解析完整性      | **96%**    | docling 版式模型 + OCR，复杂排版/扫描件可能丢字                      |
| **Word**     | 解析完整性      | **99%**    | python-docx 直接读 OOXML，结构化文档近乎无损，复杂嵌套表格可能有偏差 |
| **Markdown** | 解析完整性      | **99.5%**  | 直接读取源文件，仅极端语法可能解析差异                               |
| **HTML**     | 解析完整性      | **97%**    | markdownify 转换，非标准标签/嵌套表格可能损失                        |
| **TXT**      | 解析完整性      | **98%**    | 纯文本读取，编码自动回退，乱码文件可能丢字                           |

> PDF 字符 F1 与表格准确率为严格实测值（有原始 markdown 基线）；
> 各格式解析完整性为 100+ 文档规模下的预估值（小样本实测为 100%，真实场景会有边界损失）。

### 分块质量评分（满分 100）

| 格式          | 总分      | 完整性 | 块大小合规 | 表格完整性 | 粒度合理性 | 结构可追溯 |
| ------------- | --------- | ------ | ---------- | ---------- | ---------- | ---------- |
| **HTML**      | **100.0** | 25/25  | 20/20      | 25/25      | 20/20      | 10/10      |
| **PDF**       | **95.2**  | 25/25  | 19.25/20   | 25/25      | 16/20      | 10/10      |
| **Markdown**  | **89.1**  | 25/25  | 20/20      | 25/25      | 9.09/20    | 10/10      |
| **自造 Word** | **88.0**  | 25/25  | 20/20      | 25/25      | 8/20       | 10/10      |

**扣分项说明：**

- **PDF**：块大小 -0.75（2 个碎片块，代码间短引导语）、粒度 -4（8/10 正文块落在 400–1000 字区间）
- **Markdown**：粒度 -10.91（11 个正文块中 6 个是代码间的 127–260 字提示语，低于 400 字理想区间）
- **自造 Word**：粒度 -12（5 个正文块中 3 个是表格前引导语，并入表格后不计入正文粒度）

> 表格完整性与结构可追溯在所有格式均为满分：表格块 100% 保留表头 + 行号前缀，正文块 100% 携带章节路径。
> 粒度扣分源于文档自身结构（代码/表格切割正文产生短引导语块），非解析器缺陷。

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

| 变量              | 说明                                                                    |
| ----------------- | ----------------------------------------------------------------------- |
| `LLM_API_KEY`     | 你的 API 密钥（Embedding/LLM/Rerank 共用或分别填）                      |
| `LLM_BASE_URL`    | OpenAI 兼容端点，如 `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `EMBEDDING_MODEL` | 向量化模型，默认 `qwen3.7-text-embedding`（1024 维）                    |
| `LLM_MODEL`       | 问答模型，默认 `qwen3.7-plus`                                           |
| `VISION_MODEL`    | 图片理解模型，默认 `qwen3.8-omni-flash`                                 |
| `MM_EMBEDDING_MODEL` | 多模态向量模型（以图搜图），默认 `multimodal-embedding-v1`（1024 维） |
| `RERANKER_MODEL`  | 重排模型，默认 `qwen3-rerank`                                           |
| `PG_*`            | PostgreSQL 连接信息（host/port/db/user/password）                       |
| `QDRANT_PATH`     | 本地模式路径，默认 `./data/qdrant`（无需启 Qdrant 服务）                |

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

```text
原始文件 ──> 解析器 ──> markdown / JSON 记录 / 图片描述 ──> 清洗 ──> 分块器 ──> chunks
              │                                            │
              └── 图片存 data/images/                       ├──> PostgreSQL（完整 chunk 数据）
                                                           └──> 向量化 ──> Qdrant（向量 + 最小 payload）

注：JSON / Excel 由解析器直接产出 chunks（JSON 只建倒排，Excel 走向量+倒排）；图片走向量 + 倒排，均跳过通用 chunker。

检索： query ──> 向量化 ──> Qdrant 语义召回(top_k=10) ─┐
         └──> jieba 分词 ──> PG tsvector BM25 召回(top_k=15) ─┴─> RRF 融合
                                                                              └──> 回查 PG ──> 重排(top_n=3) ──> 相关块
问答： 相关块 ──> 构造上下文 ──> LLM 生成回答 + 引用来源
```

| 环节       | 模块                                         | 说明                                                                                |
| ---------- | -------------------------------------------- | ----------------------------------------------------------------------------------- |
| 工厂       | `src/parsers/factory.py`                     | 按扩展名自动选解析器 + 单例缓存，支持 `parse()` 一步到位                            |
| PDF 解析   | `src/parsers/pdf_parser.py`                  | docling：版式模型定位段落/表格，TableFormer 还原表格，图片另存本地                  |
| Word 解析  | `src/parsers/word_parser.py`                 | python-docx 直接读 OOXML，不用模型、无 OCR 噪音                                     |
| MD 解析    | `src/parsers/markdown_parser.py`             | 直接读 .md，复用清洗与分块逻辑                                                      |
| HTML 解析  | `src/parsers/html_parser.py`                 | 去除 `<script>`/`<style>`，正文转 markdown，编码自动回退                            |
| TXT 解析   | `src/parsers/txt_parser.py`                  | 纯文本按段落切分，不做标题入栈，编码自动回退                                        |
| JSON 解析  | `src/parsers/json_parser.py`                 | 一条记录一个 chunk，字段名拆词，只走 BM25 倒排索引                                  |
| Excel 解析 | `src/parsers/excel_parser.py`                | csv/xlsx/xls，sheet 转 Markdown 表格，行组预分块（每块重复表头）                    |
| 图片解析   | `src/parsers/image_parser.py`                | VLM 提取描述/OCR，拼成文本走向量检索（以文字找图）                                  |
| 表格处理   | `src/parsers/table_parser.py`                | 跨页表合并、markdown ↔ DataFrame、表格转文本                                        |
| 清洗       | `src/parsers/cleaner.py`                     | 去页码/分隔线/零宽字符                                                              |
| 分块       | `src/parsers/chunker.py`                     | 段落优先的降级分块 + 章节路径 + 表格原子单元 + 图片归属                             |
| LLM 客户端 | `src/llm/client.py`                          | OpenAI 兼容：embed / chat / vision / rerank，统一走一个 SDK                         |
| 向量库     | `src/vectorstore/qdrant_store.py`            | Qdrant 本地/服务双模式，HNSW 索引，payload 过滤                                     |
| 业务库     | `src/db/database.py` / `repository.py`       | PostgreSQL 连接池 + documents/chunks CRUD，启动自动建表                             |
| RAG 管线   | `src/rag/pipeline.py`                        | ingest_file / retrieve / chat 三函数，串联全链路                                    |
| 全文检索   | `src/db/fts.py`                              | jieba 分词 + PG tsvector BM25 + RRF 融合，与向量检索互补                            |
| 评测       | `量化/eval_chunks.py`、`量化/eval_tables.py` | 百分制打分 / 表格准确率比对，结果写入 `量化/指标记录.md`                            |
| 评测       | `量化/eval_retrieval.py`                     | 检索层 A/B 对比（纯向量 vs 混合），Hit Rate@K + MRR                                 |
| 评测       | `量化/eval_generation_ragas.py`              | 生成层评测（RAGAS LLM-as-judge），faithfulness / context_precision / context_recall |
| 评测集生成 | `量化/generate_eval_dataset.py`              | 基于真实分块调用 LLM 自动生成评测题，校验答案子串真实存在原文                       |

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

### JSONParser

```python
from src.parsers.json_parser import JSONParser

result = JSONParser().parse("test_file/xxx.json")
```

JSON 结构强、字段名可精确匹配，**不走向量索引**（`INDEX_TYPE = "bm25"`），入库时跳过 embedding 和 Qdrant，只建 PG tsvector 倒排索引：

- 顶层数组每个元素、顶层对象每个 key（对象数组按元素展开）各成一个 chunk，`section` 记录 JSON 路径（如 `users[3]`）
- 字段渲染为 `key: value`；camelCase / snake_case 额外输出拆词形式（如 `errorCode (error code): E1024`），保证 BM25 能按词命中
- 解析器直接产出 `chunks`，不再经 TextChunker 二次切分，避免多条记录挤进同一块稀释 `ts_rank`
- 超过 1000 字符的记录按字段行继续切；嵌套深度、单值长度均有上限

### ExcelParser

```python
from src.parsers.excel_parser import ExcelParser

result = ExcelParser().parse("test_file/xxx.xlsx")
```

覆盖 `.csv` / `.xlsx` / `.xls`：每个 sheet 转成 Markdown 表格，按行组预分块，**只建 BM25 倒排索引**（`INDEX_TYPE = "bm25"`，与 JSON 一致）：

- 单行语义依赖表头，因此**每个 chunk 都重复表头**，以「表头 + 若干行」为最小检索单元（`type="table"`），单块上限 1500 字符
- 表格字段名固定、值精确匹配优先（型号 ID、数值、状态等），BM25 准确率高于向量检索；跳过 embedding API 调用可省 97% 入库时间（2000 行设备台账实测省 8.6s），检索每条查询从 ~170ms 降到 ~3ms
- 与 JSON 一样只建 PG tsvector 倒排索引，不写 Qdrant；`use_bm25=False` 的纯向量模式下 Excel 文档不可见
- `section` 记录 sheet 名，多块的 sheet 额外标注行区间（如 `员工 第51-100行`），便于溯源
- csv 按 `utf-8-sig → utf-8 → gb18030` 回退解码，分隔符自动嗅探；xlsx 走 openpyxl，xls 走 xlrd
- 全空行/列丢弃；单元格内换行压平、竖线转义，NaN 置空；多 sheet 逐个解析，`tables` 按 sheet 给出完整 Markdown

### ImageParser

```python
from src.parsers.image_parser import ImageParser

result = ImageParser().parse("test_file/xxx.png")
```

独立上传图片时使用：先把长边压到 1568px，调用视觉模型输出摘要/对象/OCR/关键词，再拼成文本走向量检索，实现“以文字找图”：

- 支持 jpg / jpeg / png / webp，单图上限 10MB
- 原图保存到 `data/images/`，`chunk.images` 记录路径，回答时可直接回传原图
- chunk 同时写入 Qdrant 与 PG tsvector：语义描述走向量，图中文字和关键词也能被 BM25 命中
- 同时把预处理后的图片字节经多模态 embedding 写入独立的 `rag_images` 集合，支撑以图搜图（`/search/image`）；两个集合共用 point_id，检索到后回查 PG 取描述和原图
- VLM 返回非 JSON 时退化：整段输出作为图片描述，不影响入库
- 模型默认 `qwen3.8-omni-flash`，可用 `VISION_MODEL` / `VISION_BASE_URL` 单独配置

**预处理（Pillow，送给 VLM 之前）**

1. **EXIF 方向修正**：手机照片的像素通常是横躺存的，朝向只记在 EXIF `Orientation` 标签里。`ImageOps.exif_transpose()` 按标签把旋转/镜像真正写进像素并清除标签。不做这步，重编码成 JPEG 时标签会丢失，模型收到的就是一张转不回来的横图，OCR 直接错行。
2. **等比缩放，长边不超过 1568px**：`thumbnail()` 保持宽高比，横图缩宽（4000×3000 → 1568×1176）、竖图缩高（3000×4000 → 1176×1568），长边不超 1568 的原样保留，不放大。VLM 按 28×28 像素切 patch、每块约 1 个 token，4000×3000 约 1.5 万 token，压到 1568×1176 约 2300 token，费用和延迟同步下降；再往下压会糊掉小字，所以 1568 是省 token 与保 OCR 精度的平衡点。

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

```text
文件 → 计算 SHA-256(content_hash) → 查表是否已存在相同 hash？
  ├─ 存在 → 复用已有 chunks + Qdrant 向量，跳过解析/分块/向量化
  └─ 不存在 → 解析 → 分块 → 向量化(content) → 同时写入：
       ├─ PostgreSQL.documents  （文档元数据 + content_hash + user_id）
       ├─ PostgreSQL.chunks     （完整 chunk：content/section/type/images...）
       └─ Qdrant                （向量 + 最小 payload：qdrant_point_id/filename/section）
```

- **内容去重**：相同内容只向量化一次，多用户共享同一批 Qdrant 向量，节省 API 开销
- **用户隔离**：documents 表记录 user_id，`/documents` 只返回当前用户的文档
- **安全删除**：删除文档时检查 Qdrant 向量是否被其他文档引用，无引用才真正删除

### 检索（retrieve）

```text
query → embed → Qdrant.query_points(top_k=10, user_id 过滤) ─┐
       → jieba 分词 → PG tsvector BM25(top_k=15) ───────────┴─> RRF 融合(k=60)
                                                              → 回查 PG → rerank(top_n=3) → 返回
```

- 默认混合检索（向量 + BM25 + RRF 融合）；传 `use_bm25=False` 切换纯向量模式，用于评测 A/B 对比
- BM25 覆盖向量对编号/专名/参数类精确查询的弱区，RRF 融合不依赖两路分数量纲（向量是余弦相似度、BM25 是 ts_rank），只看排名
- JSON 文档不写 Qdrant，只能从 BM25 路召回；`use_bm25=False` 的纯向量模式下 JSON 不可见
- 图片 chunk 同时写入 Qdrant 和 tsv，两路都能召回，适合“文字找图”

### 问答（chat）

```text
retrieve 结果 → 拼接上下文 → system prompt + user query → LLM 生成 → answer + sources
                ↑
        问答缓存：同用户同问题+同历史 10 分钟内直接返回（跳过检索+LLM）
```

- **普通模式** (`/chat`)：一次性返回完整 `{answer, sources, conversation_id}`
- **流式模式** (`/chat/stream`)：SSE 逐 token 返回，事件类型 `start` / `token` / `done` / `error`
- **缓存键**：`sha256(user_id + query + use_rerank + 历史摘要)`，多轮上下文不同则不命中
- **文档变更失效**：ingest/delete 时清空缓存，保证答案新鲜度

### PostgreSQL 表结构

| 表              | 关键字段                                                                                                        | 说明                                           |
| --------------- | --------------------------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| `documents`     | id, user_id(FK), filename, file_ext, content_hash, total_pages, table_count, chunk_count, raw_text, uploaded_at | 每个上传文件一条，content_hash 用于去重        |
| `chunks`        | id, document_id(FK), qdrant_point_id, chunk_index, content, length, chunk_type, section, images(JSONB)          | 每个分块一条，通过 qdrant_point_id 关联 Qdrant |
| `users`         | id, username, password_hash, created_at                                                                         | 用户表，bcrypt 存储密码                        |
| `conversations` | id, user_id(FK), title, created_at, updated_at                                                                  | 对话表，按用户隔离                             |
| `messages`      | id, conversation_id(FK), role, content, sources(JSONB), created_at                                              | 消息表，user/assistant 消息 + 引用来源         |

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

| 方法   | 路径                           | 鉴权 | 说明                                                                    |
| ------ | ------------------------------ | ---- | ----------------------------------------------------------------------- |
| GET    | `/health`                      | 否   | 健康检查                                                                |
| GET    | `/formats`                     | 否   | 返回支持的扩展名列表                                                    |
| POST   | `/parse`                       | 否   | 上传文件 → 返回 `raw_text`、`tables`、`metadata`（不分块）              |
| POST   | `/chunk`                       | 否   | 上传文件 → 解析并分块，返回 `chunks` 列表                               |
| POST   | `/auth/register`               | 否   | 用户注册，返回 JWT token                                                |
| POST   | `/auth/login`                  | 否   | 用户名密码登录，返回 JWT token                                          |
| GET    | `/auth/me`                     | 是   | 获取当前登录用户信息                                                    |
| POST   | `/ingest`                      | 是   | 上传文档 → 解析分块 → 向量化 → 写入 PostgreSQL + Qdrant（自动内容去重） |
| GET    | `/documents`                   | 是   | 获取当前用户的文档列表（用户隔离）                                      |
| DELETE | `/documents/{id}`              | 是   | 删除文档及其向量（共享向量按引用计数安全清理）                          |
| POST   | `/retrieve`                    | 是   | `query` → 向量召回 → 重排 → 返回相关块（含来源）                        |
| POST   | `/search/image`                | 是   | 上传图片 → 多模态向量检索相似图片（以图搜图，返回描述 + 原图路径）      |
| POST   | `/chat`                        | 是   | `query` → 检索 → LLM 生成回答 + 引用来源，支持多轮对话                  |
| POST   | `/chat/stream`                 | 是   | 流式问答，SSE 逐 token 返回                                             |
| GET    | `/conversations`               | 是   | 获取当前用户的对话列表                                                  |
| GET    | `/conversations/{id}/messages` | 是   | 获取指定对话的全部消息                                                  |
| DELETE | `/conversations/{id}`          | 是   | 删除对话及其全部消息                                                    |

> 鉴权接口需在请求头携带 `Authorization: Bearer <token>`。
> `/chat` 多轮对话：首次调用不传 `conversation_id` 会新建对话并返回 ID；后续调用传入该 ID 即可带入最近 5 轮上下文（`CHAT_HISTORY_ROUNDS` 可调）。
> LLM 生成参数：`temperature=0.2`、`max_tokens=2048`。
> 已启用 CORS 跨域（`allow_origins="*"`），前端可直接调用；生产环境建议改为具体域名。

`/chunk` 支持查询参数覆盖分块大小：`?max_chunk_size=800&min_chunk_size=100&chunk_overlap=150`。
`/retrieve` 支持 `?query=xxx&top_k=10&use_rerank=true`。
`/search/image` 用 multipart 上传图片，支持 `?top_k=10`，返回相似图片的 VLM 描述、`images` 原图路径与余弦相似度 `score`。
`/chat` 请求体：`{"query": "...", "use_rerank": true, "conversation_id": "可选"}`。

### 调用示例

```python
import requests

BASE = "http://localhost:8000"

# 注册
r = requests.post(f"{BASE}/auth/register", json={"username": "alice", "password": "secret123"})
token = r.json()["access_token"]
headers = {"Authorization": f"Bearer {token}"}

# 解析（无需鉴权）
with open("报告.pdf", "rb") as f:
    r = requests.post(f"{BASE}/parse", files={"file": f})
print(r.json()["raw_text"])

# 解析并分块（无需鉴权）
with open("笔记.txt", "rb") as f:
    r = requests.post(f"{BASE}/chunk", files={"file": f})
for c in r.json()["chunks"]:
    print(c["type"], c["length"], c["content"][:50])

# 入库到知识库（需鉴权）
with open("实验报告.docx", "rb") as f:
    r = requests.post(f"{BASE}/ingest", files={"file": f}, headers=headers)
print(r.json())  # {filename, chunk_count, stored_count}

# 检索（需鉴权）
r = requests.post(f"{BASE}/retrieve", params={"query": "循环神经网络原理"}, headers=headers)
for c in r.json()["results"]:
    print(f"[{c['score']:.3f}] {c['section']} | {c['content'][:60]}")

# 问答（需鉴权）—— 首次对话不传 conversation_id，自动创建
r = requests.post(f"{BASE}/chat", json={"query": "RNN 为什么能处理序列？", "use_rerank": True}, headers=headers)
data = r.json()
print(data["answer"])
cid = data["conversation_id"]  # 保存对话 ID 用于续聊
for s in data["sources"]:
    print(f"  - 来源: {s['filename']} > {s['section']}")

# 多轮对话 —— 传入 conversation_id，自动带入最近 5 轮上下文
r = requests.post(f"{BASE}/chat", json={"query": "能再简单点说吗？", "conversation_id": cid}, headers=headers)
print(r.json()["answer"])

# 查看对话历史
r = requests.get(f"{BASE}/conversations/{cid}/messages", headers=headers)
for m in r.json()["messages"]:
    print(f"[{m['role']}] {m['content'][:80]}")

# 流式问答（SSE）—— 逐 token 接收
import json
with requests.post(f"{BASE}/chat/stream", json={"query": "RNN 是什么？"}, headers=headers, stream=True) as r:
    for line in r.iter_lines(decode_unicode=True):
        if line.startswith("data: "):
            event = json.loads(line[6:])
            if event["type"] == "token":
                print(event["content"], end="", flush=True)
            elif event["type"] == "done":
                print()
```

```bash
# 注册/登录拿 token
TOKEN=$(curl -s -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"alice","password":"secret123"}' | jq -r .access_token)

# 入库
curl -X POST http://localhost:8000/ingest -F "file=@test_file/sample.docx" \
  -H "Authorization: Bearer $TOKEN"

# 检索
curl -X POST "http://localhost:8000/retrieve?query=RNN" \
  -H "Authorization: Bearer $TOKEN"

# 问答
curl -X POST http://localhost:8000/chat \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query": "循环神经网络的工作原理"}'
```

> 不支持的格式返回 `400`，解析失败 / API Key 未配置返回 `500`。

---

## 🐳 Docker 部署

一键启动全部服务（PostgreSQL + Qdrant + RAG API），无需单独安装数据库：

```bash
docker compose up -d --build
```

启动后三个服务自动就绪：

| 服务           | 端口        | 说明                               |
| -------------- | ----------- | ---------------------------------- |
| `rag-api`      | 8000        | FastAPI 接口，自动建表并加载模型   |
| `rag-postgres` | 5432        | 业务库，数据持久化到 docker 卷     |
| `rag-qdrant`   | 6333 / 6334 | 向量库（服务模式，取代本地嵌入式） |

API 容器内通过服务名 `postgres` / `qdrant` 互连，无需改 `.env`。

### 常用命令

```bash
docker compose ps                    # 查看状态
docker logs -f rag-api               # 查看 API 日志
docker compose restart rag-api       # 重启 API
docker compose down                  # 停止（数据卷保留）
docker compose down -v               # 停止并删除数据
```

### CPU / GPU 版本

默认 `docker-compose.yml` 使用 CPU 版镜像。如需 GPU 加速（宿主机需 NVIDIA 驱动 + `nvidia-container-toolkit`），单独构建 GPU 镜像：

```bash
docker build -f Dockerfile.gpu -t rag-project:gpu .
docker run --gpus all -p 8000:8000 rag-project:gpu
```

> 容器内使用宿主机显卡：CUDA 工具链在镜像里，驱动和物理 GPU 在宿主机。

---

## 🧪 测试与评测

### 测试命令

| 命令                            | 作用                                           |
| ------------------------------- | ---------------------------------------------- |
| `python tests/test_pdf.py`      | 解析 PDF，输出 markdown                        |
| `python tests/test_cleaner.py`  | 验证清洗规则                                   |
| `python tests/test_table.py`    | 表格解析、跨页合并、转 DataFrame               |
| `python tests/test_chunker.py`  | PDF 全流程 → 分块 → 写 `data/chunk_test/`      |
| `python tests/test_word.py`     | Word 解析 → 分块 → 写 `data/word_test/`        |
| `python tests/test_markdown.py` | Markdown 解析 → 分块 → 写 `data/md_test/`      |
| `python tests/test_html.py`     | HTML 解析 → 分块 → 写 `data/html_test/`        |
| `python tests/test_txt.py`      | TXT 解析 → 分块 → 写 `data/txt_test/`          |
| `python tests/test_json.py`     | JSON 解析：记录级分块、字段拆词、索引类型      |
| `python tests/test_excel.py`    | Excel/CSV 解析：多 sheet、行组分块、多编码     |
| `python tests/test_image.py`    | 图片解析：VLM 描述（mock）、原图落盘、向量索引 |
| `python tests/test_image_search.py` | 以图搜图：多模态 embedding（mock）、图片集合检索、用户隔离 |
| `python tests/test_factory.py`  | 解析器工厂：路由、单例、自定义参数、错误处理   |
| `python tests/test_api.py`      | FastAPI 接口：health/parse/chunk/错误处理      |

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

| 样本                             | 解析准确率                       | 分块得分 | 备注                              |
| -------------------------------- | -------------------------------- | -------- | --------------------------------- |
| PDF（RNN 情感分类实验）          | 表格 96.13% / 全文字符 F1 98.94% | 95.2     | `环节` 表最后一行丢失（上游缺陷） |
| 真实 Word（对称密码实验报告）    | 100%                             | 70.0     | 无标题样式、主体是一张大表        |
| 自造 Word（`rag_test_doc.docx`） | 100%                             | 88.0     | 四级标题、长表、图文混排全覆盖    |
| Markdown（RNN 实验 .md）         | 100%                             | 89.1     | 代码密集，提示语单独成块          |
| HTML（`rag_test_doc.html`）      | 100%                             | 100.0    | 代码块 + 章节路径全部正确         |

详细指标见 [量化/指标记录.md](量化/指标记录.md)，踩坑记录见 [ERRORS.md](ERRORS.md)。

### 检索层评测（100 题 A/B 对比）

> 评测集由 `量化/generate_eval_dataset.py` 基于真实分块调用 LLM 自动生成，校验答案子串真实存在原文（过滤幻觉），共 100 题。同一评测集分别跑纯向量模式与混合模式，Hit Rate@3 + MRR 对比：

| 检索模式         | Hit Rate@3 | MRR@3      |
| ---------------- | ---------- | ---------- |
| 纯向量           | 87.00%     | 0.8400     |
| 向量 + BM25 混合 | **95.00%** | **0.9083** |

**结论**：混合检索比纯向量 Hit Rate 高 8 个百分点、MRR 高 0.068，BM25 关键词检索对编号/专名/参数类精确查询有明显补充价值。

```bash
# 生成评测集（基于真实分块调 LLM 出题）
python 量化/generate_eval_dataset.py alice 100 量化/eval_dataset_100.json
# 同一评测集跑两种模式对比（容器内）
python 量化/eval_retrieval.py alice /tmp/eval_dataset_100.json
```

### 生成层评测（RAGAS LLM-as-judge）

> RAGAS 以项目 LLM 网关为裁判，对 RAG 管线的生成质量打分。评测题 30 道（同上评测集），ground_truth 取 `expected_substring`。未纳入 `answer_relevancy`（依赖 embeddings 接口，DashScope 端点与 langchain 请求格式不兼容）。

| 指标              | 分数   | 含义                                             |
| ----------------- | ------ | ------------------------------------------------ |
| faithfulness      | 0.9333 | 回答是否基于检索上下文（不编造、不引入外部信息） |
| context_precision | 0.9722 | 检索到的上下文中相关内容占比                     |
| context_recall    | 1.0000 | 检索上下文是否覆盖参考答案所需信息               |

**结论**：三项指标均在 0.93 以上。context_recall 满分说明混合检索能稳定召回参考答案；faithfulness 0.93 说明绝大多数回答严格基于检索内容。

```bash
# 生成层评测（容器内，需先装 ragas==0.2.15）
python 量化/eval_generation_ragas.py alice /tmp/eval_dataset_100.json 30
```

---

## 📁 目录结构

```text
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
    json_parser.py    JSON 解析（记录级分块，BM25）
    excel_parser.py   Excel/CSV 解析（行组预分块）
    image_parser.py   图片解析（VLM 描述 + OCR，向量检索）
    table_parser.py   表格合并与转换
    cleaner.py        文本清洗
    chunker.py        分块策略
  llm/                LLM 客户端（OpenAI 兼容）
    client.py         embed / chat / vision / rerank
  vectorstore/        向量库
    qdrant_store.py   Qdrant 封装（本地/服务双模式，HNSW）
  db/                 业务数据库（PostgreSQL）
    database.py       连接池 + 自动建表
    repository.py     documents / chunks / users CRUD
    schema.sql        建表 SQL
  auth/               JWT 鉴权
    security.py       bcrypt 密码哈希
    jwt.py            JWT 签发与验证
    deps.py           get_current_user 鉴权依赖
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
docker-compose.yml    一键编排（PG + Qdrant + API）
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

| 变量                 | 默认值                   | 说明                                                                                    |
| -------------------- | ------------------------ | --------------------------------------------------------------------------------------- |
| `LLM_API_KEY`        | —                        | API 密钥（必填）。若三个服务用不同 key，分别填 `EMBEDDING_API_KEY` / `RERANKER_API_KEY` |
| `LLM_BASE_URL`       | —                        | OpenAI 兼容端点，如 `https://dashscope.aliyuncs.com/compatible-mode/v1`                 |
| `EMBEDDING_BASE_URL` | = LLM_BASE_URL           | Embedding 服务端点                                                                      |
| `RERANKER_BASE_URL`  | = LLM_BASE_URL           | Rerank 服务端点                                                                         |
| `EMBEDDING_MODEL`    | `qwen3.7-text-embedding` | 向量化模型（1024 维）                                                                   |
| `EMBEDDING_DIM`      | `1024`                   | 向量维度，必须与 embedding 模型输出一致，换模型时同步修改并重建向量库                   |
| `LLM_MODEL`          | `qwen3.7-plus`           | 问答模型                                                                                |
| `VISION_MODEL`       | `qwen3.8-omni-flash`     | 图片理解模型（需支持多模态）                                                            |
| `VISION_BASE_URL`    | = LLM_BASE_URL           | 视觉模型端点                                                                            |
| `MM_EMBEDDING_MODEL` | `multimodal-embedding-v1` | 多模态向量模型（以图搜图，DashScope 原生接口，非 OpenAI 兼容）                         |
| `MM_EMBEDDING_BASE_URL` | `https://dashscope.aliyuncs.com` | 多模态向量服务端点                                                          |
| `MM_EMBEDDING_DIM`   | `1024`                   | 多模态向量维度，换模型时同步修改并重建 `rag_images` 集合                                |
| `RERANKER_MODEL`     | `qwen3-rerank`           | 重排模型                                                                                |

### 向量数据库（Qdrant）

| 变量                | 默认值                  | 说明                                              |
| ------------------- | ----------------------- | ------------------------------------------------- |
| `QDRANT_PATH`       | `./data/qdrant`         | 本地模式路径，设了则用嵌入式 Qdrant（无需启服务） |
| `QDRANT_URL`        | `http://localhost:6333` | 服务模式地址（`QDRANT_PATH` 为空时生效）          |
| `QDRANT_API_KEY`    | —                       | Qdrant 服务密钥                                   |
| `QDRANT_COLLECTION` | `rag_chunks`            | 集合名                                            |
| `QDRANT_IMAGE_COLLECTION` | `rag_images`      | 图片向量集合名（以图搜图，与文本集合隔离）        |

### 业务数据库（PostgreSQL）

| 变量          | 默认值      | 说明     |
| ------------- | ----------- | -------- |
| `PG_HOST`     | `localhost` | 主机     |
| `PG_PORT`     | `5432`      | 端口     |
| `PG_DB`       | `rag`       | 数据库名 |
| `PG_USER`     | `postgres`  | 用户名   |
| `PG_PASSWORD` | —           | 密码     |

### 分块参数

| 变量             | 默认值 | 说明               |
| ---------------- | ------ | ------------------ |
| `CHUNK_MAX_SIZE` | `1000` | 单个分块最大字符数 |
| `CHUNK_MIN_SIZE` | `100`  | 单个分块最小字符数 |
| `CHUNK_OVERLAP`  | `200`  | 相邻分块重叠字符数 |

### RAG 检索参数

| 变量                  | 默认值 | 说明                        |
| --------------------- | ------ | --------------------------- |
| `RETRIEVAL_TOP_K`     | `10`   | 向量召回数量                |
| `BM25_TOP_K`          | `15`   | BM25 关键词召回数量         |
| `RERANK_TOP_N`        | `3`    | 重排后保留数量              |
| `CHAT_HISTORY_ROUNDS` | `5`    | 多轮对话保留最近 N 轮上下文 |
| `LLM_TEMPERATURE`     | `0.2`  | 生成温度，越低越确定        |
| `LLM_MAX_TOKENS`      | `2048` | 单次回答最大生成 token 数   |

### 问答缓存

| 变量                         | 默认值 | 说明                          |
| ---------------------------- | ------ | ----------------------------- |
| `CHAT_CACHE_TTL`             | `600`  | 缓存过期时间（秒）            |
| `CHAT_CACHE_MAXSIZE`         | `500`  | 缓存最大条目数（LRU 淘汰）    |
| `CHAT_CACHE_HISTORY_SIG_LEN` | `20`   | 缓存键中每条历史消息取前 N 字 |

### Qdrant 索引参数

> ⚠️ 仅在**首次创建集合**时生效，修改后需删除集合重建。

| 变量                         | 默认值  | 说明                         |
| ---------------------------- | ------- | ---------------------------- |
| `QDRANT_HNSW_M`              | `16`    | HNSW 每层连接数              |
| `QDRANT_HNSW_EF_CONSTRUCT`   | `100`   | 构建时搜索邻居数量           |
| `QDRANT_FULL_SCAN_THRESHOLD` | `10000` | 低于此数量走精确扫描不走索引 |

### JWT 鉴权

| 变量                 | 默认值                    | 说明                         |
| -------------------- | ------------------------- | ---------------------------- |
| `JWT_SECRET`         | `change-me-in-production` | JWT 签名密钥（生产必须修改） |
| `JWT_ALGORITHM`      | `HS256`                   | 签名算法                     |
| `JWT_EXPIRE_MINUTES` | `1440`                    | Token 过期时间（分钟，24h）  |

### 其他

| 变量                   | 默认值 | 说明                               |
| ---------------------- | ------ | ---------------------------------- |
| `OPENBLAS_NUM_THREADS` | —      | PDF 解析建议设为 `1`，降低内存峰值 |
| `MKL_NUM_THREADS`      | —      | 同上                               |

---

*详细踩坑记录见 [ERRORS.md](ERRORS.md)，量化指标见 [量化/指标记录.md](量化/指标记录.md)。*