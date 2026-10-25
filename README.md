# RAG 文档处理管线

面向 RAG 的文档解析与分块工具：把 PDF / Word 转成带结构的 markdown，再切成语义完整、可溯源、可返回原图的块，并配套一套百分制量化评测。

当前覆盖 **PDF**、**Word（.docx）**、**Markdown**、**HTML**、**TXT** 五类文档。

---

## 1. 处理流程

```
原始文件 ──> 解析器 ──> markdown ──> 清洗 ──> 分块器 ──> chunks.json
              │                                            │
              └── 图片存 data/images/，正文只留路径引用       └── 量化评分
```

| 环节      | 模块                                         | 说明                                                               |
| --------- | -------------------------------------------- | ------------------------------------------------------------------ |
| 工厂      | `src/parsers/factory.py`                     | 按扩展名自动选解析器 + 单例缓存，支持 `parse()` 一步到位            |
| PDF 解析  | `src/parsers/pdf_parser.py`                  | docling：版式模型定位段落/表格，TableFormer 还原表格，图片另存本地 |
| Word 解析 | `src/parsers/word_parser.py`                 | python-docx 直接读 OOXML，不用模型、无 OCR 噪音                    |
| MD 解析   | `src/parsers/markdown_parser.py`             | 直接读 .md，复用清洗与分块逻辑                                     |
| HTML 解析 | `src/parsers/html_parser.py`                 | 去除 <script>/<style>，正文转 markdown，编码自动回退               |
| TXT 解析  | `src/parsers/txt_parser.py`                  | 纯文本按段落切分，不做标题入栈，编码自动回退                       |
| 表格处理  | `src/parsers/table_parser.py`                | 跨页表合并、markdown ↔ DataFrame、表格转文本                       |
| 清洗      | `src/parsers/cleaner.py`                     | 去页码/分隔线/零宽字符                                             |
| 分块      | `src/parsers/chunker.py`                     | 段落优先的降级分块 + 章节路径 + 表格原子单元 + 图片归属            |
| 评测      | `量化/eval_chunks.py`、`量化/eval_tables.py` | 百分制打分 / 表格准确率比对，结果写入 `量化/指标记录.md`           |

---

## 2. 环境与安装

- Python ≥ 3.12（推荐用 [uv](https://docs.astral.sh/uv/) 管理）
- Windows / Linux 均可；PDF 解析建议 ≥ 8GB 空闲内存

```bash
uv sync                 # 安装依赖
uv run python tests/test_word.py
```

或者用已有的虚拟环境：

```bash
.venv\Scripts\python.exe tests\test_word.py      # Windows
.venv/bin/python tests/test_word.py              # Linux
```

### PDF 解析的环境变量

docling 自带布局模型和表格模型，在低内存机器上容易崩，建议固定线程数：

```bash
set OPENBLAS_NUM_THREADS=1        # Windows
export OPENBLAS_NUM_THREADS=1     # Linux
```

需要下载模型权重时还要配代理（模型首次使用会自动下载）。

---

## 3. 核心模块

### ParserFactory（推荐入口）

无需手动判断格式，工厂根据扩展名自动选择解析器，并复用单例实例（PDFParser 的 docling 模型只加载一次）。

```python
from src.parsers.factory import get_parser, parse

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

### 3.1 PDFParser

```python
from src.parsers.pdf_parser import PDFParser

parser = PDFParser(do_ocr=False)          # 文字版 PDF 建议关 OCR：更稳、更快、表格不丢
result = parser.parse("test_file/xxx.pdf")

result.raw_text    # markdown，图片位置是 ![图片1](data/images/xxx_1.png)
result.tables      # 表格列表（已合并跨页表）
result.total_pages # 页数
```

关键参数：

| 参数           | 默认          | 说明                                           |
| -------------- | ------------- | ---------------------------------------------- |
| `do_ocr`       | `True`        | 文字版 PDF 建议 `False`；开启 OCR 会随机丢表格 |
| `ocr_scale`    | `1.5`         | OCR 区域放大倍数，默认 3.0 内存占用过大        |
| `images_scale` | `1.0`         | 导出图片分辨率                                 |
| `image_dir`    | `data/images` | 图片落盘目录                                   |

### 3.2 WordParser

```python
from src.parsers.word_parser import WordParser

result = WordParser().parse("test_file/xxx.docx")
```

- 只支持 `.docx`，`.doc` 老二进制格式需先转换（见第 6 节）
- 标题样式 `Heading 1-6` / `标题 1-6` 会映射成对应数量的 `#`（最多 6 个）
- 表格在文档中的真实顺序会被保留（遍历 body 子节点，而不是 `doc.paragraphs` + `doc.tables`）
- 图片同时兼容新格式 `a:blip` 和老式 VML `v:imagedata`，**表格单元格里的图片也会提取**
- Word 没有页的概念，`total_pages` 固定为 0

### 3.3 MarkdownParser

```python
from src.parsers.markdown_parser import MarkdownParser

result = MarkdownParser().parse("test_file/xxx.md")
```

- 直接读取 `.md` / `.markdown`，不做格式转换
- 表格（`|...|` 行）会被抽取并复用 `TableProcessor` 规范化
- 标题沿用 markdown 原生 `#` 语法，分块时可正常生成章节路径

### 3.4 HTMLParser

```python
from src.parsers.html_parser import HTMLParser

result = HTMLParser().parse("test_file/xxx.html")
```

- 去除 `<script>`、`<style>`、注释等噪音，正文转 markdown
- 编码不固定时按 `utf-8 → utf-8-sig → gb18030` 顺序回退解码
- 表格、列表、标题等结构尽量保留，代码块用围栏包裹

### 3.5 TXTParser

```python
from src.parsers.txt_parser import TXTParser

result = TXTParser().parse("test_file/xxx.txt")
```

- TXT 结构化弱，**不做标题入栈**，分块走段落策略（空行分段 → 句子 → 字符）
- 编码按 `utf-8 → utf-8-sig → gb18030` 回退，兼容常见中文文本编码
- 无表格/图片，`table_count` 为 0，`section` 字段留空

### 3.6 TextChunker

```python
from src.parsers.chunker import TextChunker, ChunkStrategy

chunker = TextChunker(max_chunk_size=1000, min_chunk_size=100, chunk_overlap=200)
chunks = chunker.split(markdown_text)
```

分块策略（逐级降级）：

1. **段落优先**：按空行切段，逐段累加，超过 `max_chunk_size` 就输出当前块。
2. **段落超长 → 句子级**：按 `。！？；` 等句末标点切句再重新组装，此阶段保留 `chunk_overlap` 缓冲。
3. **单句仍超长 → 字符级**：按固定长度硬切，同样保留重叠，残余窗口并入上一块，保证不丢字。
4. **段落之间不做重叠**：相邻两段主题可能无关，重叠只会污染检索结果。

其他规则：

- **章节路径**：从标题栈实时推导，如 `三、双向 LSTM > 3.2 取最后隐状态`；一个块跨多个章节时用 `；` 并列标注。
- **表格是原子单元**：整表放得下就一整块；超过上限按行窗口切，每个子块都重复表头并带 `【表格 · 第x-y行/共N行】` 前缀。
- **图片不单独成块**：引用行并入当前块，路径记入该块的 `images` 字段。若图片紧跟在已输出的块（典型是表格）之后，则只把路径追加到上一个块的 `images`，不改动它的正文——插在表格单元格里的截图就属于这种情况；文末遗留图片挂到最后一块。

块的数据结构：

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

## 4. HTTP API（FastAPI）

基于 FastAPI + uvicorn，文件上传用 python-multipart。

### 启动

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
# 或
python -m src.api.main
```

启动时会预热 PDFParser（加载 docling 模型），首次请求不再卡顿。API 文档自动生成在 `/docs`（Swagger UI）和 `/redoc`。

### 接口

| 方法 | 路径 | 说明 |
| ---- | ---- | ---- |
| `GET`  | `/health` | 健康检查 |
| `GET`  | `/formats` | 返回支持的扩展名列表 |
| `POST` | `/parse` | 上传文件 → 返回 `raw_text`、`tables`、`metadata`（不分块） |
| `POST` | `/chunk` | 上传文件 → 解析并分块，返回 `chunks` 列表 |

`/chunk` 支持查询参数覆盖默认分块大小：`?max_chunk_size=800&min_chunk_size=100&chunk_overlap=150`。

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
```

```bash
curl -X POST http://localhost:8000/chunk -F "file=@test_file/sample.txt"
```

错误处理：不支持的格式返回 `400`，解析失败返回 `500`。

---

## 5. 测试与评测

### 5.1 跑测试

| 命令                                          | 作用                                             |
| --------------------------------------------- | ------------------------------------------------ |
| `python tests/test_pdf.py`                    | 解析 PDF，输出 markdown                          |
| `python tests/test_cleaner.py`                | 验证清洗规则                                     |
| `python tests/test_table.py`                  | 表格解析、跨页合并、转 DataFrame                 |
| `python tests/test_chunker.py`                | PDF 全流程 → 分块 → 写 `data/chunk_test/`        |
| `python tests/test_word.py`                   | Word 解析 → 分块 → 写 `data/word_test/`          |
| `python tests/test_word.py <docx> <输出目录>` | 测指定文档                                       |
| `python tests/test_markdown.py`               | Markdown 解析 → 分块 → 写 `data/md_test/`        |
| `python tests/test_html.py`                   | HTML 解析 → 分块 → 写 `data/html_test/`          |
| `python tests/test_txt.py`                    | TXT 解析 → 分块 → 写 `data/txt_test/`            |
| `python tests/test_factory.py`                | 解析器工厂：路由、单例、自定义参数、错误处理      |
| `python tests/test_api.py`                    | FastAPI 接口：health/parse/chunk/错误处理        |
| `python tests/make_test_docx.py`              | 生成结构化测试文档 `test_file/rag_test_doc.docx` |

`test_*.py` 都会同时报告：解析覆盖率、块数、超限/碎片块、丢失字符、重复字符、章节路径覆盖、图片归属。

### 5.2 打分

```bash
# 参数：数据目录 [章节号] [章节标题]
python 量化/eval_chunks.py data/word_test_gen 5 "自造 Word 测试文档"
python 量化/eval_chunks.py                     # 默认评 data/chunk_test，写第 3 节
```

评分口径（满分 100）：

| 维度       | 权重 | 口径                                         |
| ---------- | ---- | -------------------------------------------- |
| 完整性     | 25   | 逐块字符统计，丢失越少越高                   |
| 块大小合规 | 20   | 超上限的块 + 低于下限的碎片块占比            |
| 表格完整性 | 25   | 缺表头的表格块 + 表格残片混入正文块          |
| 粒度合理性 | 20   | 正文块落在 400-1000 字理想区间的比例         |
| 结构可追溯 | 10   | 表格块带行号前缀（5）+ 正文块带章节路径（5） |

样本特有的客观性说明放在数据目录的 `notes.md` 里，重跑评测时自动带上。

### 5.3 当前实测

| 样本                             | 解析准确率                       | 分块得分 | 备注                            |
| -------------------------------- | -------------------------------- | -------- | ------------------------------- |
| PDF（RNN 情感分类实验）          | 表格 96.13% / 全文字符 F1 98.94% | 95.2     | 已知缺陷：`环节` 表最后一行丢失 |
| 真实 Word（对称密码实验报告）    | 100%                             | 70.0     | 无标题样式、主体是一张大表      |
| 自造 Word（`rag_test_doc.docx`） | 100%                             | 88.0     | 四级标题、长表、图文混排全覆盖  |
| Markdown（RNN 实验 .md）         | 100%                             | 89.1     | 代码密集，提示语单独成块        |
| HTML（`rag_test_doc.html`）      | 100%                             | 100.0    | 代码块 + 章节路径全部正确       |

详细指标见 [量化/指标记录.md](量化/指标记录.md)，踩坑记录见 [ERRORS.md](ERRORS.md)。

---

## 6. 目录结构

```
src/
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
  api/                FastAPI 接口
    main.py           应用入口与路由
    schemas.py        Pydantic 响应模型
tests/                各环节测试脚本
量化/                 评测脚本与指标记录
data/
  images/             解析出的图片
  chunk_test/         PDF 分块结果
  word_test/          真实 Word 分块结果
  word_test_gen/      自造 Word 分块结果
  md_test/            Markdown 分块结果
  html_test/          HTML 分块结果
  txt_test/           TXT 分块结果
test_file/            测试样本
```

---

## 7. 已知限制

1. **`.doc` 老格式不支持**：python-docx 读不了 OLE 二进制。转换方案：
   - Windows：Word COM 自动化（本机已装 Word）
     ```powershell
     $w = New-Object -ComObject Word.Application
     $w.Documents.Open("in.doc").SaveAs2("out.docx", 16)
     $w.Quit()
     ```
     保真度最高，但转换出的图片可能仍是 VML 格式（解析器已兼容）。
   - Linux 推荐 LibreOffice headless：
     ```bash
     sudo apt install libreoffice-writer fonts-noto-cjk
     soffice --headless --convert-to docx --outdir /tmp/out in.doc
     ```
     记得装中文字体，首次启动可加 `-env:UserInstallation=file:///tmp/lo_profile` 加速并避免并发锁。
2. **PDF 解析吃内存**：低内存环境（空闲 < 4GB）版式模型可能 `MemoryError`，建议关 OCR、降 batch、限制 `OPENBLAS_NUM_THREADS=1`。
3. **docling 结构性缺陷**：个别表格最后一行会丢（如 `环节` 表），属上游模型问题。
4. **标题依赖样式**：Word 文档若没用 Heading 样式，就无法生成章节路径，可追溯性只能靠正文内容本身。
5. **短块偏多**：表格前的引导语必须先落盘，会形成 35-300 字的短块，这是"表格原子化"的固有代价。
6. **重复内容**：长表按行切分会保留上一行（约一行字符），超长段落句级切分保留 `chunk_overlap`，均为设计内行为。
