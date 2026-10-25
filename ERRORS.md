# 错误记录

> 规则：本项目每遇到一个错误，都必须在这里新增一条记录，包含「哪里错了」和「如何解决的」两部分。
> 仅记录有排查价值的错误（环境陷阱、模型/框架行为、逻辑缺陷等）。拼写错误、名字不一致、少装依赖这类一眼可见的小问题不记录。
> 状态取值：已解决 / 待修复。

## 目录

| #   | 错误                                         | 状态   |
| --- | -------------------------------------------- | ------ |
| 1   | `OpenBLAS error: Memory allocation failed`   | 已解决 |
| 2   | HuggingFace `ConnectTimeout`                 | 已解决 |
| 3   | 内存提交不足 / 页面文件太小 / CUDA OOM       | 已解决 |
| 4   | a.md 里图片只有 `<!-- image -->` 占位符      | 已解决 |
| 5   | 表格测试用假数据，与 a.md 真实表格对不上     | 已解决 |
| 6   | `markdown_to_dataframe` 把重复表头当数据行   | 已解决 |
| 7   | `merge_tables` 报 `KeyError: 'rows'`         | 已解决 |
| 8   | 合并后的 markdown 丢失分隔行，不再是合法表格 | 已解决 |
| 9   | docling OCR 阶段内存不足，表格丢失甚至崩溃   | 已解决 |
| 10  | docling 漏掉「环节」表最后一行数据           | 待修复 |

---

## 1. OpenBLAS error: Memory allocation still failed

**哪里错了**
OpenBLAS 默认按 CPU 核心数开线程，一次性申请大块内存，在内存紧张的机器上直接崩溃。

**如何解决的**
运行前设置 `OPENBLAS_NUM_THREADS=1`（必要时同时设 `OMP_NUM_THREADS=1`）。

---

## 2. HuggingFace ConnectTimeout

**哪里错了**
docling 首次运行需从 HuggingFace 下载模型，本机直连 `huggingface.co` 与 `hf-mirror.com` 均超时（`WinError 10060`）。

**如何解决的**
启动本地代理（127.0.0.1:7897）并在运行时设置 `HTTP_PROXY` / `HTTPS_PROXY`；再用 `uv run hf download docling-project/docling-layout-heron` 单独预热模型。

---

## 3. 内存提交不足 / 页面文件太小 (os error 1455) / CUDA OOM

**哪里错了**
系统 commit 上限 31.6GB，已提交 30.2GB。加载模型时报 `页面文件太小`；释放部分内存后又变成显存 `out of memory`。

**如何解决的**
关闭占用大户（Chrome 多标签、多余 VS Code 窗口、WSL 等），把提交余量提升到约 9GB 后重跑成功。

---

## 4. a.md 里图片只有 `<!-- image -->` 占位符

**哪里错了**
docling 能检测到图片（本 PDF 5 张），但 `PdfPipelineOptions.generate_picture_images` 默认是 `False`，`export_to_markdown()` 默认 `image_mode=PLACEHOLDER`，所以只输出占位符，没有图片数据。

**如何解决的**
`PDFParser.__init__` 里开启 `generate_picture_images=True`、`images_scale=2.0`，导出时用 `ImageRefMode.EMBEDDED`，图片以 base64 内嵌进 markdown。结果：5 处 `data:image/png;base64`，0 处占位符。

---

## 6. `markdown_to_dataframe` 把重复表头当数据行

**哪里错了**
`table_parser.py` 里 `data_lines = lines[2:]` 写死只跳过第一条分隔线。docling 把跨页表拆成两块时会重复输出表头，导致第二套"表头 + 分隔线"被当成数据行。

**如何解决的**
改为逐行过滤：跳过所有分隔行，以及与表头完全相同的重复表头行。抽出 `_split_row` / `_is_separator` / `_table_cols` / `_table_rows` 复用。实测 a.md 中第 107-115 行那张跨页表不再产生脏行。

---

## 7. `merge_tables` 报 `KeyError: 'rows'`

**哪里错了**
两个模块对"一张表格该有哪些字段"说法不一致，也就是**上游没给的东西，下游却伸手去要**。

上游 `PDFParser` 交过来的表格只有三样：`table_number`（编号）、`page`（页码）、`markdown`（内容）。但下游 `merge_tables` 里写的是 `last['rows'] += current['rows']`，去要 `cols`（列数）和 `rows`（行数）——这两个字段上游从来没产出过，一取值就抛 `KeyError: 'rows'`，整条"解析 → 合并"链路在真实数据上直接崩。

之前的测试发现不了，是因为它用的是手写假字典，恰好带了 `cols`/`rows` 这两个键。

同一处还有第二个隐患：`page` 可能为空，`current.get('page') == last.get('page') + 1` 就变成 `None + 1`，抛 `TypeError`。

**如何解决的**
改成"上游不给，我就自己数"：`cols` 数表头行有几个格子，`rows` 数表头以下的实际数据行（跳过分隔行和重复表头），缺了现算。`page` 为空时退化为只比列数，不做加法。合并时丢掉第二块的表头和紧随其后的分隔行；返回新字典，不修改入参。

---

## 8. 合并后的 markdown 丢失分隔行，不再是合法表格

**哪里错了**
`_data_lines_markdown` 为了去重把分隔行一并删掉，`merge_tables` 拼接时又整行丢弃第二块的表头，导致输出只有表头和数据行，渲染不出表格。

**如何解决的**
规范化表格时保留第一条分隔行、只丢弃重复的；拼接第二块时跳过"表头 + 紧随其后的分隔行"两行。实测 `merged.md` 恢复为 `| 表头 |` + `|---|` + 数据行的合法结构。

---

## 9. docling OCR 阶段内存不足，表格提取不完整甚至进程崩溃

**哪里错了**
内存/提交余量不足（可用物理内存 11.5GB，提交余量仅 7.1GB），docling 在 OCR 阶段抛 `std::bad_alloc` 或 `_ArrayMemoryError`（如 `Unable to allocate 81.9 MiB for an array with shape (736, 4864, 3)`）。失败的那一页表格直接丢失：同一份 PDF 三次运行分别得到 9、5、3 张表。把 `ocr_batch_size` / `layout_batch_size` / `table_batch_size` 从 4 降到 1 仍失败，最后一次以 `0xC0000005` 访问违例崩溃。

**如何解决的**
根因是 `ocr_options.scale` 默认 3.0——OCR 把待识别区域放大 3 倍再送模型，内存按面积增长，这才是 `(736, 4864, 3)` 这类大数组的来源；只降 batch_size 动不到它，所以无效。在 `PDFParser.__init__` 里设 `ocr_options.scale = 1.5`（峰值内存约降至 1/4），同时 `images_scale` 由 2.0 降为 1.0、三个 batch_size 设为 1。重跑后 `table_count = 8`（9 块中跨页的那对已合并），OCR 无任何失败日志。注意：`images_scale` 影响内嵌图片的分辨率，重新生成时图片会比之前小。

补充（做量化评估时发现）：`do_ocr=True` 时即使降了 scale，仍然**时好时坏**——同一天多次运行里出现过解析出 6 张、7 张表的情况，丢表会让后续指标全错。而本 PDF 有文字层，OCR 对抽取结果没有任何贡献（开关两种配置的归一化准确率都是 96.13%）。因此 `do_ocr` 已做成构造参数，对文字版 PDF 建议显式传 `do_ocr=False`，既稳定又更快。

---

## 10. docling 漏掉「环节」表最后一行数据

**哪里错了**
`环节` 表原文是 3 列 × 8 行（表头 + 7 行数据），docling 只解析出 3 列 × 7 行，最后一行「训练与评估 | 任务八、九 | 准确率、模型文件」整体丢失。这是目前唯一的内容级缺陷，会导致数据缺失，其余差异都只是引号样式和分词空格。

定位方式：`量化/eval_tables.py` 逐表逐格比对，`原始 3×8` 对 `解析 3×7`，并提示「少 1 行」。

**如何解决的**
待修复。可能原因：该行位于表格末尾、紧邻分页或页脚，被版面模型判为非表格区域；也可能是最后一行与上一行在版面上贴得过近被合并。

---

## 11. .doc 转出的 docx 里图片一张都提不出来

**哪里错了**
实验报告从老 .doc 格式转成 .docx 后，里面的 8 张截图不是新格式的 `a:blip`（DrawingML），而是老格式的 `v:imagedata`（VML）。解析器只查 `a:blip`，结果一张图都没提到。另外直接用 `元素.xpath(".//a:blip")` 在部分元素上还会报"命名空间前缀没注册"的错。

**如何解决的**
图片查询改成同时找 `blip` 和 `imagedata` 两种标签，写法上用 `local-name()` 按标签名匹配，绕开命名空间前缀注册不一致的坑；图片 id 也要先后试 `r:embed` 和 `r:id` 两个属性。改完 8 张图全部提取成功。

---

## 12. py 源文件混入空字节，整个模块无法导入

**哪里错了**
`pdf_parser.py` 的注释里混进了 2 个看不见的 null 字节（`\x00`），Python 直接拒绝加载，报 `SyntaxError: source code string cannot contain null bytes`。多半是之前某次写入时转义出了问题。

**如何解决的**
用二进制方式读文件，`data.count(b"\x00")` 定位空字节位置，替换回原本的文字后重新写回。教训：文件突然报莫名其妙的 SyntaxError 时，先查有没有 null 字节，别先怀疑代码逻辑。

