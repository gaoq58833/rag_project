import re
from typing import Dict, List, Optional

# 章节切换最小切分长度：上一章内容不足这个长度时与下一章合并成一块，避免切出大量碎块
SECTION_SPLIT_MIN = 400

class ChunkStrategy:
    """分块策略枚举"""
    PARAGRAPH = "paragraph"   # 按段落分块
    SENTENCE = "sentence"     # 按句子分块
    CHARACTER = "character"   # 按字符数分块
    SEMANTIC = "semantic"     # 语义分块（阶段三，使用模型）

class TextChunker:
    """文本分块器"""

    def __init__(
        self,
        strategy: str = ChunkStrategy.PARAGRAPH,
        max_chunk_size: int = 1000,
        min_chunk_size: int = 100,
        chunk_overlap: int = 200,
    ):
        """
        初始化分块器

        Args:
            strategy: 分块策略
            max_chunk_size: 最大块大小（字符数）
            min_chunk_size: 最小块大小
            chunk_overlap: 块之间重叠字符数
        """
        self.strategy = strategy
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    def split(self, text: str, tables: Optional[List[Dict]] = None) -> List[Dict]:
        '''
        通过策略对文本进行分块
        '''
        if self.strategy == ChunkStrategy.PARAGRAPH:
            chunks = self._split_by_paragraph(text)
        elif self.strategy == ChunkStrategy.SENTENCE:
            chunks = self._split_by_sentence(text)
        elif self.strategy == ChunkStrategy.CHARACTER:
            chunks = self._split_by_character(text)
        else:
            chunks = self._split_by_paragraph(text)
        
        # 添加表格信息
        if tables:
            chunks = self._attach_tables(chunks, tables)

        return chunks

    def _split_by_paragraph(self, text: str) -> List[Dict]:
        '''
        按段落分块，表格作为独立块处理
            1. 先按空行切出段落，并把连续的 "|" 行识别为表格单元，不让它被拆开。
            2. 普通段落逐段累加到 current_chunk；超过 max_chunk_size 时先输出当前块，
               新段落直接另起一块（不做块间重叠，避免带入和本段无关的上一块内容）。
            3. 表格单元单独成块：整表放得下就是一整块，放不下就按行窗口切，
               每个子块都重复表头并带「第 x-y 行 / 共 N 行」前缀。
            4. 图片不单独成块：引用行并入当前文本块，路径记入该块的 images 字段，
               检索命中后按路径取原图展示。
        '''
        chunks = []
        current_chunk = ""
        chunk_index = 0
        section = ""          # 当前所属章节路径，如「三、双向 LSTM > 3.2 取最后隐状态」
        stack = []            # 标题栈：[(层级, 标题)]
        span = []             # 当前块覆盖到的章节路径，块跨章节时会有多条
        pending_images = []   # 已遇到但还没归属到块的图片路径

        def label():
          # 块内跨章节时，把所有章节路径都标出来
          return "；".join(span) if span else section

        def emit(text: str):
          nonlocal chunk_index, span, pending_images
          chunk = {
            "content": text.strip(),
            "index": chunk_index,
            "length": len(text),
            "type": "text",
            "section": label()
          }
          if pending_images:
            # 图片只记位置：块里带上图片路径，回答时按路径把原图贴给用户
            chunk["images"] = pending_images
            pending_images = []
          chunks.append(chunk)
          chunk_index += 1
          span = []

        def flush():
          # 收尾当前块：达到最小块大小才输出；不足的残块留在 current_chunk，留给下一块拼接，避免丢字
          nonlocal current_chunk
          if len(current_chunk) > self.min_chunk_size:
            emit(current_chunk)
            current_chunk = ""

        for kind, body in self._split_blocks(text):
            # 图片：引用行原样并入当前块内容，路径记入 pending_images，随下一次 emit 归属到块
            if kind == "image":
                m = re.match(r'^\s*(!\[.*?\]\((.*?)\))\s*$', body)
                if not current_chunk and chunks:
                    # 紧跟在已输出块（如表格）之后的图片属于上一个块：
                    # 典型场景是截图插在表格单元格里，解析后引用行排在表格后面。
                    # 只记 images 字段、不动 content：路径引用不是检索内容，避免把块撑超长
                    chunks[-1].setdefault("images", []).append(m.group(2))
                    continue
                current_chunk = current_chunk + "\n\n" + m.group(1) if current_chunk else m.group(1)
                pending_images.append(m.group(2))
                continue

            # 表格：先收尾当前文本块，再把表格按行窗口输出
            if kind == "table":
                if current_chunk.strip():
                    emit(current_chunk)   # 表格前的正文先落盘，避免表格切分打乱原文顺序
                    current_chunk = ""
                for content in self._split_table(body):
                    chunks.append({
                        "content": content,
                        "index": chunk_index,
                        "length": len(content),
                        "type": "table",
                        "section": section
                    })
                    chunk_index += 1
                continue

            para = body.strip()
            if not para:
                continue

            # 标题行：更新标题栈；上一章够长就先收尾，然后统一走下面的长度判断
            heading = re.match(r'^(#{1,6})\s+(.+?)\s*$', para.split("\n")[0])
            if heading:
                # 章节切换：上一章内容够长才单独成块，太短就继续累加，避免切出一堆碎块
                if len(current_chunk) >= SECTION_SPLIT_MIN:
                    emit(current_chunk)
                    current_chunk = ""
                level, title = len(heading.group(1)), heading.group(2)
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, title))
                section = " > ".join(t for _, t in stack)
                if not span or span[-1] != section:
                    span.append(section)

            # 当前块 + 新块 > 最大块大小
            if len(current_chunk) + len(para) > self.max_chunk_size:
                if len(current_chunk) > self.min_chunk_size:
                    flush()
                    # 处理新块逻辑
                    if len(para) > self.max_chunk_size:
                        # 如果新块长度超过最大块长度,需要进行块的分割
                        sub_chunks = self._split_long_paragraph(para, chunk_index, label())
                        # 将子块添加到结果中
                        chunks.extend(sub_chunks)
                        chunk_index += len(sub_chunks)
                        current_chunk = ""
                    else:
                        # 不做块间重叠：上一块与这一段内容不一定连贯，重叠只会带入无关内容
                        current_chunk = para
                else:
                    # 当前块太短，先并入本段一起判超限，避免残块被丢弃
                    merged = current_chunk + "\n\n" + para
                    current_chunk = ""
                    if len(merged) > self.max_chunk_size:
                        sub_chunks = self._split_long_paragraph(merged, chunk_index, label())
                        chunks.extend(sub_chunks)
                        chunk_index += len(sub_chunks)
                        span = []
                    else:
                        current_chunk = merged
            else:
                current_chunk = current_chunk + "\n\n" + para if current_chunk else para

        # 收尾：剩余的尾部内容直接成块，不因长度不足而丢弃
        if current_chunk.strip():
            emit(current_chunk)
        # 文档末尾遗留的图片（后面没有正文块可挂）挂到最后一块上
        if pending_images and chunks:
            chunks[-1].setdefault("images", []).extend(pending_images)
        return chunks

    def _split_blocks(self, text: str) -> List[tuple]:
        """把全文切成 ("text", 段落)、("table", 表格)、("image", 图片路径) 三类单元，表格不跨块"""
        blocks = []
        buf = []

        def flush_text():
            if buf:
                joined = "\n".join(buf).strip()
                if joined:
                    blocks.append(("text", joined))
                buf.clear()

        lines = text.split("\n")
        i = 0
        while i < len(lines):
            line = lines[i]
            img = re.match(r'^\s*!\[.*?\]\((.*?)\)\s*$', line)
            if img:
                # 图片引用行单独成单元（保留完整行，alt 文本不丢），不并入段落，保证路径能归属到正确的块
                flush_text()
                blocks.append(("image", line.strip()))
                i += 1
                continue
            if line.strip().startswith("|"):
                # 连续的 "|" 行视为一张完整表格
                flush_text()
                table_lines = []
                while i < len(lines) and lines[i].strip().startswith("|"):
                    table_lines.append(lines[i])
                    i += 1
                blocks.append(("table", "\n".join(table_lines)))
                continue
            if not line.strip():
                flush_text()
            else:
                buf.append(line)
            i += 1
        flush_text()
        return blocks

    def _split_table(self, markdown: str) -> List[str]:
        """表格按行窗口切分，每个子块重复表头并标注行号范围"""
        lines = [l.rstrip() for l in markdown.split("\n") if l.strip()]
        if not lines:
            return []

        # 前两行是表头 + 分隔线；没有分隔线时只把首行当表头
        head_rows = 2 if len(lines) > 1 and self._is_separator(lines[1]) else 1
        header, body = lines[:head_rows], lines[head_rows:]
        total = len(body)

        if total == 0:
            return ["\n".join(header)]

        chunks = []
        current = []
        start = 1
        for idx, row in enumerate(body, start=1):
            trial_size = len("\n".join(header + current + [row]))
            prefix_size = len(self._table_prefix(start, idx, total))
            if current and trial_size + prefix_size > self.max_chunk_size:
                chunks.append(self._table_prefix(start, idx - 1, total) + "\n".join(header + current))
                # 行级重叠：上一块最后一行带进下一块，避免跨行语义被切断
                keep = current[-1:]
                start = idx - len(keep)
                current = list(keep)
            current.append(row)

        if current:
            chunks.append(self._table_prefix(start, total, total) + "\n".join(header + current))
        return chunks

    @staticmethod
    def _table_prefix(start: int, end: int, total: int) -> str:
        return f"【表格 · 第{start}-{end}行/共{total}行】\n"

    @staticmethod
    def _is_separator(line: str) -> bool:
        return bool(re.fullmatch(r'\|?[\s:|-]+\|?', line.strip()))

    def _split_long_paragraph(self, text: str, start_index: int, section: str = "") -> List[Dict]:
        """分割过长的段落：先按句子切，超长句子再按字符切，并控制块大小与重叠。"""
        # 用中英文句末标点+可选空白切分，保留分隔符（括号分组）
        sentences = re.split(r'([。！？.!?]\s*)', text)
        # 奇偶配对：前半是“标点前文本”，后半是“标点+空白”，拼成完整句子列表
        # 末尾若没有标点也要保留，否则最后一段文字会被丢掉
        sentences = [sentences[i] + (sentences[i + 1] if i + 1 < len(sentences) else "")
                     for i in range(0, len(sentences), 2)]

        chunks = []           # 最终块列表
        current_chunk = ""    # 当前累积的块内容
        chunk_index = start_index  # 块序号，与调用方已有块衔接

        for sent in sentences:
            # 当前块加上本句会超限 → 先收尾当前块，再处理本句
            if len(current_chunk) + len(sent) > self.max_chunk_size:
                if current_chunk:
                    chunks.append({
                        "content": current_chunk.strip(),
                        "index": chunk_index,
                        "length": len(current_chunk),
                        "type": "text",
                        "section": section
                    })
                    chunk_index += 1

                # 本句单独就超长 → 按固定步长滑动窗口按字符切
                if len(sent) > self.max_chunk_size:
                    # 步长 = 块大小 - 重叠，使相邻子块有 overlap
                    for i in range(0, len(sent), self.max_chunk_size - self.chunk_overlap):
                        sub_chunk = sent[i:i + self.max_chunk_size]
                        if len(sub_chunk) >= self.min_chunk_size:
                            chunks.append({
                                "content": sub_chunk,
                                "index": chunk_index,
                                "length": len(sub_chunk),
                                "type": "text",
                                "section": section
                            })
                            chunk_index += 1
                        elif sub_chunk.strip() and chunks:
                            # 末尾不足最小长度的残余窗口并入上一块，避免丢字
                            last = chunks[-1]
                            last["content"] = last["content"] + sub_chunk
                            last["length"] = len(last["content"])
                    current_chunk = ""  # 长句已全部切成子块，清空
                else:
                    # 句级重叠：同一段内相邻块共享上一块尾部，保证语义连贯；
                    # 空间不够就截短，避免新块超过 max_chunk_size
                    room = self.max_chunk_size - len(sent)
                    n = min(self.chunk_overlap, room) if room > 0 else 0
                    tail = current_chunk[-n:] if n > 0 else ""
                    current_chunk = tail + sent
            else:
                current_chunk += sent  # 未超限，直接追加本句

        # 收尾：剩余内容太短就并入最后一块，够长才单独成块，避免丢字或产生碎片
        if current_chunk.strip():
          if chunks and len(current_chunk) < self.min_chunk_size:
            last = chunks[-1]
            last["content"] = last["content"] + "\n" + current_chunk.strip()
            last["length"] = len(last["content"])
          else:
            chunks.append({
                "content": current_chunk.strip(),
                "index": chunk_index,
                "length": len(current_chunk),
                "type": "text",
                "section": section
            })

        return chunks

    def _split_by_sentence(self, text: str) -> List[Dict]:
        """按句子分割。

        整体逻辑：先用正则按中英文句末标点切分文本并保留标点，得到“句子+标点”列表；
        然后顺序遍历句子，在不超过 max_chunk_size 的前提下把多句合并成一块；
        若加上当前句会超长，则先输出当前块（若达到 min_chunk_size），再把当前句作为新块起点；
        最后若还有剩余内容且达到最小块大小，作为最后一块输出。
        """
        # 按句末标点（。！？.!?）及其后可选空白切分，括号分组使标点被保留在结果中
        sentences = re.split(r'([。！？.!?]\s*)', text)
        # 奇偶配对：split 结果中偶数位是“标点前文本”，奇数位是“标点+空白”，拼成完整句子列表
        sentences = [''.join(pair) for pair in zip(sentences[::2], sentences[1::2])]

        chunks = []           # 最终输出的块列表
        current_chunk = ""    # 当前正在累积的一块内容
        chunk_index = 0       # 当前块的序号

        for sent in sentences:
            sent = sent.strip()   # 去掉句子首尾空白
            if not sent:          # 空句跳过
                continue

            # 若当前块加上本句会超过最大块大小，需要先收尾再开新块
            if len(current_chunk) + len(sent) > self.max_chunk_size:
                # 当前块已达到最小块大小时，输出为一块
                if len(current_chunk) >= self.min_chunk_size:
                    chunks.append({
                        "content": current_chunk.strip(),
                        "index": chunk_index,
                        "length": len(current_chunk),
                        "type": "text"
                    })
                    chunk_index += 1
                # 本句作为新块的起点（可能单句就超长，后续会在别处再切）
                current_chunk = sent
            else:
                # 未超限：拼到当前块，非首句时前加空格
                current_chunk += " " + sent if current_chunk else sent

        # 收尾：剩余内容达到最小块大小时作为最后一块
        if current_chunk and len(current_chunk) >= self.min_chunk_size:
            chunks.append({
                "content": current_chunk.strip(),
                "index": chunk_index,
                "length": len(current_chunk),
                "type": "text"
            })

        return chunks
        
    def _split_by_character(self, text: str) -> List[Dict]:
        """按字符数分割。

        整体逻辑：从文本开头起，以 (max_chunk_size - chunk_overlap) 为步长滑动截取，
        每次取长度为 max_chunk_size 的子串作为一块；步长小于块长从而实现块与块之间的重叠；
        仅当块去掉首尾空白后长度不小于 min_chunk_size 时才加入结果，避免过短碎片。
        """
        chunks = []           # 最终输出的块列表
        chunk_index = 0       # 当前块的序号

        # 按固定步长滑动：步长 = 块长 - 重叠，使相邻块有 chunk_overlap 的重叠
        for i in range(0, len(text), self.max_chunk_size - self.chunk_overlap):
            chunk = text[i:i + self.max_chunk_size]   # 截取本段，最多 max_chunk_size 字符
            # 仅当去掉空白后仍达到最小块大小时才计入
            if len(chunk.strip()) >= self.min_chunk_size:
                chunks.append({
                    "content": chunk.strip(),
                    "index": chunk_index,
                    "length": len(chunk),
                    "type": "text"
                })
                chunk_index += 1

        return chunks

    def _attach_tables(self,chunks: List[Dict], tables: List[Dict]) -> List[Dict]:
        '''
        将表格作为单独的块添加到分块列表中
        '''
        chunk_index = len(chunks)

        for table in tables:
            table_text = f'表格 (第{table.get("page","?")}页)：\n {table.get("markdown", "")}'

            chunks.append({
                "content": table_text,
                "index": chunk_index,
                "length": len(table_text),
                "type": "table",
                "page":table.get("page","?"),
                "table_data":table
            })
            chunk_index += 1
        
        return chunks