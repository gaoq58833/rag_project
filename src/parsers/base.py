from abc import ABC,abstractmethod
from typing import Dict,List
from dataclasses import dataclass
@dataclass
class ParseResult:
    filename:str
    total_pages:int
    total_chunks:int
    table_count:int
    chunks:List[Dict]
    raw_text:str
    tables:List[Dict]
    metadata:Dict
class BaseParser(ABC):
    # 该格式应使用的索引类型：vector（向量+倒排混合）/ bm25（只建倒排索引）
    # 入库流程据此决定是否调用 embedding 与写入 Qdrant
    INDEX_TYPE = "vector"

    @abstractmethod
    def parse(self,file_path:str) ->ParseResult:
        pass
    @abstractmethod
    def parse_from_bytes(self,file_bytes:bytes,filename:str) ->ParseResult:
        pass
    @abstractmethod
    def supported_extensions(self) ->List[str]:
        pass