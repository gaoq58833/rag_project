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
    @abstractmethod
    def parse(self,file_path:str) ->ParseResult:
        pass
    @abstractmethod
    def parse_from_bytes(self,file_bytes:bytes,filename:str) ->ParseResult:
        pass
    @abstractmethod
    def supported_extensions(self) ->List[str]:
        pass