from pydantic import BaseModel
from typing import List

class CodeChunk(BaseModel):
    chunk_id: str
    repo_id: str
    version_id: str
    file_path: str
    symbol_name: str
    symbol_type: str
    start_line: int
    end_line: int
    code: str
    normalized_code: str
    context: str
    imports: List[str]
    exports: List[str]
    calls: List[str]
    called_by: List[str]
    chunk_hash: str
