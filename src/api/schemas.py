
from pydantic import BaseModel

class AskRequest(BaseModel):
    question: str
    top_k: int = 5

class SourceInfo(BaseModel):
    title: str
    source_path: str

class AskResponse(BaseModel):
    answer: str
    sources: list[SourceInfo]