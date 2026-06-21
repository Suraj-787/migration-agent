from pydantic import BaseModel, Field


class CodeChunk(BaseModel):
    file_path: str
    language: str
    content: str
    node_type: str = Field(description="function | class | import | code")
    parent_class: str | None = None
    parent_context: str = Field(description="Ancestor path or file_path; always non-empty")
    start_line: int
    end_line: int
    docstring: str | None = None


class DocChunk(BaseModel):
    file_path: str
    title: str
    content: str
    chunk_index: int
    source: str = Field(description="Derived from the markdown filename (no extension)")
