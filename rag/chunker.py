import re
from pathlib import Path

from astchunk.astchunk_builder import ASTChunkBuilder  # type: ignore[import-untyped]
from loguru import logger

from rag.models import CodeChunk

_EXT_TO_LANGUAGE: dict[str, str] = {
    ".py": "python",
    ".java": "java",
    ".cs": "csharp",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "typescript",
    ".jsx": "typescript",
}

# Map tree-sitter node types to our simplified node_type vocabulary.
_TS_NODE_TYPE_MAP: dict[str, str] = {
    # Python
    "function_definition": "function",
    "class_definition": "class",
    "import_statement": "import",
    "import_from_statement": "import",
    # Java / C#
    "method_declaration": "function",
    "constructor_declaration": "function",
    "class_declaration": "class",
    # JS / TS
    "function_declaration": "function",
    "method_definition": "function",
    "arrow_function": "function",
    "import_declaration": "import",
}


def _extract_docstring(code: str, language: str) -> str | None:
    if language == "python":
        m = re.search(r'"""(.*?)"""', code, re.DOTALL)
        if not m:
            m = re.search(r"'''(.*?)'''", code, re.DOTALL)
        if m:
            return m.group(1).strip() or None
    return None


class CodeChunker:
    """Wraps ASTChunkBuilder to produce typed CodeChunk objects per file."""

    def __init__(self) -> None:
        self._builders: dict[str, ASTChunkBuilder] = {}

    def _get_builder(self, language: str) -> ASTChunkBuilder:
        if language not in self._builders:
            self._builders[language] = ASTChunkBuilder(
                max_chunk_size=1200,
                language=language,
                metadata_template="default",
            )
        return self._builders[language]

    def chunk_file(self, file_path: str) -> list[CodeChunk]:
        path = Path(file_path)
        language = _EXT_TO_LANGUAGE.get(path.suffix.lower())
        if language is None:
            logger.debug("Skipping unsupported extension: {}", file_path)
            return []

        try:
            code = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            logger.warning("Cannot read {}: {}", file_path, exc)
            return []

        if not code.strip():
            return []

        builder = self._get_builder(language)
        ast_tree = builder.parser.parse(bytes(code, "utf-8"))
        ast_windows = list(builder.assign_tree_to_windows(code=code, root_node=ast_tree.root_node))
        if not ast_windows:
            return []

        raw_chunks = builder.convert_windows_to_chunks(
            ast_windows,
            repo_level_metadata={"filepath": str(file_path)},
            chunk_expansion=False,
        )

        return [self._to_code_chunk(chunk, str(file_path), language) for chunk in raw_chunks]

    def _to_code_chunk(self, chunk: object, file_path: str, language: str) -> CodeChunk:
        node_type = self._classify_node_type(chunk)
        parent_class = self._extract_parent_class(chunk)
        ancestors: list[str] = chunk.chunk_ancestors  # type: ignore[attr-defined]
        parent_context = "\n".join(ancestors) if ancestors else file_path

        return CodeChunk(
            file_path=file_path,
            language=language,
            content=chunk.chunk_text,  # type: ignore[attr-defined]
            node_type=node_type,
            parent_class=parent_class,
            parent_context=parent_context,
            start_line=chunk.start_line,  # type: ignore[attr-defined]
            end_line=chunk.end_line,  # type: ignore[attr-defined]
            docstring=_extract_docstring(chunk.chunk_text, language),  # type: ignore[attr-defined]
        )

    def _classify_node_type(self, chunk: object) -> str:
        ast_window = chunk.ast_window  # type: ignore[attr-defined]
        if not ast_window:
            return "code"
        ts_type: str = ast_window[0].node.type
        return _TS_NODE_TYPE_MAP.get(ts_type, "code")

    def _extract_parent_class(self, chunk: object) -> str | None:
        for ancestor in reversed(chunk.chunk_ancestors):  # type: ignore[attr-defined]
            stripped = ancestor.strip()
            if stripped.startswith("class "):
                # "class Foo(Bar):" → "Foo"
                name = stripped[len("class "):].split("(")[0].split(":")[0].strip()
                return name or None
        return None
