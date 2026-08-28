from typing import List, Optional
from pathlib import Path
import ast
import logging
from src.chunking_modules.abstract_chunker import AbstractChunker
from src.chunking_modules.chunk import CodeChunk


logger = logging.getLogger(__name__)


class CodeChunking(AbstractChunker):
    def __init__(self,
                 max_chunk_size: int) -> None:
        super().__init__(max_chunk_size)
        self.id_prefix = "py_"

    def make_chunk(self, text: str, source: str, first_char_idx: int, last_char_idx: int,
                   original_chunk_id: str | None, type: str | None, parent_id: str | None,
                   child_ids: List[str] | None) -> CodeChunk:
        return CodeChunk(id=self.id_prefix + str(next(self.id_generator)),
                         text=text,
                         source=source,
                         first_character_index=first_char_idx,
                         last_character_index=last_char_idx,
                         tokens=len(text),
                         original_chunk_id=original_chunk_id,
                         type=type,
                         parent_id=parent_id,
                         child_ids=child_ids
                         )

    def chunk_file(self, file_name: Path, content: str) -> List[CodeChunk]:
        chunks: List[CodeChunk] = []
        try:
            tree = ast.parse(content, filename=str(file_name))
        except SyntaxError as error:
            logger.warning(
                "Skipping %s: invalid Python syntax at line %s, column %s: %s",
                file_name,
                error.lineno,
                error.offset,
                error.msg,
            )
            return chunks
        self._walk(tree, content, file_name, chunks, parent=None)
        if chunks:
            return chunks
        # skips pure imports files
        is_pure_imports = all(
            isinstance(n, (ast.Import, ast.ImportFrom)) for n in tree.body
        )
        if is_pure_imports:
            return chunks

        module_docstring = ast.get_docstring(tree)
        # chunk on max chunk size
        if module_docstring:
            sub_texts = self.enforce_char_limit(module_docstring)
            docstring_start, _ = self.find_span(content, module_docstring)
            docstring_cursor = 0
            first_chunk = None
            for text in sub_texts:
                first_char_rel, last_char_rel = self.find_span(
                    module_docstring,
                    text,
                    docstring_cursor,
                )
                first_char_idx = docstring_start + first_char_rel
                last_char_idx = docstring_start + last_char_rel - 1
                chunk = self.make_chunk(
                    text=text,
                    source=str(file_name),
                    first_char_idx=first_char_idx,
                    last_char_idx=last_char_idx,
                    original_chunk_id=first_chunk.id if first_chunk else None,
                    type="Docstring",
                    parent_id=None,
                    child_ids=None,
                )
                chunks.append(chunk)
                first_chunk = first_chunk or chunk
                docstring_cursor = last_char_rel
        return chunks

    def _walk(self,
              node: ast.AST,
              content: str,
              file_name: Path,
              chunks: List[CodeChunk],
              parent: Optional[str]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                first_chunk_id = self._chunk_function(
                    child, content, file_name, chunks, parent
                )
                self._walk(child, content, file_name, chunks, parent=first_chunk_id)

            elif isinstance(child, ast.ClassDef):
                self._walk(child, content, file_name, chunks, parent=parent)

    def _chunk_function(self,
                        node,
                        content: str,
                        file_name: Path,
                        chunks: List[CodeChunk],
                        parent: Optional[str]) -> str:
        text = ast.get_source_segment(content, node) or ""
        sub_texts = self.enforce_char_limit(text)
        node_start, _ = self.find_span(content, text)
        cursor = 0
        first_chunk_id = ""
        chunk_type = "AsyncFunction" if isinstance(node, ast.AsyncFunctionDef) else "Function"
        for sub_text in sub_texts:
            sub_start_rel, sub_end_rel = self.find_span(text, sub_text, cursor)
            sub_start = node_start + sub_start_rel
            sub_end = node_start + sub_end_rel
            chunk = self.make_chunk(
                text=sub_text,
                source=str(file_name),
                first_char_idx=sub_start,
                last_char_idx=sub_end - 1,
                original_chunk_id=first_chunk_id or None,
                type=chunk_type,
                parent_id=parent,
                child_ids=None,
            )
            chunks.append(chunk)
            first_chunk_id = first_chunk_id or chunk.id
            cursor = sub_end_rel
        return first_chunk_id
