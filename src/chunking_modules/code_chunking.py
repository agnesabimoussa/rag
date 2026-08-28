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

    @staticmethod
    def _get_docstring_source(node: ast.AST, content: str) -> str | None:
        if not hasattr(node, "body") or not node.body:
            return None
        first_stmt = node.body[0]
        if not isinstance(first_stmt, ast.Expr):
            return None
        value = first_stmt.value
        if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
            return None
        return ast.get_source_segment(content, first_stmt)

    def _chunk_source_text(
        self,
        text: str,
        source: str,
        first_char_idx: int,
        original_chunk_id: str | None,
        type: str | None,
        parent_id: str | None,
        child_ids: List[str] | None,
        chunks: List[CodeChunk],
    ) -> str:
        sub_texts = self.enforce_char_limit(text)
        cursor = 0
        first_chunk_id = ""
        for sub_text in sub_texts:
            sub_start_rel, sub_end_rel = self.find_span(text, sub_text, cursor)
            chunk = self.make_chunk(
                text=sub_text,
                source=source,
                first_char_idx=first_char_idx + sub_start_rel,
                last_char_idx=first_char_idx + sub_end_rel - 1,
                original_chunk_id=original_chunk_id if first_chunk_id == "" else first_chunk_id,
                type=type,
                parent_id=parent_id,
                child_ids=child_ids,
            )
            chunks.append(chunk)
            first_chunk_id = first_chunk_id or chunk.id
            cursor = sub_end_rel
        return first_chunk_id

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

        module_docstring_source = self._get_docstring_source(tree, content)
        if module_docstring_source:
            module_docstring_start, _ = self.find_span(content, module_docstring_source)
            self._chunk_source_text(
                text=module_docstring_source,
                source=str(file_name),
                first_char_idx=module_docstring_start,
                original_chunk_id=None,
                type="ModuleDocstring",
                parent_id=None,
                child_ids=None,
                chunks=chunks,
            )

        self._walk(tree, content, file_name, chunks, parent=None)
        # skips pure imports files
        is_pure_imports = all(
            isinstance(n, (ast.Import, ast.ImportFrom)) for n in tree.body
        )
        if is_pure_imports:
            return chunks
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
                class_docstring_source = self._get_docstring_source(child, content)
                if class_docstring_source:
                    class_source = ast.get_source_segment(content, child) or ""
                    class_start, _ = self.find_span(content, class_source)
                    docstring_start, _ = self.find_span(class_source, class_docstring_source)
                    self._chunk_source_text(
                        text=class_docstring_source,
                        source=str(file_name),
                        first_char_idx=class_start + docstring_start,
                        original_chunk_id=None,
                        type="ClassDocstring",
                        parent_id=parent,
                        child_ids=None,
                        chunks=chunks,
                    )
                self._walk(child, content, file_name, chunks, parent=parent)

    def _chunk_function(self,
                        node,
                        content: str,
                        file_name: Path,
                        chunks: List[CodeChunk],
                        parent: Optional[str]) -> str:
        text = ast.get_source_segment(content, node) or ""
        node_start, _ = self.find_span(content, text)
        chunk_type = "AsyncFunction" if isinstance(node, ast.AsyncFunctionDef) else "Function"
        return self._chunk_source_text(
            text=text,
            source=str(file_name),
            first_char_idx=node_start,
            original_chunk_id=None,
            type=chunk_type,
            parent_id=parent,
            child_ids=None,
            chunks=chunks,
        )
