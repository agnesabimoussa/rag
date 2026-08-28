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
    def _build_line_offsets(content: str) -> List[int]:
        offsets = [0]
        total = 0
        for line in content.splitlines(keepends=True):
            total += len(line)
            offsets.append(total)
        return offsets

    @staticmethod
    def _node_span(node: ast.AST, line_offsets: List[int]) -> tuple[int, int]:
        if not hasattr(node, "lineno") or not hasattr(node, "end_lineno"):
            raise ValueError("AST node does not expose source offsets.")
        start = line_offsets[node.lineno - 1] + node.col_offset
        end = line_offsets[node.end_lineno - 1] + node.end_col_offset
        return start, end

    @staticmethod
    def _get_docstring_node(node: ast.AST) -> ast.Expr | None:
        if not hasattr(node, "body") or not node.body:
            return None
        first_stmt = node.body[0]
        if not isinstance(first_stmt, ast.Expr):
            return None
        value = first_stmt.value
        if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
            return None
        return first_stmt

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

    def _chunk_body_statements(
        self,
        body: List[ast.stmt],
        content: str,
        file_name: Path,
        chunks: List[CodeChunk],
        line_offsets: List[int],
        parent: Optional[str],
        type_prefix: str,
    ) -> None:
        for stmt in body:
            if isinstance(stmt, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            source_text = ast.get_source_segment(content, stmt) or ""
            if not source_text.strip():
                continue
            start, _ = self._node_span(stmt, line_offsets)
            self._chunk_source_text(
                text=source_text,
                source=str(file_name),
                first_char_idx=start,
                original_chunk_id=None,
                type=f"{type_prefix}{stmt.__class__.__name__}",
                parent_id=parent,
                child_ids=None,
                chunks=chunks,
            )

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
        line_offsets = self._build_line_offsets(content)
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

        module_docstring_node = self._get_docstring_node(tree)
        if module_docstring_node is not None:
            module_docstring_source = ast.get_source_segment(content, module_docstring_node) or ""
            module_docstring_start, _ = self._node_span(module_docstring_node, line_offsets)
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

        module_body = tree.body[1:] if module_docstring_node is not None else tree.body
        self._chunk_body_statements(
            module_body,
            content,
            file_name,
            chunks,
            line_offsets,
            parent=None,
            type_prefix="Module",
        )
        self._walk(tree, content, file_name, chunks, parent=None)
        return chunks

    def _walk(self,
              node: ast.AST,
              content: str,
              file_name: Path,
              chunks: List[CodeChunk],
              parent: Optional[str],
              line_offsets: List[int]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                first_chunk_id = self._chunk_function(
                    child, content, file_name, chunks, parent, line_offsets
                )
                self._walk(child, content, file_name, chunks, parent=first_chunk_id, line_offsets=line_offsets)

            elif isinstance(child, ast.ClassDef):
                class_docstring_node = self._get_docstring_node(child)
                if class_docstring_node is not None:
                    class_docstring_source = ast.get_source_segment(content, class_docstring_node) or ""
                    class_docstring_start, _ = self._node_span(class_docstring_node, line_offsets)
                    self._chunk_source_text(
                        text=class_docstring_source,
                        source=str(file_name),
                        first_char_idx=class_docstring_start,
                        original_chunk_id=None,
                        type="ClassDocstring",
                        parent_id=parent,
                        child_ids=None,
                        chunks=chunks,
                    )
                class_body = child.body[1:] if class_docstring_node is not None else child.body
                self._chunk_body_statements(
                    class_body,
                    content,
                    file_name,
                    chunks,
                    line_offsets,
                    parent=parent,
                    type_prefix="Class",
                )
                self._walk(child, content, file_name, chunks, parent=parent, line_offsets=line_offsets)

    def _chunk_function(self,
                        node,
                        content: str,
                        file_name: Path,
                        chunks: List[CodeChunk],
                        parent: Optional[str],
                        line_offsets: List[int]) -> str:
        text = ast.get_source_segment(content, node) or ""
        node_start, _ = self._node_span(node, line_offsets)
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
