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

        self._chunk_body(
            tree.body,
            content,
            file_name,
            chunks,
            line_offsets,
            parent=None,
            type_prefix="Module",
        )
        return chunks

    # Greedily merges consecutive statements (including whole small
    # functions/classes) into blocks up to max_chunk_size, mirroring how a
    # size-based splitter would chunk the same source, instead of emitting
    # one tiny chunk per statement. A single item too large to fit on its
    # own is recursed into (classes) or hard-split by lines (functions and
    # other statements), so only the pieces that actually need splitting
    # end up split.
    def _chunk_body(
        self,
        body: List[ast.stmt],
        content: str,
        file_name: Path,
        chunks: List[CodeChunk],
        line_offsets: List[int],
        parent: Optional[str],
        type_prefix: str,
    ) -> None:
        pending: List[ast.stmt] = []

        def flush() -> None:
            if not pending:
                return
            start, _ = self._node_span(pending[0], line_offsets)
            _, end = self._node_span(pending[-1], line_offsets)
            text = content[start:end]
            if text.strip():
                self._chunk_source_text(
                    text=text,
                    source=str(file_name),
                    first_char_idx=start,
                    original_chunk_id=None,
                    type=f"{type_prefix}Block",
                    parent_id=parent,
                    child_ids=None,
                    chunks=chunks,
                )
            pending.clear()

        for stmt in body:
            if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                continue
            start, end = self._node_span(stmt, line_offsets)
            if end - start > self.max_chunk_size:
                flush()
                if isinstance(stmt, ast.ClassDef):
                    self._chunk_body(
                        stmt.body,
                        content,
                        file_name,
                        chunks,
                        line_offsets,
                        parent=parent,
                        type_prefix=f"{type_prefix}{stmt.name}",
                    )
                else:
                    type_name = stmt.__class__.__name__
                    text = ast.get_source_segment(content, stmt) or content[start:end]
                    self._chunk_source_text(
                        text=text,
                        source=str(file_name),
                        first_char_idx=start,
                        original_chunk_id=None,
                        type=f"{type_prefix}{type_name}",
                        parent_id=parent,
                        child_ids=None,
                        chunks=chunks,
                    )
                continue
            if pending:
                pending_start, _ = self._node_span(pending[0], line_offsets)
                if end - pending_start > self.max_chunk_size:
                    flush()
            pending.append(stmt)
        flush()
