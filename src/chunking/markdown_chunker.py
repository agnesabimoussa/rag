from typing import List
from pathlib import Path
from src.data_models.chunk import MarkdownChunk
from src.chunking.chunker import Chunker
from langchain_text_splitters import MarkdownHeaderTextSplitter


class MarkdownChunker(Chunker):
    def __init__(self,
                 max_chunk_size: int) -> None:
        super().__init__(max_chunk_size)
        self.id_prefix = "md_"
        self.headers = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
            ("####", "Header 4"),
            ("#####", "Header 5"),
            ("######", "Header 6"),
        ]
        self.splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers,
            strip_headers=True,
        )

    def make_chunk(self, text: str, source: str, first_char_idx: int, last_char_idx: int,
                   original_chunk_id: str | None, section: str) -> MarkdownChunk:
        return MarkdownChunk(id=self.make_chunk_id(self.id_prefix, source, first_char_idx, last_char_idx),
                             text=text,
                             source=source,
                             first_character_index=first_char_idx,
                             last_character_index=last_char_idx,
                             tokens=len(text),
                             original_chunk_id=original_chunk_id,
                             section=section
                             )

    def split_section(self, text: str) -> list[str]:
        if len(text) <= self.max_chunk_size:
            return [text]
        return self.enforce_char_limit(text)

    def chunk_file(self, file_name: Path, content: str) -> List[MarkdownChunk]:
        results = []
        structure_chunks = self.splitter.split_text(content)
        search_cursor = 0
        for chunk in structure_chunks:
            section = " > ".join(chunk.metadata.values())
            source = str(file_name)
            section_text = chunk.page_content
            section_start, section_end = self.find_span(content, section_text, search_cursor)
            search_cursor = section_end
            sub_texts = self.split_section(section_text)
            sub_spans: list[tuple[str, int, int]] = []
            sub_cursor = 0
            for text in sub_texts:
                sub_start_relative, sub_end_relative = self.find_span(section_text, text, sub_cursor)
                sub_cursor = sub_end_relative
                sub_spans.append(
                    (
                        text,
                        section_start + sub_start_relative,
                        section_start + sub_end_relative,
                    )
                )

            first_text, first_start, first_end = sub_spans[0]
            first_chunk = self.make_chunk(
                first_text,
                source,
                first_start,
                max(first_start, first_end - 1),
                None,
                section,
            )
            results.append(first_chunk)

            for text, sub_start, sub_end in sub_spans[1:]:
                results.append(self.make_chunk(
                    text,
                    source,
                    sub_start,
                    max(sub_start, sub_end - 1),
                    first_chunk.id,
                    section,
                ))
        return results
