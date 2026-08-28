from typing import List
from pathlib import Path
from src.chunking_modules.chunk import MarkdownChunk
from src.chunking_modules.abstract_chunker import AbstractChunker
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings


class MarkdwonChunking(AbstractChunker):
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
        self.semantic_splitter = SemanticChunker(
            embeddings=HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            ),
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=95,
        )

    def make_chunk(self, text, source, first_char_idx, last_char_idx,
                   original_chunk_id, section: str) -> MarkdownChunk:
        return MarkdownChunk(id=self.id_prefix + str(next(self.id_generator)),
                             text=text,
                             source=source,
                             first_character_index=first_char_idx,
                             last_character_index=last_char_idx,
                             tokens=len(text),
                             original_chunk_id=original_chunk_id,
                             section=section
                             )

    def semantic_chunk_section(self, text: str) -> list[str]:
        if len(text) <= self.max_chunk_size:
            return [text]
        semantic_chunks = self.semantic_splitter.split_text(text)
        final = []
        for sc in semantic_chunks:
            final.extend(self.enforce_char_limit(sc))
        return final

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
            sub_texts = self.semantic_chunk_section(section_text)
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
