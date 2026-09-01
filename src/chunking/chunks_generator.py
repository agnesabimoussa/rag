"""Chunk generation and source-span mapping for the document corpus."""
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
from src.chunking.code_chunker import CodeChunker
from src.chunking.markdown_chunker import MarkdownChunker
from src.data_models.chunk import Chunk, CodeChunk, MarkdownChunk
from src.error_handling.inavlid_json import InvalidJSON
from src.indexing.manifest import FileManifestEntry, IndexManifest
from src.utils.file_operations import FileOperations


class ChunksGenerator:
    """Splits every Markdown/Python file under a folder into `Chunk`s,
    using a distinct chunking strategy per file type.
    """

    def __init__(self, corpus_path: str = "data/raw/",
                 output_dir: str = "data/processed/",
                 max_chunk_size: int = 2000) -> None:
        """Initialize the chunker.
        Args:
            corpus_path: Root directory to recursively scan for `.md`/`.py`
                files.
            output_file_path: Where `write_result` persists the chunks.
            max_chunk_size: Maximum characters per chunk.
        """
        self.corpus_path = Path(corpus_path)
        self.output_dir = Path(output_dir)
        self.markdown_files: List[Path] = []
        self.python_files: List[Path] = []
        self.get_files()
        self.markdown_chunking_strategy = MarkdownChunker(max_chunk_size)
        self.code_chunking_strategy = CodeChunker(max_chunk_size)
        # Populated by apply_chunking(): the chunk-level diff needed to update
        # the vector/lexical indexes incrementally instead of rebuilding them.
        self.chunks_to_embed: List[Chunk] = []
        self.removed_chunk_ids: List[str] = []
        self.has_changes = False

    def get_files(self) -> None:
        if not self.corpus_path.is_dir():
            raise FileNotFoundError(
                f"Input directory does not exist: {self.corpus_path}"
            )

        file_paths = sorted(self.corpus_path.rglob("*"))
        for file_path in tqdm(file_paths, desc="Chunking"):
            if file_path.is_file() and file_path.suffix.lower() == ".md":
                self.markdown_files.append(file_path)
            elif file_path.is_file() and file_path.suffix.lower() == ".py":
                self.python_files.append(file_path)

    @staticmethod
    def write_result(path: Path, file_name: str, content: List[Chunk]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        full_path = path / file_name
        FileOperations.write_json(full_path, content)

    def _load_existing_chunks(self) -> Tuple[List[MarkdownChunk], List[CodeChunk]]:
        markdown_path = self.output_dir / "markdown_chunks.json"
        code_path = self.output_dir / "code_chunks.json"
        try:
            markdown_chunks = FileOperations.load_content(
                markdown_path, List[MarkdownChunk]) if markdown_path.is_file() else []
        except InvalidJSON:
            markdown_chunks = []
        try:
            code_chunks = FileOperations.load_content(
                code_path, List[CodeChunk]) if code_path.is_file() else []
        except InvalidJSON:
            code_chunks = []
        return markdown_chunks, code_chunks

    def apply_chunking(self) -> List[Chunk]:
        """Chunk only the files that are new or changed since the last run.

        Diffs the corpus against `manifest.json` (source path -> content
        hash + chunk IDs). Unchanged files keep their previously written
        chunks; changed/deleted files have their old chunks dropped and
        contribute to `removed_chunk_ids`; new/changed files are (re-)chunked
        and land in `chunks_to_embed`. Both are consumed by the indexing step
        to update the vector index in place instead of re-embedding
        everything.
        """
        manifest_path = self.output_dir / "manifest.json"
        manifest = IndexManifest.load(manifest_path)

        all_files = self.markdown_files + self.python_files
        current_paths = {str(file) for file in all_files}
        current_hashes = {
            str(file): FileOperations.get_file_hash(file) for file in all_files
        }

        removed_paths = set(manifest.files) - current_paths
        changed_paths = {
            path for path in current_paths
            if path not in manifest.files or manifest.files[path].hash != current_hashes[path]
        }
        stale_paths = changed_paths | removed_paths

        self.removed_chunk_ids = [
            chunk_id
            for path in stale_paths
            for chunk_id in manifest.files.get(path, FileManifestEntry(hash="")).chunk_ids
        ]

        existing_markdown, existing_code = self._load_existing_chunks()
        unchanged_markdown = [chunk for chunk in existing_markdown if chunk.source not in stale_paths]
        unchanged_code = [chunk for chunk in existing_code if chunk.source not in stale_paths]

        files_to_chunk_md = [file for file in self.markdown_files if str(file) in changed_paths]
        files_to_chunk_py = [file for file in self.python_files if str(file) in changed_paths]

        new_markdown_chunks = self.markdown_chunking_strategy.chunk_files(files_to_chunk_md)
        new_code_chunks = self.code_chunking_strategy.chunk_files(files_to_chunk_py)

        self.chunks_to_embed = [*new_markdown_chunks, *new_code_chunks]
        self.has_changes = bool(self.chunks_to_embed) or bool(removed_paths)

        markdown_chunks = unchanged_markdown + new_markdown_chunks
        code_chunks = unchanged_code + new_code_chunks
        self.write_result(self.output_dir, "markdown_chunks.json", markdown_chunks)
        self.write_result(self.output_dir, "code_chunks.json", code_chunks)

        all_chunks = markdown_chunks + code_chunks
        chunk_ids_by_source: Dict[str, List[str]] = {}
        for chunk in all_chunks:
            chunk_ids_by_source.setdefault(chunk.source, []).append(chunk.id)
        manifest.files = {
            path: FileManifestEntry(hash=current_hashes[path], chunk_ids=chunk_ids_by_source.get(path, []))
            for path in current_paths
        }
        manifest.save(manifest_path)

        return all_chunks
