from rank_bm25 import BM25Okapi


class LexicalRetriever:
    def __init__(self, bm25: BM25Okapi) -> None:
        self.bm25 = bm25
