import numpy as np

from sentence_transformers import SentenceTransformer
from gptcache.embedding.base import BaseEmbedding


class BGE(BaseEmbedding):
    def __init__(self, model: str="BAAI/bge-large-zh-v1.5"):
        self.model = SentenceTransformer(model)
        self.model.eval()
        self.__dimension = None

    def to_embeddings(self, data, **_):
        embeddings = self.model.encode(data, normalize_embeddings=True)
        return np.array(embeddings).astype('float32')

    @property
    def dimension(self):
        if not self.__dimension:
            self.__dimension = len(self.to_embeddings("foo"))
        return self.__dimension