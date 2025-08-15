import requests
from tenacity import retry, stop_after_attempt, wait_exponential
import numpy as np
test_string = "你好"
ollama_server_url = "http://47.109.138.137:11433/api/embeddings"

def get_embedding(text):
    resp = requests.post(
        ollama_server_url,
        json={'model': "nomic-embed-text", 'prompt': text}
    )
    resp.raise_for_status()
    return np.array(resp.json()['embedding'])

if __name__ == "__main__":
    embedding = get_embedding(test_string)
    print(embedding)
    print(embedding.shape)


# -------- Ollama Embedder --------
class OllamaEmbedder:
    def __init__(self, model: str, url: str = "http://localhost:11434"):
        self.model = model
        self.url = url.rstrip("/")

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=0.5, min=0.5, max=8))
    def embed_one(self, text: str) -> List[float]:
        # Ollama embeddings API expects 'model' and 'prompt'
        resp = requests.post(
            f"{self.url}/api/embeddings",
            json={"model": self.model, "prompt": text},
            timeout=60
        )
        resp.raise_for_status()
        data = resp.json()
        return data["embedding"]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        # Simple sequential embedding; can be parallelized if needed
        return [self.embed_one(t) for t in texts]

    def embedding_dim(self) -> int:
        vec = self.embed_one("dimension probe")
        return len(vec)