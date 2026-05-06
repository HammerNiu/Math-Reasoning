"""
Interface for different LLM implementations
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import json
from pathlib import Path
from dataclasses import dataclass
from openai import OpenAI
import anthropic


class LocalEmbedder:
    """Free local embeddings via sentence-transformers — no API key required.

    Default model: all-MiniLM-L6-v2  (dim=384, ~80 MB, fast)
    Install once:  pip install sentence-transformers

    Used automatically by DeepSeekModel and AnthropicModel so they no longer
    need an OpenAI key for embeddings.
    """

    _instances: Dict[str, "LocalEmbedder"] = {}

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. Run: pip install sentence-transformers"
            )
        self._model = SentenceTransformer(model_name)
        self.dim: int = self._model.get_sentence_embedding_dimension()

    @classmethod
    def get(cls, model_name: str = "all-MiniLM-L6-v2") -> "LocalEmbedder":
        """Return a cached singleton per model name to avoid repeated disk loads."""
        if model_name not in cls._instances:
            cls._instances[model_name] = cls(model_name)
        return cls._instances[model_name]

    def embed_text(self, text: str) -> List[float]:
        return self._model.encode(text, convert_to_numpy=True).tolist()

@dataclass
class ModelConfig:
    model: str
    temperature: float = 0.7
    max_tokens: int = 1000
    timeout: float = 30.0

class LLMInterface(ABC):
    @abstractmethod
    def generate_response(self,
                         prompt: str,
                         temperature: float = 0.7,
                         max_tokens: int = 1000) -> str:
        """Generate a response from the LLM."""
        pass

    @abstractmethod
    def evaluate_reasoning(self,
                          problem: str,
                          solution_steps: List[str]) -> float:
        """Evaluate the quality of reasoning steps."""
        pass

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Generate embeddings for text."""
        pass

    def encode(self, text: str) -> List[float]:
        """Alias for embed_text, used by PPM."""
        return self.embed_text(text)

class OpenAIModel(LLMInterface):
    def __init__(self, api_key: str, config: Optional[ModelConfig] = None):
        self.api_key = api_key
        self.config = config or ModelConfig(model="gpt-4o-mini")
        self.client = OpenAI(api_key=api_key, timeout=self.config.timeout, max_retries=0)

    @classmethod
    def from_config_file(cls, config_path: str, api_key: str) -> 'OpenAIModel':
        """Create OpenAI model instance from config file."""
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        config = ModelConfig(**config_data['models']['openai'])
        return cls(api_key, config)

    def generate_response(self,
                         prompt: str,
                         temperature: Optional[float] = None,
                         max_tokens: Optional[int] = None) -> str:
        """Generate a response using OpenAI API."""
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature if temperature is not None else self.config.temperature,
            max_tokens=max_tokens or self.config.max_tokens
        )
        return response.choices[0].message.content

    def evaluate_reasoning(self,
                          problem: str,
                          solution_steps: List[str]) -> float:
        """Evaluate reasoning steps using OpenAI."""
        prompt = f"""Problem: {problem}
Solution Steps:
{chr(10).join(f'{i+1}. {step}' for i, step in enumerate(solution_steps))}

Rate the quality of these solution steps from 0 to 1, where:
0 = completely incorrect or invalid reasoning
1 = perfect, clear, and mathematically sound reasoning

Respond with only the numerical rating, nothing else."""

        response = self.generate_response(prompt, max_tokens=10)
        try:
            rating = float(response.strip())
            return max(0.0, min(1.0, rating))
        except ValueError:
            return 0.5

    def embed_text(self, text: str) -> List[float]:
        """Generate embeddings using OpenAI API."""
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding

class AnthropicModel(LLMInterface):
    def __init__(self, api_key: str, config: Optional[ModelConfig] = None):
        self.api_key = api_key
        self.config = config or ModelConfig(model="claude-3-5-haiku-20241022")
        self.client = anthropic.Anthropic(
            api_key=api_key,
            timeout=self.config.timeout,
            max_retries=0,
        )

    @classmethod
    def from_config_file(cls, config_path: str, api_key: str) -> 'AnthropicModel':
        """Create Anthropic model instance from config file."""
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        config = ModelConfig(**config_data['models']['anthropic'])
        return cls(api_key, config)

    def generate_response(self,
                         prompt: str,
                         temperature: Optional[float] = None,
                         max_tokens: Optional[int] = None) -> str:
        """Generate a response using Anthropic API."""
        response = self.client.messages.create(
            model=self.config.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature if temperature is not None else self.config.temperature,
            max_tokens=max_tokens or self.config.max_tokens
        )
        return response.content[0].text

    def evaluate_reasoning(self,
                          problem: str,
                          solution_steps: List[str]) -> float:
        """Evaluate reasoning steps using Anthropic."""
        prompt = f"""Problem: {problem}
Solution Steps:
{chr(10).join(f'{i+1}. {step}' for i, step in enumerate(solution_steps))}

Rate the quality of these solution steps from 0 to 1.
Respond with only the numerical rating."""

        response = self.generate_response(prompt, max_tokens=10)
        try:
            rating = float(response.strip())
            return max(0.0, min(1.0, rating))
        except ValueError:
            return 0.5

    def embed_text(self, text: str) -> List[float]:
        """Anthropic has no embedding API; use free local sentence-transformers."""
        return LocalEmbedder.get().embed_text(text)

class DeepSeekModel(LLMInterface):
    """DeepSeek model using the OpenAI-compatible API."""

    def __init__(self, api_key: str, config: Optional[ModelConfig] = None):
        self.api_key = api_key
        self.config = config or ModelConfig(model="deepseek-chat")
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
            timeout=self.config.timeout,
            max_retries=0,
        )

    @classmethod
    def from_config_file(cls, config_path: str, api_key: str) -> 'DeepSeekModel':
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        config = ModelConfig(**config_data['models']['deepseek'])
        return cls(api_key, config)

    def generate_response(self,
                         prompt: str,
                         temperature: Optional[float] = None,
                         max_tokens: Optional[int] = None) -> str:
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature if temperature is not None else self.config.temperature,
            max_tokens=max_tokens or self.config.max_tokens
        )
        return response.choices[0].message.content

    def evaluate_reasoning(self,
                          problem: str,
                          solution_steps: List[str]) -> float:
        prompt = f"""Problem: {problem}
Solution Steps:
{chr(10).join(f'{i+1}. {step}' for i, step in enumerate(solution_steps))}

Rate the quality of these solution steps from 0 to 1, where:
0 = completely incorrect or invalid reasoning
1 = perfect, clear, and mathematically sound reasoning

Respond with only the numerical rating, nothing else."""
        response = self.generate_response(prompt, max_tokens=10)
        try:
            return max(0.0, min(1.0, float(response.strip())))
        except ValueError:
            return 0.5

    def embed_text(self, text: str) -> List[float]:
        """DeepSeek has no embedding API; use free local sentence-transformers."""
        return LocalEmbedder.get().embed_text(text)


class OllamaModel(LLMInterface):
    """Local model via Ollama (OpenAI-compatible, no API key needed).

    Start Ollama and pull a model first:
        ollama pull qwen2.5-math:7b
    Then use model_name="ollama" with ollama_model="qwen2.5-math:7b".
    """

    def __init__(self, ollama_model: str = "qwen2.5-math:7b",
                 base_url: str = "http://localhost:11434",
                 config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig(model=ollama_model)
        self.client = OpenAI(
            api_key="ollama",
            base_url=f"{base_url}/v1",
            timeout=self.config.timeout,
            max_retries=0,
        )
        self._embed_dim = 768  # fallback for sentence-transformers or random
        self._st_model = None  # lazy-loaded sentence-transformers model

    def generate_response(self,
                         prompt: str,
                         temperature: Optional[float] = None,
                         max_tokens: Optional[int] = None) -> str:
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature if temperature is not None else self.config.temperature,
            max_tokens=max_tokens or self.config.max_tokens,
        )
        return response.choices[0].message.content

    def evaluate_reasoning(self,
                          problem: str,
                          solution_steps: List[str]) -> float:
        prompt = f"""Problem: {problem}
Solution Steps:
{chr(10).join(f'{i+1}. {step}' for i, step in enumerate(solution_steps))}

Rate the quality of these solution steps from 0 to 1.
Respond with only the numerical rating, nothing else."""
        response = self.generate_response(prompt, max_tokens=10)
        try:
            return max(0.0, min(1.0, float(response.strip())))
        except ValueError:
            return 0.5

    def embed_text(self, text: str) -> List[float]:
        """Use sentence-transformers for local embeddings (no API key needed).

        Falls back to zero vector if sentence-transformers is not installed.
        Install with: pip install sentence-transformers
        """
        if self._st_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._st_model = SentenceTransformer("all-MiniLM-L6-v2")
                self._embed_dim = 384
            except ImportError:
                return [0.0] * self._embed_dim
        embedding = self._st_model.encode(text, convert_to_numpy=True)
        return embedding.tolist()


class ModelFactory:
    """Factory for creating LLM instances."""

    @staticmethod
    def create_model(model_type: str,
                    api_key: str = "",
                    config_path: Optional[str] = None,
                    **kwargs) -> LLMInterface:
        """Create a model instance based on type.

        model_type: "openai" | "anthropic" | "deepseek" | "ollama"
        For "ollama", api_key is ignored; pass ollama_model= and base_url= via kwargs.
        """
        name = model_type.lower()
        if name == "ollama":
            return OllamaModel(**kwargs)
        if config_path:
            if name == "openai":
                return OpenAIModel.from_config_file(config_path, api_key)
            elif name == "anthropic":
                return AnthropicModel.from_config_file(config_path, api_key)
            elif name == "deepseek":
                return DeepSeekModel.from_config_file(config_path, api_key)
        else:
            if name == "openai":
                return OpenAIModel(api_key)
            elif name == "anthropic":
                return AnthropicModel(api_key)
            elif name == "deepseek":
                return DeepSeekModel(api_key)

        raise ValueError(f"Unknown model type: {model_type}. Choose from: openai, anthropic, deepseek, ollama")
