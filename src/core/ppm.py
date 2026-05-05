"""
Process Preference Model (PPM) for evaluating reasoning steps.

Innovations over the baseline single-step encoder:
  1. ContextAwarePPM   — joint (problem, step) dual-encoder; scores steps
                         relative to the problem being solved, not in isolation.
  2. Soft margin loss  — margin scales with the actual reward gap between pairs
                         instead of a fixed constant of 1.0.
  3. Hard negative mining — per-batch, the highest-cosine-similarity
                            non-preferred steps are substituted to surface the
                            hardest confusing examples.
  4. Dropout + gradient clipping — prevents overfitting on small training sets.
  5. Cosine LR annealing — stable convergence without manual schedule tuning.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class PPMConfig:
    input_dim: int = 384
    hidden_dim: int = 256
    learning_rate: float = 0.001
    batch_size: int = 32
    # Innovation 4: regularization
    dropout_rate: float = 0.1
    # Innovation 5: cosine LR annealing
    use_scheduler: bool = True
    scheduler_T_max: int = 100
    # Innovation 3: hard negative mining
    use_hard_negatives: bool = True
    hard_negative_ratio: float = 0.3


# ── Building blocks ───────────────────────────────────────────────────────────

class StepEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout_rate: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


# ── Base model ────────────────────────────────────────────────────────────────

class ProcessPreferenceModel(nn.Module):
    """Step-only PPM (baseline).

    Encodes each reasoning step independently and predicts a quality score.
    For context-sensitive scoring see ContextAwarePPM.
    """

    def __init__(self, config: Optional[PPMConfig] = None):
        super().__init__()
        self.config = config or PPMConfig()

        self.step_encoder = StepEncoder(
            self.config.input_dim,
            self.config.hidden_dim,
            dropout_rate=self.config.dropout_rate,
        )

        self.value_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim // 2),
            nn.LayerNorm(self.config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.config.dropout_rate),
            nn.Linear(self.config.hidden_dim // 2, 1),
        )

        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=self.config.learning_rate
        )

    @classmethod
    def from_config_file(cls, config_path: str) -> "ProcessPreferenceModel":
        with open(config_path, "r") as f:
            config_data = json.load(f)
        config = PPMConfig(**config_data["ppm"])
        return cls(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.value_head(self.step_encoder(x))

    # ── Embedding helpers ─────────────────────────────────────────────────────

    def _embed_steps(
        self,
        steps: List[str],
        embedder: Any,
        problems: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """Embed a list of steps. `problems` is accepted for interface
        compatibility with ContextAwarePPM but is ignored here."""
        return torch.FloatTensor([embedder.embed_text(s) for s in steps])

    # ── Inference ─────────────────────────────────────────────────────────────

    def evaluate_step(self, step: str, embedder: Any, problem: str = "") -> float:
        """Score a single reasoning step.

        `problem` is accepted for interface compatibility with ContextAwarePPM
        but unused in this base class — only the step text is encoded.
        """
        self.eval()
        with torch.no_grad():
            tensor = torch.FloatTensor(embedder.embed_text(step)).unsqueeze(0)
            return self(tensor).item()

    # ── Training ──────────────────────────────────────────────────────────────

    def train_step(
        self,
        preferred_steps: List[str],
        non_preferred_steps: List[str],
        embedder: Any,
        preferred_rewards: Optional[List[float]] = None,
        non_preferred_rewards: Optional[List[float]] = None,
        problems: Optional[List[str]] = None,
    ) -> float:
        """One gradient step with soft reward-proportional margin loss (Innovation 2).

        Loss:  relu(v_neg − v_pos + margin).mean()
          - No rewards supplied  → margin = 1.0  (original behavior)
          - Rewards supplied     → margin = clamp(r_pos − r_neg, min=0.1)
            Pairs with a larger quality gap contribute a larger margin,
            making high-confidence preferences push harder during training.
        """
        self.train()

        pref_embs = self._embed_steps(preferred_steps, embedder, problems)
        nonpref_embs = self._embed_steps(non_preferred_steps, embedder, problems)

        if (
            self.config.use_hard_negatives
            and pref_embs.size(0) >= 4
            and nonpref_embs.size(0) >= 4
        ):
            n_hard = max(1, int(pref_embs.size(0) * self.config.hard_negative_ratio))
            pref_embs, nonpref_embs = _apply_hard_negatives(pref_embs, nonpref_embs, n_hard)

        pref_vals = self(pref_embs)
        nonpref_vals = self(nonpref_embs)

        # Reward-proportional margin
        if preferred_rewards and non_preferred_rewards:
            n = min(len(preferred_rewards), len(non_preferred_rewards), pref_vals.size(0))
            margins = torch.FloatTensor(
                [max(0.1, preferred_rewards[i] - non_preferred_rewards[i]) for i in range(n)]
            ).unsqueeze(1)
            if n < pref_vals.size(0):
                extra = torch.ones(pref_vals.size(0) - n, 1)
                margins = torch.cat([margins, extra], dim=0)
        else:
            margins = torch.ones(pref_vals.size(0), 1)

        loss = F.relu(nonpref_vals - pref_vals + margins).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()
        return loss.item()

    # ── Persistence ───────────────────────────────────────────────────────────

    def save_model(self, path: str) -> None:
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.config,
                "class": self.__class__.__name__,
            },
            path,
        )

    def load_model(self, path: str) -> None:
        checkpoint = torch.load(path, weights_only=False)
        self.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.config = checkpoint["config"]


# ── Innovation 1: Context-Aware PPM ──────────────────────────────────────────

class ContextAwarePPM(ProcessPreferenceModel):
    """Context-Aware PPM: jointly encodes (problem, step) pairs (Innovation 1).

    Architecture
    ────────────
      problem_emb (D) ──┐
                         ├─ concat → (2D) → StepEncoder(2D→H) → ValueHead(H→1)
      step_emb    (D) ──┘

    Unlike the base PPM that scores steps in isolation, ContextAwarePPM
    conditions step quality on the specific problem being solved.  This lets
    the model distinguish steps that are valid in one context but irrelevant in
    another (e.g. chain rule is helpful for calculus, useless for combinatorics).

    Parameters
    ──────────
    embedding_dim : int
        Dimension of a single embedding vector (e.g. 384 for all-MiniLM-L6-v2).
        The PPM input_dim is automatically set to 2 × embedding_dim.
    config : PPMConfig, optional
        All fields except input_dim are respected; input_dim is overridden.
    """

    def __init__(self, embedding_dim: int = 384, config: Optional[PPMConfig] = None):
        base = config or PPMConfig()
        merged = PPMConfig(
            input_dim=embedding_dim * 2,
            hidden_dim=base.hidden_dim,
            learning_rate=base.learning_rate,
            batch_size=base.batch_size,
            dropout_rate=base.dropout_rate,
            use_scheduler=base.use_scheduler,
            scheduler_T_max=base.scheduler_T_max,
            use_hard_negatives=base.use_hard_negatives,
            hard_negative_ratio=base.hard_negative_ratio,
        )
        super().__init__(merged)
        self.embedding_dim = embedding_dim

    # ── Pair encoding ─────────────────────────────────────────────────────────

    def _encode_pair(self, step: str, embedder: Any, problem: str = "") -> List[float]:
        step_emb = embedder.embed_text(step)
        prob_emb = embedder.embed_text(problem) if problem else [0.0] * len(step_emb)
        return prob_emb + step_emb

    def _embed_steps(
        self,
        steps: List[str],
        embedder: Any,
        problems: Optional[List[str]] = None,
    ) -> torch.Tensor:
        if problems and len(problems) == len(steps):
            return torch.FloatTensor(
                [self._encode_pair(s, embedder, p) for s, p in zip(steps, problems)]
            )
        return torch.FloatTensor([self._encode_pair(s, embedder) for s in steps])

    def evaluate_step(self, step: str, embedder: Any, problem: str = "") -> float:
        """Score step relative to the problem context."""
        self.eval()
        with torch.no_grad():
            combined = self._encode_pair(step, embedder, problem)
            tensor = torch.FloatTensor(combined).unsqueeze(0)
            return self(tensor).item()


# ── Innovation 3: Hard Negative Mining ───────────────────────────────────────

def _apply_hard_negatives(
    pref_embs: torch.Tensor,
    nonpref_embs: torch.Tensor,
    n_hard: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Replace the first n_hard non-preferred rows with hard negatives (Innovation 3).

    Hard negatives are non-preferred steps with the *highest* cosine similarity
    to their paired preferred step.  These examples look like good steps but
    lead to lower-quality trajectories — the hardest cases for the PPM to learn.
    Surfacing them per-batch sharpens the decision boundary without requiring
    a separate offline mining pass.
    """
    with torch.no_grad():
        pref_norm = F.normalize(pref_embs, dim=1)
        nonpref_norm = F.normalize(nonpref_embs, dim=1)
        sim = pref_norm @ nonpref_norm.T   # (n_pref, n_nonpref)
        hard_idx = sim.argmax(dim=1)       # most-similar nonpref for each pref row

    n = min(n_hard, pref_embs.size(0), nonpref_embs.size(0))
    new_nonpref = nonpref_embs.clone()
    new_nonpref[:n] = nonpref_embs[hard_idx[:n]]
    return pref_embs, new_nonpref


# ── Trainer ───────────────────────────────────────────────────────────────────

class PPMTrainer:
    def __init__(self, model: ProcessPreferenceModel):
        self.model = model
        self._scheduler: Optional[torch.optim.lr_scheduler.CosineAnnealingLR] = None

    def train(
        self,
        training_data: List[Dict[str, Any]],
        embedder: Any,
        num_epochs: int = 100,
        validation_data: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, List[float]]:
        """Train with cosine LR annealing (Innovation 5).

        Expects training_data dicts with keys:
          required: "preferred", "non_preferred"
          optional: "preferred_reward", "non_preferred_reward", "problem"
        """
        train_losses: List[float] = []
        val_losses: List[float] = []

        # Innovation 5: cosine LR annealing
        if self.model.config.use_scheduler and num_epochs > 1:
            self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.model.optimizer,
                T_max=min(num_epochs, self.model.config.scheduler_T_max),
                eta_min=self.model.config.learning_rate * 0.01,
            )

        for _epoch in range(num_epochs):
            epoch_loss = 0.0
            batches = self._create_batches(training_data, self.model.config.batch_size)
            for batch in batches:
                epoch_loss += self.model.train_step(
                    batch["preferred"],
                    batch["non_preferred"],
                    embedder,
                    preferred_rewards=batch.get("preferred_rewards"),
                    non_preferred_rewards=batch.get("non_preferred_rewards"),
                    problems=batch.get("problems"),
                )
            if self._scheduler is not None:
                self._scheduler.step()
            train_losses.append(epoch_loss / max(1, len(batches)))

            if validation_data:
                val_losses.append(self._validate(validation_data, embedder))

        return {"train_losses": train_losses, "val_losses": val_losses}

    def _create_batches(
        self, data: List[Dict[str, Any]], batch_size: int
    ) -> List[Dict[str, Any]]:
        batches: List[Dict[str, Any]] = []
        for i in range(0, len(data), batch_size):
            chunk = data[i : i + batch_size]
            batch: Dict[str, Any] = {
                "preferred": [d["preferred"] for d in chunk],
                "non_preferred": [d["non_preferred"] for d in chunk],
            }
            if all("preferred_reward" in d for d in chunk):
                batch["preferred_rewards"] = [d["preferred_reward"] for d in chunk]
                batch["non_preferred_rewards"] = [d["non_preferred_reward"] for d in chunk]
            if all("problem" in d for d in chunk):
                batch["problems"] = [d.get("problem", "") for d in chunk]
            batches.append(batch)
        return batches

    def _validate(
        self, validation_data: List[Dict[str, Any]], embedder: Any
    ) -> float:
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in self._create_batches(validation_data, self.model.config.batch_size):
                probs = batch.get("problems")
                pref_embs = self.model._embed_steps(batch["preferred"], embedder, probs)
                nonpref_embs = self.model._embed_steps(batch["non_preferred"], embedder, probs)
                pref_vals = self.model(pref_embs)
                nonpref_vals = self.model(nonpref_embs)
                loss = F.relu(nonpref_vals - pref_vals + 1.0).mean()
                total_loss += loss.item()
        return total_loss / max(1, len(validation_data))
