from __future__ import annotations

import os

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


class VHEModel(nn.Module):
    """The single selected VHE-Net architecture."""

    def __init__(
        self,
        *,
        num_hosts: int,
        pretrained_model: str,
        host_similarity: torch.Tensor,
        embed_dim: int = 256,
        num_heads: int = 2,
        dropout: float = 0.3,
        load_pretrained_encoder: bool = True,
        empty_init: bool = False,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)

        encoder_config = AutoConfig.from_pretrained(
            pretrained_model,
            trust_remote_code=True,
        )
        if load_pretrained_encoder:
            self.llm = AutoModel.from_pretrained(
                pretrained_model,
                config=encoder_config,
                trust_remote_code=True,
            )
        else:
            self.llm = AutoModel.from_config(
                encoder_config,
                trust_remote_code=True,
            )
        self.llm.requires_grad_(False)
        self.llm.eval()
        hidden_size = int(getattr(encoder_config, "hidden_size", 2560))

        self.virus_projector = nn.Sequential(
            nn.Linear(hidden_size, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.host_embedding = nn.Embedding(num_hosts, embed_dim)
        self.host_transformer = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        expected_shape = (num_hosts, num_hosts)
        if tuple(host_similarity.shape) != expected_shape:
            raise ValueError(
                f"host_similarity must have shape {expected_shape}, "
                f"got {tuple(host_similarity.shape)}"
            )
        if empty_init:
            host_similarity = torch.empty(expected_shape, device="meta")
        else:
            host_similarity = torch.as_tensor(host_similarity, dtype=torch.float32)
        self.register_buffer("sim_matrix", host_similarity, persistent=True)
        self.sim_projector = nn.Sequential(
            nn.Linear(num_hosts, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.host_fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.key_projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.value_projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
            bias=True,
        )
        self.attention_residual = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.ln_cross_attn = nn.LayerNorm(embed_dim)
        self.ln_fusion = nn.LayerNorm(embed_dim * 4)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 4, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        if not empty_init:
            self._initialize_prediction_head()

    def _initialize_prediction_head(self) -> None:
        nn.init.normal_(self.host_embedding.weight, mean=0.0, std=0.02)
        for name, module in self.named_modules():
            if name.startswith("llm."):
                continue
            if isinstance(module, nn.Linear):
                if "attention" in name.lower() or "attn" in name.lower():
                    nn.init.xavier_uniform_(module.weight)
                else:
                    nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def train(self, mode: bool = True):
        super().train(mode)
        self.llm.eval()
        return self

    def encode_virus(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, num_fragments, sequence_length = input_ids.shape
        flat_input_ids = input_ids.reshape(-1, sequence_length)
        flat_mask = (
            attention_mask.reshape(-1, sequence_length)
            if attention_mask is not None
            else None
        )

        if flat_mask is not None:
            unique_rows, inverse_rows = torch.unique(
                torch.cat([flat_input_ids, flat_mask], dim=1),
                dim=0,
                return_inverse=True,
            )
            encoder_ids = unique_rows[:, :sequence_length]
            encoder_mask = unique_rows[:, sequence_length:]
        else:
            encoder_ids, inverse_rows = torch.unique(
                flat_input_ids,
                dim=0,
                return_inverse=True,
            )
            encoder_mask = None

        with torch.no_grad():
            outputs = self.llm(
                input_ids=encoder_ids,
                attention_mask=encoder_mask,
                output_hidden_states=False,
                output_attentions=False,
                return_dict=False,
            )
        hidden_state = outputs[0] if isinstance(outputs, (tuple, list)) else outputs.last_hidden_state
        cls_token = hidden_state[:, 0, :]
        fragment_features = self.virus_projector(cls_token)[inverse_rows]
        return fragment_features.reshape(batch_size, num_fragments, self.embed_dim)

    def score_encoded(
        self,
        virus_features: torch.Tensor,
        host_ids: torch.Tensor,
        fragment_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if host_ids.ndim == 0:
            host_ids = host_ids.reshape(1)
        batch_size = host_ids.size(0)

        if virus_features.ndim == 2:
            virus_features = virus_features.unsqueeze(0)
        if virus_features.size(0) == 1 and batch_size > 1:
            virus_features = virus_features.expand(batch_size, -1, -1)
        elif virus_features.size(0) != batch_size:
            raise ValueError(
                f"Virus feature batch must be 1 or {batch_size}; "
                f"got {virus_features.size(0)}."
            )

        num_fragments = virus_features.size(1)
        if fragment_mask is None:
            valid_fragments = torch.ones(
                batch_size,
                num_fragments,
                dtype=torch.bool,
                device=virus_features.device,
            )
        else:
            if fragment_mask.ndim != 2 or fragment_mask.size(1) != num_fragments:
                raise ValueError(
                    "fragment_mask must have shape (batch_size, num_fragments)."
                )
            valid_fragments = fragment_mask.to(
                device=virus_features.device,
                dtype=torch.bool,
            )
            if valid_fragments.size(0) == 1 and batch_size > 1:
                valid_fragments = valid_fragments.expand(batch_size, -1)
            elif valid_fragments.size(0) != batch_size:
                raise ValueError(
                    f"fragment_mask batch size must be 1 or {batch_size}; "
                    f"got {valid_fragments.size(0)}."
                )
        if not valid_fragments.any(dim=1).all():
            raise ValueError("Each virus must contain at least one valid fragment.")

        host_embedding = self.host_embedding(host_ids)
        host_representation = self.host_transformer(host_embedding)
        similarity_features = self.sim_projector(self.sim_matrix[host_ids])
        host_representation = self.host_fusion(
            torch.cat([host_representation, similarity_features], dim=1)
        )

        query = host_representation.unsqueeze(1)
        key = self.key_projection(virus_features)
        value = self.value_projection(virus_features)
        attention_output, _ = self.cross_attn(
            query=query,
            key=key,
            value=value,
            key_padding_mask=~valid_fragments,
            need_weights=False,
        )
        attention_output = attention_output.squeeze(1)
        attention_output = self.attention_residual(
            torch.cat([host_representation, attention_output], dim=1)
        )
        attention_output = self.ln_cross_attn(attention_output)

        valid_weights = valid_fragments.unsqueeze(-1).to(virus_features.dtype)
        global_virus_feature = (virus_features * valid_weights).sum(dim=1)
        global_virus_feature = global_virus_feature / valid_weights.sum(dim=1)
        interaction = host_representation * attention_output
        combined = torch.cat(
            [
                host_representation,
                attention_output,
                interaction,
                global_virus_feature,
            ],
            dim=1,
        )
        return self.classifier(self.ln_fusion(combined))

    def forward(
        self,
        input_ids: torch.Tensor,
        host_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        virus_features = self.encode_virus(input_ids, attention_mask)
        fragment_mask = attention_mask.any(dim=-1) if attention_mask is not None else None
        return self.score_encoded(virus_features, host_ids, fragment_mask)
