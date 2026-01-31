"""
Qwen3 Model Architecture.

Contains the MLP, decoder layers, and full model classes.
"""

from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .config import Qwen3Config
    from .attention import Qwen3RMSNorm, Qwen3Attention
except ImportError:
    from config import Qwen3Config
    from attention import Qwen3RMSNorm, Qwen3Attention


class Qwen3MLP(nn.Module):
    """MLP module for Qwen3 with SwiGLU activation."""
    
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Qwen3DecoderLayer(nn.Module):
    """A single transformer decoder layer for Qwen3."""
    
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        self.self_attn = Qwen3Attention(config, layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Self attention with residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        hidden_states, attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states
        
        # MLP with residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states, attn_weights, present_key_value


class Qwen3Model(nn.Module):
    """The Qwen3 model outputting raw hidden-states without any head on top."""
    
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            Qwen3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> Tuple:
        batch_size, seq_len = input_ids.shape
        
        # Get embeddings
        hidden_states = self.embed_tokens(input_ids)
        
        # Create position_ids if not provided
        if position_ids is None:
            past_seen_tokens = past_key_values[0][0].shape[2] if past_key_values is not None else 0
            position_ids = torch.arange(
                past_seen_tokens, past_seen_tokens + seq_len, dtype=torch.long, device=input_ids.device
            ).unsqueeze(0).expand(batch_size, -1)
        
        # Create causal mask
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=input_ids.device)
        
        causal_mask = self._prepare_causal_mask(
            attention_mask, hidden_states, past_key_values
        )
        
        # Initialize outputs
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        next_cache = () if use_cache else None
        
        # Forward through layers
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            past_key_value = past_key_values[idx] if past_key_values is not None else None
            
            hidden_states, attn_weights, present_key_value = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            
            if use_cache:
                next_cache += (present_key_value,)
            
            if output_attentions:
                all_attentions += (attn_weights,)
        
        # Final layer norm
        hidden_states = self.norm(hidden_states)
        
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        
        return hidden_states, next_cache, all_hidden_states, all_attentions
    
    def _prepare_causal_mask(
        self,
        attention_mask: torch.Tensor,
        hidden_states: torch.Tensor,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]],
    ) -> torch.Tensor:
        """Prepare the causal attention mask."""
        batch_size, seq_len = attention_mask.shape
        dtype = hidden_states.dtype
        device = hidden_states.device
        
        past_seen_tokens = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        total_len = past_seen_tokens + seq_len
        
        # Create causal mask
        causal_mask = torch.zeros((seq_len, total_len), dtype=dtype, device=device)
        
        if seq_len > 1:
            mask = torch.full((seq_len, seq_len), float("-inf"), dtype=dtype, device=device)
            mask = torch.triu(mask, diagonal=1)
            causal_mask[:, past_seen_tokens:] = mask
        
        # Expand for batch and heads
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        
        # Apply padding mask if needed
        if attention_mask is not None and 0 in attention_mask:
            expanded_mask = attention_mask[:, None, None, :].to(dtype=torch.float32)
            inverted_mask = (1.0 - expanded_mask) * torch.finfo(dtype).min
            causal_mask = causal_mask + inverted_mask.to(dtype=dtype)
        
        return causal_mask


class Qwen3ForCausalLM(nn.Module):
    """Qwen3 model with a causal language modeling head."""
    
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Tie weights if specified
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights using the model's initializer_range."""
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> dict:
        hidden_states, next_cache, all_hidden_states, all_attentions = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        
        return {
            "loss": loss,
            "logits": logits,
            "past_key_values": next_cache,
            "hidden_states": all_hidden_states,
            "attentions": all_attentions,
        }
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Simple generation method with sampling."""
        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id
        if pad_token_id is None:
            pad_token_id = eos_token_id
        
        generated = input_ids
        past_key_values = None
        
        for _ in range(max_new_tokens):
            if past_key_values is None:
                outputs = self.forward(input_ids=generated, use_cache=True)
            else:
                outputs = self.forward(
                    input_ids=generated[:, -1:],
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            
            logits = outputs["logits"][:, -1, :]
            past_key_values = outputs["past_key_values"]
            
            if temperature > 0:
                logits = logits / temperature
            
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
                logits = torch.full_like(logits, float("-inf"))
                logits.scatter_(-1, top_k_indices, top_k_logits)
            
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float("-inf")
            
            if do_sample:
                probs = F.softmax(logits, dim=-1)
                probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
                if probs.sum() == 0:
                    probs = torch.ones_like(probs) / probs.shape[-1]
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            generated = torch.cat([generated, next_token], dim=-1)
            
            if (next_token == eos_token_id).all():
                break
        
        return generated
    
    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> "Qwen3ForCausalLM":
        """Load a pretrained Qwen3 model from a local directory."""
        import os
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        config_path = os.path.join(model_path, "config.json")
        config = Qwen3Config.from_json(config_path)
        
        model = cls(config)
        
        weights_path = os.path.join(model_path, "model.safetensors")
        if os.path.exists(weights_path):
            from safetensors.torch import load_file
            state_dict = load_file(weights_path)
        else:
            weights_path = os.path.join(model_path, "pytorch_model.bin")
            state_dict = torch.load(weights_path, map_location="cpu")
        
        model.load_state_dict(state_dict, strict=False)
        model = model.to(device=device, dtype=dtype)
        model.eval()
        
        return model


def load_tokenizer(model_path: str):
    """Load the tokenizer for Qwen3."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
