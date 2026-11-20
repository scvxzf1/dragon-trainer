"""Qwen3.5 final-state text conditioning matching MOSS encode_prompt."""

import torch

from library.strategies.text import TextEncodingStrategy, TokenizeStrategy
from .cache_policy import MOSS_HIDDEN_SIZE, MOSS_MAX_LENGTH, resolve_tokenizer_path


class MossImageTokenizeStrategy(TokenizeStrategy):
    def __init__(self, qwen_path, max_length=MOSS_MAX_LENGTH):
        from transformers import AutoTokenizer

        if not 0 < max_length <= MOSS_MAX_LENGTH:
            raise ValueError("MOSS max_length must be between 1 and 4096")
        self.max_length = max_length
        self.tokenizer_path = resolve_tokenizer_path(qwen_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, local_files_only=True)
        if not self.tokenizer.chat_template:
            raise ValueError("MOSS tokenizer requires the Qwen3.5 chat template")
        if self.tokenizer.pad_token_id is None:
            raise ValueError("MOSS tokenizer requires a pad token")

    def tokenize(self, text):
        captions = [text] if isinstance(text, str) else list(text)
        if not captions or not all(isinstance(value, str) for value in captions):
            raise ValueError("MOSS captions must be a string or a nonempty list of strings")
        rows = []
        for caption in captions:
            formatted = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": caption}], tokenize=False,
                add_generation_prompt=True, enable_thinking=True,
            )
            encoded = self.tokenizer(
                formatted, return_tensors="pt", truncation=True, max_length=self.max_length,
            )
            rows.append(encoded["input_ids"][0])
        length = max(row.numel() for row in rows)
        ids = torch.full((len(rows), length), self.tokenizer.pad_token_id, dtype=torch.long)
        mask = torch.zeros_like(ids, dtype=torch.bool)
        for index, row in enumerate(rows):
            ids[index, :row.numel()] = row
            mask[index, :row.numel()] = True
        return [ids, mask]

    def tokenize_with_weights(self, text):
        raise ValueError("MOSS weighted captions are unsupported")


class MossImageTextEncodingStrategy(TextEncodingStrategy):
    @torch.no_grad()
    def encode_tokens(self, tokenize_strategy, models, tokens):
        model = models[0]
        if model is None:
            raise RuntimeError("MOSS text cache is missing; preprocess it with a text encoder before training")
        ids, mask = tokens
        device = next(model.parameters()).device
        rows = []
        for row_ids, row_mask in zip(ids, mask):
            valid_ids = row_ids[row_mask.bool()].unsqueeze(0).to(device)
            if not valid_ids.shape[1]:
                raise ValueError("MOSS text encoding requires at least one token per caption")
            output = model(
                input_ids=valid_ids, attention_mask=torch.ones_like(valid_ids),
                use_cache=False, return_dict=True,
            )
            row = output.last_hidden_state[0]
            if row.ndim != 2 or row.shape != (valid_ids.shape[1], MOSS_HIDDEN_SIZE):
                raise ValueError(f"MOSS requires final Qwen3.5 hidden states [L,4096], got {tuple(row.shape)}")
            rows.append(row)
        # Padding happens only after every GDN forward, matching the original pipeline.
        length = max(row.shape[0] for row in rows)
        hiddens = rows[0].new_zeros((len(rows), length, MOSS_HIDDEN_SIZE))
        output_mask = torch.zeros(hiddens.shape[:2], device=device, dtype=torch.bool)
        for index, row in enumerate(rows):
            hiddens[index, :row.shape[0]] = row
            output_mask[index, :row.shape[0]] = True
        return [hiddens, output_mask]

    def encode_tokens_with_weights(self, *args, **kwargs):
        raise ValueError("MOSS weighted captions are unsupported")

    def apply_caption_dropout_inplace(self, caption_dropout_rates, **kwargs):
        if torch.as_tensor(caption_dropout_rates).ne(0).any():
            raise ValueError("MOSS caption dropout is unsupported; set caption_dropout_rate=0")
