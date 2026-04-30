"""Block-level chunking for HTML classification."""

from __future__ import annotations

import re

SEP_TOKEN = "<|sep|>"


def extract_blocks(simplified_html: str) -> list[str]:
    """Split simplified HTML into individual block strings at _item_id boundaries."""
    pattern = re.compile(r'_item_id="(\d+)"')
    matches = list(pattern.finditer(simplified_html))

    if not matches:
        return []

    blocks = []
    for i, m in enumerate(matches):
        attr_start = m.start()
        tag_start = simplified_html.rfind("<", 0, attr_start)

        if i + 1 < len(matches):
            next_attr_start = matches[i + 1].start()
            next_tag_start = simplified_html.rfind("<", 0, next_attr_start)
            block_html = simplified_html[tag_start:next_tag_start]
        else:
            block_html = simplified_html[tag_start:]

        blocks.append(block_html.strip())

    return blocks


def tokenize_blocks(blocks: list[str], tokenizer) -> list[list[int]]:
    """Tokenize each block independently."""
    return [tokenizer.encode(block, add_special_tokens=False) for block in blocks]


def pack_chunks(
    block_token_ids: list[list[int]],
    max_tokens: int = 8192,
    sep_token_id: int | None = None,
    bos_token_id: int | None = None,
    eos_token_id: int | None = None,
) -> list[tuple[list[int], list[int]]]:
    """Pack blocks into chunks within max_tokens budget.

    Each chunk: [BOS] block_0 [SEP] block_1 [SEP] ... block_N [SEP] [EOS]
    Returns list of (token_ids, block_indices) tuples.
    """
    if sep_token_id is None:
        raise ValueError("sep_token_id is required")

    chunks = []
    current_tokens = []
    current_block_indices = []

    overhead = (1 if bos_token_id is not None else 0) + (1 if eos_token_id is not None else 0)
    budget = max_tokens - overhead
    current_len = 0

    for block_idx, block_toks in enumerate(block_token_ids):
        block_cost = len(block_toks) + 1

        if current_len + block_cost > budget:
            if current_tokens:
                chunk_ids = []
                if bos_token_id is not None:
                    chunk_ids.append(bos_token_id)
                chunk_ids.extend(current_tokens)
                if eos_token_id is not None:
                    chunk_ids.append(eos_token_id)
                chunks.append((chunk_ids, current_block_indices))

            current_tokens = []
            current_block_indices = []
            current_len = 0

            if block_cost > budget:
                truncated = block_toks[: budget - 1]
                current_tokens.extend(truncated)
                current_tokens.append(sep_token_id)
                current_block_indices.append(block_idx)
                current_len = len(truncated) + 1
                continue

        current_tokens.extend(block_toks)
        current_tokens.append(sep_token_id)
        current_block_indices.append(block_idx)
        current_len += block_cost

    if current_tokens:
        chunk_ids = []
        if bos_token_id is not None:
            chunk_ids.append(bos_token_id)
        chunk_ids.extend(current_tokens)
        if eos_token_id is not None:
            chunk_ids.append(eos_token_id)
        chunks.append((chunk_ids, current_block_indices))

    return chunks
