"""Block-level chunking for HTML classification.

Splits simplified HTML into individual blocks, measures token counts,
and packs them into chunks that fit within a max token budget.

Each chunk is: block_1_html <|sep|> block_2_html <|sep|> ... <|sep|> block_N_html
Classification happens at each <|sep|> position.
"""

import re
from typing import List, Tuple

SEP_TOKEN = "<|sep|>"


def extract_blocks(simplified_html: str) -> List[str]:
    """Split simplified HTML into individual block strings.

    Each block is the HTML content between consecutive _item_id markers.
    Returns a list of block HTML strings (including the _item_id attribute).
    """
    pattern = re.compile(r'_item_id="(\d+)"')
    matches = list(pattern.finditer(simplified_html))

    if not matches:
        return []

    blocks = []
    for i, m in enumerate(matches):
        # Find a good split point: look backwards for the nearest '<' before _item_id
        # which is the opening of the block element's tag
        attr_start = m.start()
        tag_start = simplified_html.rfind('<', 0, attr_start)

        if i + 1 < len(matches):
            next_attr_start = matches[i + 1].start()
            next_tag_start = simplified_html.rfind('<', 0, next_attr_start)
            block_html = simplified_html[tag_start:next_tag_start]
        else:
            block_html = simplified_html[tag_start:]

        blocks.append(block_html.strip())

    return blocks


def tokenize_blocks(blocks: List[str], tokenizer) -> List[List[int]]:
    """Tokenize each block independently. Returns list of token ID lists."""
    return [
        tokenizer.encode(block, add_special_tokens=False)
        for block in blocks
    ]


def pack_chunks(
    block_token_ids: List[List[int]],
    max_tokens: int = 8192,
    sep_token_id: int = None,
    bos_token_id: int = None,
    eos_token_id: int = None,
) -> List[Tuple[List[int], List[int]]]:
    """Greedily pack blocks into chunks within max_tokens budget.

    Each chunk is formatted as:
        [BOS] block_0 [SEP] block_1 [SEP] ... block_N [SEP] [EOS]

    The [SEP] after each block is the classification position for that block.

    Returns:
        List of (token_ids, block_indices) tuples.
        block_indices maps each [SEP] in the chunk to its global block index.
    """
    if sep_token_id is None:
        raise ValueError("sep_token_id is required")

    chunks = []
    current_tokens = []
    current_block_indices = []

    # Account for BOS/EOS overhead
    overhead = 0
    if bos_token_id is not None:
        overhead += 1
    if eos_token_id is not None:
        overhead += 1

    budget = max_tokens - overhead
    current_len = 0

    for block_idx, block_toks in enumerate(block_token_ids):
        # Cost of adding this block: block tokens + 1 sep token
        block_cost = len(block_toks) + 1  # +1 for <|sep|>

        if current_len + block_cost > budget:
            # Flush current chunk
            if current_tokens:
                chunk_ids = []
                if bos_token_id is not None:
                    chunk_ids.append(bos_token_id)
                chunk_ids.extend(current_tokens)
                if eos_token_id is not None:
                    chunk_ids.append(eos_token_id)
                chunks.append((chunk_ids, current_block_indices))

            # Start new chunk
            current_tokens = []
            current_block_indices = []
            current_len = 0

            # If single block exceeds budget, truncate it
            if block_cost > budget:
                truncated = block_toks[:budget - 1]  # leave room for sep
                current_tokens.extend(truncated)
                current_tokens.append(sep_token_id)
                current_block_indices.append(block_idx)
                current_len = len(truncated) + 1
                continue

        current_tokens.extend(block_toks)
        current_tokens.append(sep_token_id)
        current_block_indices.append(block_idx)
        current_len += block_cost

    # Flush remaining
    if current_tokens:
        chunk_ids = []
        if bos_token_id is not None:
            chunk_ids.append(bos_token_id)
        chunk_ids.extend(current_tokens)
        if eos_token_id is not None:
            chunk_ids.append(eos_token_id)
        chunks.append((chunk_ids, current_block_indices))

    return chunks


def chunk_page(
    simplified_html: str,
    tokenizer,
    max_tokens: int = 8192,
    sep_token_id: int = None,
) -> List[Tuple[List[int], List[int]]]:
    """Full pipeline: extract blocks, tokenize, pack into chunks.

    Args:
        simplified_html: Output of simplify_html()
        tokenizer: HuggingFace tokenizer with <|sep|> token added
        max_tokens: Maximum tokens per chunk
        sep_token_id: Token ID for <|sep|>

    Returns:
        List of (token_ids, block_indices) tuples.
    """
    blocks = extract_blocks(simplified_html)
    if not blocks:
        return []

    block_token_ids = tokenize_blocks(blocks, tokenizer)

    return pack_chunks(
        block_token_ids,
        max_tokens=max_tokens,
        sep_token_id=sep_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
