# Benchmarks

## WebMainBench (English-only, 6,647 pages)

Evaluation on the full English subset of [WebMainBench](https://github.com/nickochar/WebMainBench) using ROUGE-5 F1 (5-gram overlap, whitespace tokenized).

Ground truth: `convert_main_content` field (html2text with bodywidth=0, ignore_links=True, ignore_images=True).

| Method | Params | All | Simple | Mid | Hard | Empty | P | R | pg/s |
|--------|--------|-----|--------|-----|------|-------|---|---|------|
| **Pulpie Large** | 2.1B | **0.873** | 0.914 | **0.878** | **0.827** | 21 | 0.865 | 0.917 | 1.2 |
| Dripper | 0.6B | 0.864 | **0.914** | 0.865 | 0.818 | 135 | 0.860 | 0.901 | 3.5 |
| **Pulpie Base** | 610M | 0.863 | 0.906 | 0.868 | 0.818 | 36 | 0.858 | 0.906 | 2.1 |
| **Pulpie Small** | 210M | 0.862 | 0.906 | 0.868 | 0.813 | 45 | 0.854 | 0.910 | 4.0 |
| magic-html | — | 0.700 | 0.773 | 0.697 | 0.637 | 384 | 0.778 | 0.704 | 14.7 |
| Raw html2text | — | 0.620 | 0.779 | 0.605 | 0.491 | 0 | 0.515 | 0.943 | — |
| Trafilatura | — | 0.619 | 0.721 | 0.619 | 0.526 | 16 | 0.688 | 0.610 | 17.9 |

**Difficulty distribution**: simple=1,888 | mid=2,720 | hard=2,039

### Notes

- **pg/s** (this table): pages/second during the full WebMainBench quality run on a
  single A100 80GB, with Pulpie using a naive sequential loop (no batching). The
  batched throughput numbers in the section below are the ones to use for
  speed/cost — batching roughly 6x's Pulpie Small on A100 (4.0 → 25.7).
- **Dripper** ([MinerU-HTML v1.1](https://huggingface.co/opendatalab/MinerU-HTML-v1.1-hunyuan0.5B-compact)) uses vLLM with guided regex decoding for batched inference.
- **Dripper** ([MinerU-HTML v1.1](https://huggingface.co/opendatalab/MinerU-HTML-v1.1-hunyuan0.5B-compact)) uses vLLM with guided regex decoding for batched inference.
- **Empty**: pages where the method produced no output (context overflow for Dripper, no blocks detected for Pulpie).
- **magic-html**: [opendatalab/magic-html](https://github.com/opendatalab/magic-html) rule-based extractor.
- **Trafilatura**: [trafilatura](https://github.com/adbar/trafilatura) with `include_tables=True`.
- **Raw html2text**: full HTML passed directly through html2text (no extraction).

### Throughput (500 real Common Crawl pages)

L4 throughput:

| Method | Pages/sec (L4) | Pages/sec (A100, batched) |
|--------|----------------|---------------------------|
| Pulpie Small | **13.7** | 25.7 |
| Pulpie Base | 3.9 | 7.7 |
| Pulpie Large | 1.3 | 3.5 |
| Dripper | 0.68 | 3.6 |

Pulpie Small runs **20x faster than Dripper on L4** and **7.1x faster on A100**.

### Cost comparison (1B pages, projected)

L4 at $0.39/hr, using the throughputs above:

| Setup | Pages/sec | GPU-hours / 1B | Cost / 1B pages |
|-------|-----------|----------------|-----------------|
| Pulpie Small on L4 | 13.7 | 20,300 | **~$7,900** |
| Pulpie Base on L4 | 3.9 | 71,200 | ~$28,000 |
| Pulpie Large on L4 | 1.3 | 214,000 | ~$83,000 |
| Dripper on L4 | 0.68 | 408,000 | ~$159,000 |

Pulpie Small matches Dripper quality (0.862 vs 0.864) at **~20x lower cost** on an L4.

Full analysis: [Pulpie: Pareto-Optimal Models for Cleaning the Web](https://usefeyn.com/blog/pulpie-pareto-optimal-models-for-cleaning-the-web/).

### Hardware

- GPU: NVIDIA A100 80GB
- Framework: PyTorch 2.6 + transformers 4.57
- vLLM 0.11.1 (for Dripper)

### Reproducibility

```bash
python eval/bench_all_methods.py \
    --methods pulpie-small,pulpie-base,pulpie-large,dripper,trafilatura,magic-html,raw-h2t \
    --limit 0 \
    --device cuda:0
```
