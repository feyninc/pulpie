"""Test: does stripping <marked-tail> from HTML improve ROUGE?"""
import json, subprocess, tempfile, os, re
from collections import Counter

BENCH_PATH = "data/webmainbench.jsonl"
HBIRD_BIN = "target/release/hummingbird"

def ngrams(tokens, n):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

def rouge5(ref, pred):
    rt = ref.split(); pt = pred.split()
    if not rt or not pt: return 0.
    rn = Counter(ngrams(rt, 5)); pn = Counter(ngrams(pt, 5))
    if not rn or not pn: return 0.
    ov = sum((rn & pn).values())
    p = ov / max(sum(pn.values()), 1)
    r = ov / max(sum(rn.values()), 1)
    if p + r == 0: return 0.
    return 2*p*r/(p+r)

def strip_annotations(html):
    # Remove <marked-tail>...</marked-tail> content
    html = re.sub(r'<marked-tail[^>]*>.*?</marked-tail>', '', html, flags=re.DOTALL)
    # Also clean annotation attributes (they're benign but cleanup)
    html = re.sub(r'\s+data-anno-uid="[^"]*"', '', html)
    html = re.sub(r'\s+cc-select="[^"]*"', '', html)
    html = re.sub(r'\s+class="mark-selected"', '', html)
    # Unwrap <marked-text> (keep its content)
    html = re.sub(r'</?marked-text[^>]*>', '', html)
    return html

def extract(html):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
        f.write(html)
        tmp = f.name
    try:
        r = subprocess.run([HBIRD_BIN, tmp], capture_output=True, text=True, timeout=30)
        return r.stdout.strip()
    except:
        return ""
    finally:
        os.unlink(tmp)

with open(BENCH_PATH) as f:
    lines = f.readlines()

scores_normal = []
scores_stripped = []

for i in range(200):
    rec = json.loads(lines[i])
    html = rec.get("html", "")
    ref = rec.get("convert_main_content", "")
    if not html or not ref:
        continue

    # Normal
    pred_normal = extract(html)
    scores_normal.append(rouge5(ref, pred_normal) if pred_normal else 0.0)

    # Stripped
    clean_html = strip_annotations(html)
    pred_stripped = extract(clean_html)
    scores_stripped.append(rouge5(ref, pred_stripped) if pred_stripped else 0.0)

    if (i+1) % 50 == 0:
        n = len(scores_normal)
        print(f"  {i+1}/200: normal={sum(scores_normal)/n:.4f} stripped={sum(scores_stripped)/n:.4f}", flush=True)

n = len(scores_normal)
print(f"\n200-page comparison:")
print(f"  Normal HTML:    {sum(scores_normal)/n:.4f}")
print(f"  Stripped annot: {sum(scores_stripped)/n:.4f}")
diff = (sum(scores_stripped) - sum(scores_normal)) / n
print(f"  Difference:     {diff:+.4f}")
