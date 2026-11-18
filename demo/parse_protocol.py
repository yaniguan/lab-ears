"""Rule-based baseline: find Amount, Temperature, Time and Speed spans in wet-lab
protocol text, and score them against the WLP gold annotations (Kulkarni et al., 2018).

Usage: python demo/parse_protocol.py path/to/WLP-Dataset/test
"""
import re
import sys
from pathlib import Path

NUM = r"(?:\d+(?:[.,]\d+)?(?:\s*(?:-|–|to)\s*\d+(?:[.,]\d+)?)?)"
RULES = {
    "Amount": NUM + r"\s*(?:µl|μl|ul|ml|mL|µL|μL|uL|L|l|nl|nL|mg|µg|μg|ug|ng|g|kg|pmol|nmol|µmol|mmol|mol|volumes?|vol)\b",
    "Temperature": NUM + r"\s*(?:°\s*C|º\s*C|˚\s*C|° C|degrees?\s*C(?:elsius)?|C\b|°F|K\b)|\b(?:room temperature|RT|on ice|ice)\b",
    "Time": NUM + r"\s*(?:h|hr|hrs|hours?|min|mins|minutes?|s|sec|secs|seconds?|days?|overnight)\b|\bovernight\b",
    "Speed": NUM + r"\s*(?:x\s*g|×\s*g|g\b|rpm|RPM|xg)",
}


def find_spans(text):
    spans = set()
    for label, pat in RULES.items():
        for m in re.finditer(pat, text, flags=re.IGNORECASE if label != "Speed" else 0):
            spans.add((label, m.start(), m.end()))
    # Speed "1000g" also matches Amount "g": keep the Speed one
    speed = {(s, e) for l, s, e in spans if l == "Speed"}
    return {(l, s, e) for l, s, e in spans if not (l == "Amount" and (s, e) in speed)}


def gold_spans(ann_path):
    out = set()
    for line in ann_path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("T"):
            continue
        _, info, _ = line.split("\t", 2)
        parts = info.split()
        if parts[0] in RULES and ";" not in info:
            out.add((parts[0], int(parts[1]), int(parts[2])))
    return out


def overlaps(a, b):
    return a[0] == b[0] and a[1] < b[2] and b[1] < a[2]


def main(folder):
    files = sorted(Path(folder).glob("*.ann"))
    stats = {k: [0, 0, 0, 0] for k in RULES}  # tp_exact, tp_overlap, n_pred, n_gold
    for ann in files:
        text = ann.with_suffix(".txt").read_text(encoding="utf-8")
        pred, gold = find_spans(text), gold_spans(ann)
        for k in RULES:
            p = [x for x in pred if x[0] == k]
            g = [x for x in gold if x[0] == k]
            stats[k][0] += len(set(p) & set(g))
            stats[k][1] += sum(any(overlaps(x, y) for y in g) for x in p)
            stats[k][2] += len(p)
            stats[k][3] += len(g)
    print(f"protocols: {len(files)}")
    print(f"{'label':<12}{'gold':>6}{'pred':>6}{'P_exact':>9}{'R_exact':>9}{'F1_exact':>9}{'P_overlap':>11}")
    for k, (te, to, n_p, n_g) in stats.items():
        p = te / n_p if n_p else 0
        r = te / n_g if n_g else 0
        f = 2 * p * r / (p + r) if p + r else 0
        po = to / n_p if n_p else 0
        print(f"{k:<12}{n_g:>6}{n_p:>6}{p:>9.3f}{r:>9.3f}{f:>9.3f}{po:>11.3f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/WLP-Dataset/test")
