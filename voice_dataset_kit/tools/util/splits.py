import hashlib
from collections import defaultdict

def _dh(s: str) -> int:
    return int(hashlib.md5(s.encode('utf-8')).hexdigest(), 16)

def assign_splits_by_group(items, key_fn, ratios=(0.8, 0.1, 0.1)):
    buckets = {"train": [], "val": [], "test": []}
    groups = defaultdict(list)
    for x in items:
        groups[key_fn(x)].append(x)
    a, b, _ = [int(r*100) for r in ratios]
    for k in sorted(groups.keys()):
        h = _dh(str(k)) % 100
        s = 'train' if h < a else ('val' if h < a + b else 'test')
        buckets[s].extend(groups[k])
    return buckets
