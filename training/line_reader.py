"""Train and evaluate a letter reader on book-disjoint print crops."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random

import numpy as np
from PIL import Image


def glyphs(rows, data):
    by_page = defaultdict(list)
    for row in rows:
        by_page[row["image"]].append(row)
    for path, lines in by_page.items():
        with Image.open(data / path) as image:
            gray = image.convert("L")
        for row in lines:
            for word in row["words"]:
                for char in word["chars"]:
                    c = char["c"]
                    if not c.isascii() or not c.isalpha():
                        continue
                    crop = gray.crop(char["box"]).resize((16, 24), Image.Resampling.BILINEAR)
                    raw = np.asarray(crop, dtype=np.float32)
                    feature = (raw.mean() - raw).ravel()
                    norm = np.linalg.norm(feature)
                    if norm > 0:
                        yield row["book"], c, feature / norm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    data, output = args.data, args.output
    output.mkdir(exist_ok=True, parents=True)
    rows = [json.loads(line) for line in (data / "lines.jsonl").read_text().splitlines()]
    train = [r for r in rows if r["split"] == "train"]
    val = [r for r in rows if r["split"] == "validation"]
    used = random.Random(1128).sample(val, 36)
    val = [r for r in val if r not in used]
    counts, means = Counter(), {}
    examples = list(glyphs(train, data))
    for book, c, feature in examples:
        key = (book, c)
        counts[key] += 1
        if key in means:
            means[key] += feature
        else:
            means[key] = feature.copy()
    labels = np.array([key[1] for key in means])
    prototypes = np.stack(list(means.values()))
    prototypes /= np.linalg.norm(prototypes, axis=1, keepdims=True).clip(min=1e-6)
    validation = list(glyphs(val, data))
    correct = np.array([c for book, c, feature in validation])
    predictions = labels[(np.stack([f for book, c, f in validation]) @ prototypes.T).argmax(axis=1)]
    frequency = Counter()
    for (_, c), n in counts.items():
        frequency[c] += n
    common = frequency.most_common(1)[0][0]
    result = dict(training_letters=sum(counts.values()), training_books=len({r["book"] for r in train}),
                  validation_letters=len(correct), validation_lines=len(val), validation_book=val[0]["book"],
                  top1=int((predictions == correct).sum()), most_frequent_letter=common,
                  baseline_top1=int((correct == common).sum()), observed_labels=len(set(correct)))
    (output / "letter-probe.json").write_text(json.dumps(result, indent=2) + "\n")
    print(result, flush=True)
    train_reader(examples, validation, output)


def make_reader():
    from torch import nn
    return nn.Sequential(nn.Conv2d(1, 24, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                         nn.Conv2d(24, 48, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                         nn.Conv2d(48, 48, 3, padding=1), nn.ReLU(), nn.Flatten(),
                         nn.Linear(48 * 6 * 4, 96), nn.ReLU(), nn.Linear(96, 52))


def train_reader(examples, validation, output):
    import torch
    from torch.nn import functional as F

    classes = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    torch.manual_seed(1127)
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    def pack(rows):
        x = np.stack([r[2] for r in rows]).reshape(-1, 1, 24, 16) * math.sqrt(384)
        y = [classes.index(r[1]) for r in rows]
        return torch.from_numpy(x).to(device), torch.tensor(y, device=device)
    x, y = pack(examples)
    test_x, test_y = pack(validation)
    reader = make_reader().to(device)
    optimizer = torch.optim.AdamW(reader.parameters(), lr=.001)
    for _ in range(700):
        idx = torch.randint(len(x), (128,), device=device)
        loss = F.cross_entropy(reader(x[idx]), y[idx])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.inference_mode():
        predictions = torch.cat([reader(batch).argmax(1) for batch in test_x.split(128)])
        match = (predictions == test_y).cpu()
    result = dict(training_characters=len(x), evaluated=len(test_y), correct=int(match.sum()),
                  case_sensitive=True, classes=classes, steps=700, seed=1127,
                  groups={c: {"n":int((test_y==k).sum()), "correct":int(match[(test_y==k).cpu()].sum())}
                          for k,c in enumerate(classes) if (test_y==k).any()})
    torch.save({k:v.cpu() for k,v in reader.state_dict().items()}, output / "letter-reader.pt")
    (output / "letter-reader.json").write_text(json.dumps(result, indent=2) + "\n")
    print(result["correct"], "/", result["evaluated"], flush=True)


if __name__ == "__main__":
    main()
