# smoke_test.py
"""
Smoke test for the fallacy model.
Run: python smoke_test.py
Pass condition: loads model + runs predict_proba without crashing.
"""

import sys
import joblib

MODEL_PATH = "FINAL_TRUE_MULTI_LABEL_model.joblib"
THRESHOLD = 0.35

TESTS = [
    ("multi-ish", "Everyone is doing it, so it must be true."),
    ("multi-ish", "If we allow this, society will collapse."),
    ("single-ish", "You're wrong because you're ignorant."),
    ("none", "The sky is blue and grass is green."),
    ("edge-empty", ""),
]


def main() -> int:
    pkg = joblib.load(MODEL_PATH)
    model = pkg["model"]
    lb = pkg["label_binarizer"]
    labels = list(lb.classes_)

    for name, text in TESTS:
        text = (text or "").strip()
        if not text:
            print(f"\n[{name}] (empty) -> SKIP")
            continue

        probs = model.predict_proba([text])[0]
        hits = [(lab, float(p)) for lab, p in zip(labels, probs) if p >= THRESHOLD]
        hits.sort(key=lambda x: x[1], reverse=True)

        top3 = sorted(zip(labels, probs), key=lambda x: x[1], reverse=True)[:3]
        print(f"\n[{name}] {text}")
        print("  top3:", [(lab, float(p)) for lab, p in top3])
        print("  hits:", hits if hits else "NO FALLACY DETECTED")

    print("\n✅ smoke_test complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())