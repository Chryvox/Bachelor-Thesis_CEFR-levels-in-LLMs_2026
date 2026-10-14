import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

FILE_MAP = {
    "DE": "BOS-unsplit/qwen_baseline_de_l4.pt",
    "EN": "BOS-unsplit/qwen_baseline_en_l4.pt",
    "AR": "BOS-unsplit/qwen_baseline_ar_l4.pt",
    "HI": "BOS-unsplit/qwen_baseline_hi_l4.pt",
}

#FILE_MAP = {
#    "DE": "BOS-unsplit/aya_baseline_de_l16.pt",
#    "EN": "BOS-unsplit/aya_baseline_en_l16.pt",
#    "AR": "BOS-unsplit/aya_baseline_ar_l16.pt",
#    "HI": "BOS-unsplit/aya_baseline_hi_l16.pt",
#}

OUT_DIR = Path("pairwise_similarity_output_full")
OUT_DIR.mkdir(exist_ok=True)

# ── Load data ──
all_vecs, all_langs, all_levels = [], [], []
for lang, path in FILE_MAP.items():
    d = torch.load(path, map_location="cpu")
    all_vecs.append(d["states"].numpy())
    all_langs += [str(l).upper() for l in d["languages"]]
    all_levels += [str(l).upper() for l in d["cefr_levels"]]

X = np.vstack(all_vecs)
langs = np.array(all_langs)
levels = np.array(all_levels)
n = len(X)
print(f"Total: {n} texts -> {n*(n-1)//2:,} pairs")

# ── Full similarity matrix (vectorized, no sampling) ──
sim = cosine_similarity(X)  # [n, n], for n=3960 -> approx. 125 MB, easy for M1

# ── Build masks vectorized (no Python loop over pairs!) ──
same_lang_mask = langs[:, None] == langs[None, :]
same_level_mask = levels[:, None] == levels[None, :]

# upper triangular matrix only, excluding the diagonal
triu_mask = np.triu(np.ones((n, n), dtype=bool), k=1)

same_level_diff_lang_mask = same_level_mask & ~same_lang_mask & triu_mask
diff_level_same_lang_mask = ~same_level_mask & same_lang_mask & triu_mask

same_level_diff_lang = sim[same_level_diff_lang_mask]
diff_level_same_lang = sim[diff_level_same_lang_mask]

print(f"\nPairs 'same level, different language': {len(same_level_diff_lang):,}")
print(f"Mean: {same_level_diff_lang.mean():.4f}")

print(f"\nPairs 'different level, same language': {len(diff_level_same_lang):,}")
print(f"Mean: {diff_level_same_lang.mean():.4f}")

observed_diff = same_level_diff_lang.mean() - diff_level_same_lang.mean()
print(f"\nDifference: {observed_diff:+.4f}")

# ── Permutation test (on a sample of the pairs, since otherwise there are too many values) ──
rng = np.random.default_rng(42)
PERM_SAMPLE = 50000  # subset of pairs for the test, sufficient for a stable p-value

idx1 = rng.choice(len(same_level_diff_lang), size=min(PERM_SAMPLE, len(same_level_diff_lang)), replace=False)
idx2 = rng.choice(len(diff_level_same_lang), size=min(PERM_SAMPLE, len(diff_level_same_lang)), replace=False)

combined = np.concatenate([same_level_diff_lang[idx1], diff_level_same_lang[idx2]])
n1 = len(idx1)

n_perm = 1000
perm_diffs = np.zeros(n_perm)
for k in range(n_perm):
    rng.shuffle(combined)
    perm_diffs[k] = combined[:n1].mean() - combined[n1:].mean()

p_value = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))
print(f"p-value (permutation test on {PERM_SAMPLE:,}+{PERM_SAMPLE:,} pairs): {p_value:.4f}")

# ── Save ──
summary_df = pd.DataFrame({
    "Comparison Type": ["Same level, different language", "Different level, same language"],
    "Mean Cosine Similarity": [same_level_diff_lang.mean(), diff_level_same_lang.mean()],
    "Standard Deviation": [same_level_diff_lang.std(), diff_level_same_lang.std()],
    "Number of Pairs (ALL, no sampling)": [len(same_level_diff_lang), len(diff_level_same_lang)]
})
summary_df.to_csv(OUT_DIR / "pairwise_similarity_summary_FULL.csv", index=False)
print(f"\nSaved in: {OUT_DIR.resolve()}")
# ── Bar chart ──
fig, ax = plt.subplots(figsize=(7, 6))

means = [same_level_diff_lang.mean(), diff_level_same_lang.mean()]
sems = [
    same_level_diff_lang.std() / np.sqrt(len(same_level_diff_lang)),
    diff_level_same_lang.std() / np.sqrt(len(diff_level_same_lang))
]
labels = ["Same level,\ndifferent language", "Different level,\nsame language"]
colors = ["#4C72B0", "#DD8452"]

bars = ax.bar(labels, means, yerr=sems, capsize=8, color=colors, width=0.5, edgecolor="black")

for bar, mean in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width()/2, mean + 0.01, f"{mean:.3f}",
            ha="center", fontsize=12, fontweight="bold")

ax.set_ylabel("Ø Cosine Similarity", fontsize=12)
ax.set_title("Representation similarity (all texts):\nLanguage vs. CEFR level", fontsize=13)
ax.set_ylim(0, max(means) * 1.25)

sig_label = "p < 0.001" if p_value < 0.001 else f"p = {p_value:.3f}"
y_sig = max(means) * 1.08
ax.plot([0, 0, 1, 1], [y_sig, y_sig*1.02, y_sig*1.02, y_sig], color="black")
ax.text(0.5, y_sig*1.03, sig_label, ha="center", fontsize=11)

plt.tight_layout()
plt.savefig(OUT_DIR / "pairwise_similarity_barplot_FULL.png", dpi=300)
plt.close()

# ── Distribution histogram ──
fig, ax = plt.subplots(figsize=(8, 6))

# for millions of values: subsample ONLY for plotting (otherwise the histogram takes forever)
plot_sample = 100_000
same_plot = np.random.choice(same_level_diff_lang, size=min(plot_sample, len(same_level_diff_lang)), replace=False)
diff_plot = np.random.choice(diff_level_same_lang, size=min(plot_sample, len(diff_level_same_lang)), replace=False)

ax.hist(diff_plot, bins=60, alpha=0.6, label="Different level, same language",
        color="#DD8452", density=True)
ax.hist(same_plot, bins=60, alpha=0.6, label="Same level, different language",
        color="#4C72B0", density=True)
ax.axvline(diff_level_same_lang.mean(), color="#DD8452", linestyle="--")
ax.axvline(same_level_diff_lang.mean(), color="#4C72B0", linestyle="--")
ax.set_xlabel("Cosine Similarity")
ax.set_ylabel("Density")
ax.set_title("Distribution of pairwise similarities (all texts)")
ax.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "pairwise_similarity_distribution_FULL.png", dpi=300)
plt.close()

print(f"\nPlots saved in: {OUT_DIR.resolve()}")