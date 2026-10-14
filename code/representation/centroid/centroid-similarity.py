"""
Centroid similarity analysis
Model: Aya Expanse 8B / Qwen3.5 9B, layer 16 / 4, pooling: mean, subset: full
Languages: DE, EN, AR, HI × CEFR A1-C2
"""

import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

# -----------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------

DATA_DIR = Path(".")  # adjust to actual path if needed

LANGUAGES = ["DE", "EN", "AR", "HI"]
LEVELS = ["A1", "A2", "B1", "B2", "C1", "C2"]

FILE_MAP = {
    "DE": "BOS-unsplit/aya_baseline_de_l16.pt",
    "EN": "BOS-unsplit/aya_baseline_en_l16.pt",
    "AR": "BOS-unsplit/aya_baseline_ar_l16.pt",
    "HI": "BOS-unsplit/aya_baseline_hi_l16.pt",
}

#FILE_MAP = {
#    "DE": "BOS-unsplit/qwen_baseline_de_l4.pt",
#    "EN": "BOS-unsplit/qwen_baseline_en_l4.pt",
#    "AR": "BOS-unsplit/qwen_baseline_ar_l4.pt",
#    "HI": "BOS-unsplit/qwen_baseline_hi_l4.pt",
#}

OUT_DIR = Path("centroid_analysis_output")
OUT_DIR.mkdir(exist_ok=True)


# -----------------------------------------------------------------
# LOADING
# -----------------------------------------------------------------

def load_vectors_and_levels(path, expected_language=None):
    data = torch.load(path, map_location="cpu")

    states = data["states"]
    if isinstance(states, torch.Tensor):
        states = states.numpy()
    states = np.asarray(states)

    cefr_levels = np.array([str(l).upper() for l in data["cefr_levels"]])
    languages = np.array([str(l).upper() for l in data["languages"]])

    assert states.shape[0] == len(cefr_levels) == len(languages), \
        f"Inconsistent lengths in {path.name}: " \
        f"states={states.shape[0]}, cefr={len(cefr_levels)}, lang={len(languages)}"

    unique_langs = set(languages)
    print(f"[{path.name}] Languages: {unique_langs} | "
          f"Subset: {data.get('subset')} | Layer: {data.get('layer')} | "
          f"Pooling: {data.get('pooling')} | N={states.shape[0]}")

    if expected_language is not None and unique_langs != {expected_language}:
        print(f"  WARNING: expected only '{expected_language}', found: {unique_langs}")

    return states, cefr_levels, languages


# -----------------------------------------------------------------
# STEP 1 & 2 — Grouping + centroid computation
# -----------------------------------------------------------------

centroids = {}
group_sizes = {}

for lang in LANGUAGES:
    file_path = DATA_DIR / FILE_MAP[lang]
    states, cefr_levels, languages = load_vectors_and_levels(file_path, expected_language=lang)

    for level in LEVELS:
        mask = (cefr_levels == level) & (languages == lang)
        n = mask.sum()
        if n == 0:
            print(f"  WARNING: No vectors found for {lang}_{level}!")
            continue
        centroid = states[mask].mean(axis=0)
        key = f"{lang}_{level}"
        centroids[key] = centroid
        group_sizes[key] = int(n)

print("\nGroup sizes:")
for k, v in group_sizes.items():
    print(f"  {k}: {v} texts")

sizes_df = pd.DataFrame(
    [{"group": k, "n_texts": v} for k, v in group_sizes.items()]
)
sizes_df.to_csv(OUT_DIR / "group_sizes.csv", index=False)

# -----------------------------------------------------------------
# STEP 3 — Pairwise cosine similarity
# -----------------------------------------------------------------

ordered_keys = [f"{lang}_{level}" for lang in LANGUAGES for level in LEVELS
                if f"{lang}_{level}" in centroids]

matrix = np.vstack([centroids[k] for k in ordered_keys])
sim_matrix = cosine_similarity(matrix)
sim_df = pd.DataFrame(sim_matrix, index=ordered_keys, columns=ordered_keys)

# -----------------------------------------------------------------
# OUTPUT 1 — CSV / Excel
# -----------------------------------------------------------------

sim_df.to_csv(OUT_DIR / "centroid_similarity_matrix.csv")
sim_df.to_excel(OUT_DIR / "centroid_similarity_matrix.xlsx")

# -----------------------------------------------------------------
# OUTPUT 2 — Heatmap
# -----------------------------------------------------------------

plt.figure(figsize=(14, 12))
sns.heatmap(
    sim_df,
    annot=False,
    cmap="viridis",
    vmin=sim_df.values.min(),
    vmax=1.0,
    square=True,
    cbar_kws={"label": "Cosine Similarity"}
)
plt.title("Centroid Cosine Similarity — Language × CEFR-Level\n"
          "Aya Expanse 8B, Layer 16, mean-pooling, full dataset") #"Qwen3.5 9B, Layer 4, mean-pooling, full dataset")
plt.tight_layout()
plt.savefig(OUT_DIR / "centroid_similarity_heatmap.png", dpi=300)
plt.close()

# -----------------------------------------------------------------
# OUTPUT 3 — Within-language vs. cross-language metrics
# -----------------------------------------------------------------

within_language_diff_level = []
cross_language_same_level = []

for i, key_a in enumerate(ordered_keys):
    lang_a, level_a = key_a.split("_")
    for j, key_b in enumerate(ordered_keys):
        if i >= j:
            continue
        lang_b, level_b = key_b.split("_")
        sim_value = sim_df.loc[key_a, key_b]

        if lang_a == lang_b and level_a != level_b:
            within_language_diff_level.append(sim_value)

        if lang_a != lang_b and level_a == level_b:
            cross_language_same_level.append(sim_value)

mean_within_diff_level = float(np.mean(within_language_diff_level))
mean_cross_same_level = float(np.mean(cross_language_same_level))

print("\n--- Metrics ---")
print(f"Mean within-language similarity (different levels): {mean_within_diff_level:.4f}")
print(f"Mean cross-language similarity (same level):        {mean_cross_same_level:.4f}")
print(f"Difference (cross-same minus within-diff): {mean_cross_same_level - mean_within_diff_level:+.4f}")

summary_df = pd.DataFrame({
    "Kennzahl": [
        "Mean within-language similarity (different levels)",
        "Mean cross-language similarity (same level)"
    ],
    "Wert": [mean_within_diff_level, mean_cross_same_level],
    "N_Paare": [len(within_language_diff_level), len(cross_language_same_level)]
})
summary_df.to_csv(OUT_DIR / "summary_metrics.csv", index=False)

print(f"\nAll results saved in: {OUT_DIR.resolve()}")