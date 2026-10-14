import pandas as pd
import numpy as np

# ── Load CSV with the 24x24 Aya centroid matrix ──
sim_df = pd.read_csv("aya_centroid_analysis_output/centroid_similarity_matrix.csv", index_col=0)  # adjust file name if needed

keys = sim_df.index.tolist()

same_level_cross_lang = []
diff_level_cross_lang = []

for i, key_a in enumerate(keys):
    lang_a, level_a = key_a.split("_")
    for j, key_b in enumerate(keys):
        if i >= j:
            continue

        lang_b, level_b = key_b.split("_")
        value = sim_df.loc[key_a, key_b]

        if lang_a == lang_b:
            continue  # ignore within-language

        if level_a == level_b:
            same_level_cross_lang.append(value)
        else:
            diff_level_cross_lang.append(value)

same_level_cross_lang = np.array(same_level_cross_lang)
diff_level_cross_lang = np.array(diff_level_cross_lang)

print(f"Same level, cross-lingual:      n={len(same_level_cross_lang)}, Ø={same_level_cross_lang.mean():.4f}")
print(f"Different level, cross-lingual: n={len(diff_level_cross_lang)}, Ø={diff_level_cross_lang.mean():.4f}")

observed_diff = same_level_cross_lang.mean() - diff_level_cross_lang.mean()
print(f"Observed difference: {observed_diff:+.4f}")

# ── Permutation test ──
combined = np.concatenate([same_level_cross_lang, diff_level_cross_lang])
n1 = len(same_level_cross_lang)

rng = np.random.default_rng(42)
n_perm = 10000

perm_diffs = np.zeros(n_perm)
for k in range(n_perm):
    shuffled = rng.permutation(combined)
    perm_diffs[k] = shuffled[:n1].mean() - shuffled[n1:].mean()

p_value = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))
print(f"p-value (permutation test, {n_perm} permutations): {p_value:.4f}")