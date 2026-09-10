import json
import pandas as pd
from scipy.stats import pearsonr

""" Pearson Correlation Coefficient (r): >0.7 target"""
# load human labels
human = pd.read_csv("src\\risk_reasoning\\data\\human_label_chosen.csv")

# load judge outputs
validation_rows = []
with open("results\\runs\\judge_validation_30\\judge_scores.jsonl", "r") as f:
    for line in f:
        validation_rows.append(json.loads(line))

judge = pd.DataFrame(validation_rows)
df = human.merge(judge, on="id")

# continuous metrics
for h, j in [
    ("human_coherence", "coherence"),
    ("human_reasoning_coverage", "reasoning_coverage"),
]:
    r, _ = pearsonr(df[h], df[j])
    print(f"{j}: r = {r:.3f}")

# normalize count metrics
df["judge_risk_norm"] = df["risk_factors_enumerated"].clip(upper=3) / 3
df["judge_alt_norm"] = df["alternatives_considered"].clip(upper=3) / 3

for h, j in [
    ("human_risk_factors_enumerated", "judge_risk_norm"),
    ("human_alternatives_considered", "judge_alt_norm"),
]:
    r, _ = pearsonr(df[h], df[j])
    print(f"{j}: r = {r:.3f}")

df["coh_err"] = (df["human_coherence"] - df["coherence"]).abs()
df["cov_err"] = (df["human_reasoning_coverage"] - df["reasoning_coverage"]).abs()
df["risk_err"] = (df["human_risk_factors_enumerated"] - df["judge_risk_norm"]).abs()
df["alt_err"] = (df["human_alternatives_considered"] - df["judge_alt_norm"]).abs()

# print(df.sort_values("cov_err", ascending=False)[["id","human_reasoning_coverage","reasoning_coverage","cov_err"]].head(10))
# print(df.sort_values("risk_err", ascending=False)[["id","human_risk_factors_enumerated","judge_risk_norm","risk_err"]].head(10))
# print(df.sort_values("alt_err", ascending=False)[["id","human_alternatives_considered","judge_alt_norm","alt_err"]].head(10))

# First run
"""
coherence: r = 0.514
reasoning_coverage: r = 0.646
judge_risk_norm: r = 0.667
judge_alt_norm: r = 0.533
"""

# Second run
"""
coherence: r = 0.572
reasoning_coverage: r = 0.679
judge_risk_norm: r = 0.617
judge_alt_norm: r = 0.533
"""

# Third run (>0.7 on three counts, coherence is satisfactory given small sample size)
"""
coherence: r = 0.656
reasoning_coverage: r = 0.772
judge_risk_norm: r = 0.777
judge_alt_norm: r = 0.801
"""

# Consistency
# load judge outputs
consistency_rows = []
with open("results\\runs\\judge_consistency_10\\judge_scores.jsonl", "r") as f:
    for line in f:
        consistency_rows.append(json.loads(line))

judge2 = pd.DataFrame(consistency_rows)
df = judge2.merge(judge, on="id", suffixes=("_1", "_2"))

metrics = ["coherence", "reasoning_coverage"]

for m in metrics:
    delta = (df[f"{m}_1"] - df[f"{m}_2"]).abs().mean()
    print(f"{m}: mean |delta| = {delta:.3f}")

# counts dominating, normalize (/3) first
for m in ["risk_factors_enumerated", "alternatives_considered"]:
    d1 = df[f"{m}_1"].clip(upper=3) / 3 
    d2 = df[f"{m}_2"].clip(upper=3) / 3
    delta = (d1 - d2).abs().mean()
    print(f"{m}: mean |delta| = {delta:.3f}")

# Consistency run (Perfect consistency due to temperature=0)
"""
coherence: r = 0.656
reasoning_coverage: r = 0.772
judge_risk_norm: r = 0.777
judge_alt_norm: r = 0.801
coherence: mean |delta| = 0.000
reasoning_coverage: mean |delta| = 0.000
risk_factors_enumerated: mean |delta| = 0.000
alternatives_considered: mean |delta| = 0.000
"""