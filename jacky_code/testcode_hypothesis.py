from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

from scipy.stats import chi2

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# =========================
# CONFIG
# =========================
DATA_SHEET = "data"
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset_algorithmic_persuasion_10000.xlsx"
OUT_DIR = BASE_DIR / "model_outputs"

ALPHA = 0.05


def decision(p: float, alpha: float = ALPHA) -> str:
    return "SUPPORTED ✅" if p < alpha else "NOT SUPPORTED ❌"


def savefig(filename: str) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUT_DIR / filename, dpi=200)
    plt.show()


def anova_term_p(model, term: str) -> float:
    """Return p-value for an ANOVA term from a fitted OLS model."""
    a = anova_lm(model, typ=2)
    if term not in a.index:
        raise KeyError(f"ANOVA term '{term}' not found. Available: {list(a.index)}")
    return float(a.loc[term, "PR(>F)"])


def main() -> None:
    print("PYTHON EXECUTABLE:", sys.executable)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Could not find: {DATA_PATH}\n"
            "Put dataset_algorithmic_persuasion_10000.xlsx in the same folder as this script."
        )

    # -------------------------
    # Load
    # -------------------------
    df = pd.read_excel(DATA_PATH, sheet_name=DATA_SHEET)
    df.columns = [c.strip() for c in df.columns]

    required = {
        "content_type", "post_format",
        "persuasive_power_index",
        "authority_log", "verified", "account_age_years",
        "early_window_mins",
        "early_likes", "early_comments", "early_shares",
        "early_total_engagement",
        "early_engagement_velocity",
    }
    missing = sorted([c for c in required if c not in df.columns])
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # -------------------------
    # Build the 9-level post design configuration (combo)
    # -------------------------
    df["combo"] = df["content_type"].astype(str) + " | " + df["post_format"].astype(str)

    # -------------------------
    # Early-window check (should be 60 mins)
    # -------------------------
    df["early_window_mins"] = pd.to_numeric(df["early_window_mins"], errors="coerce")
    print("\n--- EARLY WINDOW CHECK ---")
    print(df["early_window_mins"].describe())

    # -------------------------
    # Derived outcome variables
    # -------------------------
    mins = df["early_window_mins"].replace(0, np.nan)

    # Rates (per minute)
    df["likes_rate"] = df["early_likes"] / mins
    df["comments_rate"] = df["early_comments"] / mins
    df["shares_rate"] = df["early_shares"] / mins

    # Composition (share of early total engagement)
    denom = df["early_total_engagement"].replace(0, np.nan)
    df["likes_share"] = df["early_likes"] / denom
    df["comments_share"] = df["early_comments"] / denom
    df["shares_share"] = df["early_shares"] / denom

    # Log transforms (robustness for skew)
    df["log1p_early_likes"] = np.log1p(df["early_likes"])
    df["log1p_early_comments"] = np.log1p(df["early_comments"])
    df["log1p_early_shares"] = np.log1p(df["early_shares"])

    # High engagement definition (your choice A): PPI > 0
    df["high_engagement"] = (df["persuasive_power_index"] > 0).astype(int)

    # Order combos consistently (by mean PPI)
    combo_order = (
        df.groupby("combo")["persuasive_power_index"]
          .mean()
          .sort_values(ascending=False)
          .index
          .tolist()
    )

    OUT_DIR.mkdir(exist_ok=True)
    results = []

    # ==========================================================
    # H1
    # ==========================================================
    # H1: Influencer authority significantly predicts early engagement.
    # Test: OLS regression
    # DV: persuasive_power_index
    # IV: authority_log
    # Controls: verified, account_age_years
    print("\n" + "=" * 95)
    print("H1: Influencer authority significantly predicts early engagement (PPI).")
    print("Model: PPI ~ authority_log + verified + account_age_years")
    print("=" * 95)

    m_h1 = smf.ols("persuasive_power_index ~ authority_log + verified + account_age_years", data=df).fit()
    p_h1 = float(m_h1.pvalues["authority_log"])
    print(m_h1.summary())
    print(f"\nH1 decision: p(authority_log)={p_h1:.4g} → {decision(p_h1)}")
    results.append(("H1", "authority_log → PPI (controls)", p_h1, decision(p_h1)))

    plt.figure(figsize=(7, 4))
    plt.scatter(df["authority_log"], df["persuasive_power_index"], s=10)
    plt.title("PPI vs Influencer Authority (log)")
    plt.xlabel("authority_log")
    plt.ylabel("persuasive_power_index")
    savefig("H1_ppi_vs_authority_scatter.png")

    # ==========================================================
    # H2
    # ==========================================================
    # H2: Content design configuration significantly predicts the overall engagement momentum (velocity).
    # Test: ANOVA (9-level combo factor)
    # DV: early_engagement_velocity
    # IV: combo (9 configs)
    print("\n" + "=" * 95)
    print("H2: Post design configuration (9 combos) predicts engagement momentum (velocity).")
    print("Model: early_engagement_velocity ~ C(combo)")
    print("=" * 95)

    m_h2 = smf.ols("early_engagement_velocity ~ C(combo)", data=df).fit()
    p_h2 = anova_term_p(m_h2, "C(combo)")
    print(anova_lm(m_h2, typ=2))
    print(f"\nH2 decision: p(C(combo))={p_h2:.4g} → {decision(p_h2)}")
    results.append(("H2", "combo → early_engagement_velocity (ANOVA)", p_h2, decision(p_h2)))

    # Boxplot velocity by combo
    vel_data = [df.loc[df["combo"] == c, "early_engagement_velocity"].dropna() for c in combo_order]
    plt.figure(figsize=(11, 4))
    plt.boxplot(vel_data, labels=combo_order, showfliers=False)
    plt.xticks(rotation=45, ha="right")
    plt.title("Early Engagement Velocity by Post Combo (outliers hidden)")
    plt.ylabel("early_engagement_velocity")
    savefig("H2_velocity_box_by_combo.png")

    # Helper to test combo effect for each outcome + plot
    def combo_test(h_label: str, dv: str, pretty: str, prefix: str) -> None:
        """
        Runs ANOVA test: dv ~ C(combo)
        Saves mean bar chart + boxplot.
        """
        model = smf.ols(f"{dv} ~ C(combo)", data=df).fit()
        p = anova_term_p(model, "C(combo)")

        print("\n" + "-" * 95)
        print(f"{h_label}: {pretty}")
        print(f"Model: {dv} ~ C(combo)")
        print("-" * 95)
        print(anova_lm(model, typ=2))
        print(f"{h_label} decision: p(C(combo))={p:.4g} → {decision(p)}")
        results.append((h_label, f"combo → {dv} (ANOVA)", p, decision(p)))

        # Mean bar chart
        means = df.groupby("combo")[dv].mean().reindex(combo_order)
        plt.figure(figsize=(11, 4))
        plt.bar(means.index, means.values)
        plt.xticks(rotation=45, ha="right")
        plt.title(f"{pretty} — Mean by Combo")
        plt.ylabel(f"mean {dv}")
        savefig(f"{prefix}_mean_{dv}_by_combo.png")

        # Distribution boxplot (outliers hidden)
        data = [df.loc[df["combo"] == c, dv].dropna() for c in combo_order]
        plt.figure(figsize=(11, 4))
        plt.boxplot(data, labels=combo_order, showfliers=False)
        plt.xticks(rotation=45, ha="right")
        plt.title(f"{pretty} — Distribution by Combo (outliers hidden)")
        plt.ylabel(dv)
        savefig(f"{prefix}_box_{dv}_by_combo.png")

    # ==========================================================
    # H3
    # ==========================================================
    # H3: Content design configuration significantly predicts the rate and composition of early_likes.
    # Tests (all using 9 combos):
    #   - likes_rate ~ C(combo)
    #   - likes_share ~ C(combo)
    #   - log1p_early_likes ~ C(combo)  (robustness)
    print("\n" + "=" * 95)
    print("H3: Post design configuration (9 combos) predicts likes (rate + composition).")
    print("=" * 95)
    combo_test("H3_rate", "likes_rate", "Likes Rate (per minute)", "H3")
    combo_test("H3_comp", "likes_share", "Likes Share of Early Engagement", "H3")
    combo_test("H3_log", "log1p_early_likes", "Log(1 + Early Likes)", "H3")

    # ==========================================================
    # H4
    # ==========================================================
    # H4: Content design configuration significantly predicts the rate and composition of early_shares.
    print("\n" + "=" * 95)
    print("H4: Post design configuration (9 combos) predicts shares (rate + composition).")
    print("=" * 95)
    combo_test("H4_rate", "shares_rate", "Shares Rate (per minute)", "H4")
    combo_test("H4_comp", "shares_share", "Shares Share of Early Engagement", "H4")
    combo_test("H4_log", "log1p_early_shares", "Log(1 + Early Shares)", "H4")

    # ==========================================================
    # H5
    # ==========================================================
    # H5: Content design configuration significantly predicts the rate and composition of early_comments.
    print("\n" + "=" * 95)
    print("H5: Post design configuration (9 combos) predicts comments (rate + composition).")
    print("=" * 95)
    combo_test("H5_rate", "comments_rate", "Comments Rate (per minute)", "H5")
    combo_test("H5_comp", "comments_share", "Comments Share of Early Engagement", "H5")
    combo_test("H5_log", "log1p_early_comments", "Log(1 + Early Comments)", "H5")

    # ==========================================================
    # H6
    # ==========================================================
    # H6: authority_log moderates the relation between content design configuration and
    #     significant engagement levels in the early window.
    # DV: high_engagement (PPI > 0)
    # Test: Logistic regression + LR test:
    #   Base: high_engagement ~ C(combo) + authority_log + verified + account_age_years
    #   Int:  high_engagement ~ C(combo) * authority_log + verified + account_age_years
    print("\n" + "=" * 95)
    print("H6: Authority (log) moderates combo → high engagement (PPI > 0).")
    print("Test: Likelihood Ratio (LR) test comparing interaction vs non-interaction logit models")
    print("=" * 95)

    base_f = "high_engagement ~ C(combo) + authority_log + verified + account_age_years"
    int_f = "high_engagement ~ C(combo) * authority_log + verified + account_age_years"

    logit_base = smf.logit(base_f, data=df).fit(disp=0)
    logit_int = smf.logit(int_f, data=df).fit(disp=0)

    lr = 2 * (logit_int.llf - logit_base.llf)
    df_diff = int(logit_int.df_model - logit_base.df_model)
    p_h6 = float(chi2.sf(lr, df_diff))

    print("Base:", base_f)
    print("Interaction:", int_f)
    print(f"LR={lr:.3f}, df={df_diff}, p={p_h6:.4g} → {decision(p_h6)}")
    results.append(("H6", "combo × authority_log → high_engagement (LR test)", p_h6, decision(p_h6)))

    # Plot predicted probability vs authority for top3 + bottom3 combos
    top3 = df.groupby("combo")["persuasive_power_index"].mean().sort_values(ascending=False).head(3).index.tolist()
    bot3 = df.groupby("combo")["persuasive_power_index"].mean().sort_values(ascending=False).tail(3).index.tolist()
    plot_combos = top3 + bot3

    agrid = np.linspace(df["authority_log"].quantile(0.05), df["authority_log"].quantile(0.95), 25)

    plt.figure(figsize=(8, 4))
    for c in plot_combos:
        temp = pd.DataFrame({
            "combo": [c] * len(agrid),
            "authority_log": agrid,
            "verified": df["verified"].median(),
            "account_age_years": df["account_age_years"].median(),
        })
        pred = logit_int.predict(temp)
        plt.plot(agrid, pred, label=c)

    plt.title("H6: Predicted P(high_engagement) vs Authority (Top3 + Bottom3 combos)")
    plt.xlabel("authority_log")
    plt.ylabel("Predicted probability (PPI > 0)")
    plt.legend(fontsize=7)
    savefig("H6_predprob_vs_authority_top3_bottom3.png")

    # ==========================================================
    # H7
    # ==========================================================
    # H7: verified moderates the relation between content design configuration and
    #     significant engagement in the early window.
    # DV: high_engagement (PPI > 0)
    # Test: Logistic regression + LR test:
    #   Base: high_engagement ~ C(combo) + verified + authority_log + account_age_years
    #   Int:  high_engagement ~ C(combo) * verified + authority_log + account_age_years
    print("\n" + "=" * 95)
    print("H7: Verified moderates combo → high engagement (PPI > 0).")
    print("Test: Likelihood Ratio (LR) test comparing interaction vs non-interaction logit models")
    print("=" * 95)

    base_v = "high_engagement ~ C(combo) + verified + authority_log + account_age_years"
    int_v = "high_engagement ~ C(combo) * verified + authority_log + account_age_years"

    logit_base_v = smf.logit(base_v, data=df).fit(disp=0)
    logit_int_v = smf.logit(int_v, data=df).fit(disp=0)

    lr_v = 2 * (logit_int_v.llf - logit_base_v.llf)
    df_diff_v = int(logit_int_v.df_model - logit_base_v.df_model)
    p_h7 = float(chi2.sf(lr_v, df_diff_v))

    print("Base:", base_v)
    print("Interaction:", int_v)
    print(f"LR={lr_v:.3f}, df={df_diff_v}, p={p_h7:.4g} → {decision(p_h7)}")
    results.append(("H7", "combo × verified → high_engagement (LR test)", p_h7, decision(p_h7)))

    # Predicted probabilities by combo split by verified for top 5 combos
    top5 = df.groupby("combo")["persuasive_power_index"].mean().sort_values(ascending=False).head(5).index.tolist()

    temp0 = pd.DataFrame({
        "combo": top5,
        "verified": 0,
        "authority_log": df["authority_log"].median(),
        "account_age_years": df["account_age_years"].median(),
    })
    temp1 = temp0.copy()
    temp1["verified"] = 1

    p0 = logit_int_v.predict(temp0)
    p1 = logit_int_v.predict(temp1)

    x = np.arange(len(top5))
    plt.figure(figsize=(10, 4))
    plt.bar(x - 0.2, p0, width=0.4, label="verified=0")
    plt.bar(x + 0.2, p1, width=0.4, label="verified=1")
    plt.xticks(x, top5, rotation=45, ha="right")
    plt.title("H7: Predicted P(high_engagement) by Combo (Top 5) — Verified vs Not")
    plt.ylabel("Predicted probability (PPI > 0)")
    plt.legend()
    savefig("H7_predprob_by_combo_verified_split_top5.png")

    # ==========================================================
    # Naive Bayes (classification)
    # ==========================================================
    # Predict high_engagement from:
    #   combo (9 configs) + authority_log + verified + account_age_years
    print("\n" + "=" * 95)
    print("Naive Bayes: Predict high_engagement (PPI > 0) from combo + influencer attributes")
    print("=" * 95)

    X = df[["combo", "authority_log", "verified", "account_age_years"]].copy()
    y = df["high_engagement"].copy()

    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_combo = ohe.fit_transform(X[["combo"]])
    X_num = X[["authority_log", "verified", "account_age_years"]].to_numpy(dtype=float)
    X_all = np.concatenate([X_combo, X_num], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y, test_size=0.25, random_state=42, stratify=y
    )

    nb = GaussianNB()
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Accuracy: {acc:.4f}")
    print("Confusion matrix:\n", cm)
    print("\nClassification report:\n", classification_report(y_test, y_pred))

    plt.figure(figsize=(4, 4))
    plt.imshow(cm, aspect="auto")
    plt.title("Naive Bayes Confusion Matrix")
    plt.xticks([0, 1], ["Pred 0", "Pred 1"])
    plt.yticks([0, 1], ["True 0", "True 1"])
    for (i, j), val in np.ndenumerate(cm):
        plt.text(j, i, str(val), ha="center", va="center")
    savefig("NB_confusion_matrix.png")

    # -------------------------
    # Save summary
    # -------------------------
    summary = pd.DataFrame(results, columns=["Hypothesis", "Test", "p_value", "Decision"])
    summary.to_csv(OUT_DIR / "hypothesis_support_summary.csv", index=False)

    print("\nSaved:", OUT_DIR / "hypothesis_support_summary.csv")
    print("All plots saved to:", OUT_DIR)


if __name__ == "__main__":
    main()
