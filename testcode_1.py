from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# CONFIG
# ----------------------------
DATA_SHEET = "data"
CODEBOOK_SHEET = "codebook"

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset_algorithmic_persuasion_10000.xlsx"
OUTPUT_DIR = BASE_DIR / "eda_outputs"


def main() -> None:
    print("PYTHON EXECUTABLE:", sys.executable)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Could not find: {DATA_PATH}\n"
            "Fix: make sure dataset_algorithmic_persuasion_10000.xlsx is in the same folder as this script."
        )

    # ---------- Load ----------
    xls = pd.ExcelFile(DATA_PATH)
    print("\nSheets:", xls.sheet_names)

    df = pd.read_excel(DATA_PATH, sheet_name=DATA_SHEET)
    codebook = pd.read_excel(DATA_PATH, sheet_name=CODEBOOK_SHEET)

    # Clean column names
    df.columns = [c.strip() for c in df.columns]
    codebook.columns = [c.strip() for c in codebook.columns]

    # ---------- Basic shape (Head) ----------
    print("\n--- BASIC SHAPE ---")
    print("data shape:", df.shape)
    print("codebook shape:", codebook.shape)

    print("\n--- HEAD (data) ---")
    print(df.head(5))

    # ---------- Dtypes / Missing values check ----------
    print("\n--- DTYPES (top 25) ---")
    print(df.dtypes.head(25))

    miss = (
        df.isna()
        .mean()
        .mul(100)
        .sort_values(ascending=False)
        .to_frame("missing_%")
    )
    print("\n--- MISSINGNESS (top 20) ---")
    print(miss.head(20).round(2))

    # ---------- Duplicates check ----------
    print("\n--- DUPLICATES ---")
    print("duplicate rows:", int(df.duplicated().sum()))
    if "post_id" in df.columns:
        print("duplicate post_id:", int(df["post_id"].duplicated().sum()))

    # ---------- Datetime check ----------
    if "post_datetime" in df.columns:
        df["post_datetime"] = pd.to_datetime(df["post_datetime"], errors="coerce")
        print("\n--- DATE RANGE ---")
        print("min:", df["post_datetime"].min(), "max:", df["post_datetime"].max())
        df["post_hour"] = df["post_datetime"].dt.hour
        df["post_dow"] = df["post_datetime"].dt.day_name()

    # ---------- Category counts ----------
    print("\n--- CATEGORY COUNTS ---")
    for c in ["content_type", "post_format", "verified"]:
        if c in df.columns:
            print(f"\n{c}:")
            print(df[c].value_counts(dropna=False))

    # ---------- Early window validation (60 mins check) ----------
    if "early_window_mins" in df.columns:
        print("\n--- EARLY WINDOW CHECK ---")
        print(df["early_window_mins"].describe())
        print("\nTop early_window_mins values:")
        print(df["early_window_mins"].value_counts().head(10))

    # ---------- Numeric summary ----------
    key_nums = [
        "follower_count",
        "authority_log",
        "account_age_years",
        "early_window_mins",
        "early_likes", "early_comments", "early_shares",
        "early_total_engagement",
        "early_engagement_velocity",
        "early_comment_share_ratio",
        "persuasive_power_index",
        "reach", "impressions",
        "impressions_per_reach",
        "algorithmic_amplification_index",
        "link_clicks", "saves", "profile_visits",
        "engagement_success_index",
    ]
    key_nums = [c for c in key_nums if c in df.columns]
    print("\n--- NUMERIC SUMMARY (selected) ---")
    print(df[key_nums].describe(percentiles=[0.01, 0.05, 0.50, 0.95, 0.99]).T.round(3))

    # ---------- Outlier snapshot ----------
    print("\n--- OUTLIER SNAPSHOT (p99, p99.5, max) ---")
    outlier_cols = [c for c in ["reach", "impressions", "early_likes", "persuasive_power_index"] if c in df.columns]
    for c in outlier_cols:
        p99 = df[c].quantile(0.99)
        p995 = df[c].quantile(0.995)
        mx = df[c].max()
        print(f"{c:25s} p99={p99:,.3f}  p99.5={p995:,.3f}  max={mx:,.3f}")

    # ---------- Sanity checks (derived fields & logical relationships) ----------
    print("\n--- SANITY CHECKS ---")

    # impressions_per_reach should ~ impressions / reach
    if {"impressions", "reach", "impressions_per_reach"}.issubset(df.columns):
        approx = df["impressions"] / df["reach"].replace(0, np.nan)
        diff = (df["impressions_per_reach"] - approx).abs()
        print("impressions_per_reach | median abs diff:", float(diff.median(skipna=True)))

        # logical: impressions should generally be >= reach
        bad = int((df["impressions"] < df["reach"]).sum())
        print("Rows where impressions < reach:", bad)

    # click_through_rate should ~ link_clicks / impressions
    if {"link_clicks", "impressions", "click_through_rate"}.issubset(df.columns):
        ctr_impr = df["link_clicks"] / df["impressions"].replace(0, np.nan)
        diff = (df["click_through_rate"] - ctr_impr).abs()
        print("click_through_rate vs clicks/impressions | median abs diff:", float(diff.median(skipna=True)))

    # early_total_engagement should ~ early_likes + early_comments + early_shares
    if {"early_total_engagement", "early_likes", "early_comments", "early_shares"}.issubset(df.columns):
        approx = df["early_likes"] + df["early_comments"] + df["early_shares"]
        med_abs_diff = float((df["early_total_engagement"] - approx).abs().median(skipna=True))
        print("early_total_engagement vs (likes+comments+shares) | median abs diff:", med_abs_diff)

    # ---------- Correlation sanity (does PPI correlate with early signals?) ----------
    corr_cols = [c for c in [
        "persuasive_power_index",
        "early_likes", "early_comments", "early_shares",
        "early_total_engagement",
        "early_engagement_velocity",
        "authority_log"
    ] if c in df.columns]
    if len(corr_cols) >= 2:
        print("\n--- QUICK CORRELATIONS (sanity) ---")
        corr = df[corr_cols].corr(numeric_only=True).round(3)
        print(corr)

    # ---------- Combo table (your RQ1 descriptive baseline) ----------
    combo = None
    if {"persuasive_power_index", "content_type", "post_format"}.issubset(df.columns):
        combo = (
            df.groupby(["content_type", "post_format"])["persuasive_power_index"]
            .agg(count="count", mean="mean", median="median")
            .sort_values("mean", ascending=False)
            .reset_index()
        )
        print("\n--- PPI by (content_type x post_format) ---")
        print(combo)

        # Also show smallest combos (balance check)
        combo_counts = (
            df.groupby(["content_type", "post_format"])
            .size()
            .sort_values()
        )
        print("\n--- COMBO SAMPLE SIZES (smallest first) ---")
        print(combo_counts)

        # Plot mean PPI by combo
        labels = combo["content_type"].astype(str) + " | " + combo["post_format"].astype(str)
        plt.figure()
        plt.bar(labels, combo["mean"])
        plt.xticks(rotation=45, ha="right")
        plt.title("Mean persuasive_power_index by post combination")
        plt.tight_layout()
        plt.show()

    # ---------- Influencer distribution (are a few creators dominating?) ----------
    if "influencer_id" in df.columns:
        posts_per_influencer = df.groupby("influencer_id").size().sort_values(ascending=False)
        print("\n--- POSTS PER INFLUENCER (top 10) ---")
        print(posts_per_influencer.head(10))
        print("\nPosts per influencer summary:")
        print(posts_per_influencer.describe().round(2))
    # ----------------------------
    # GRAPHS (descriptive visuals)
    # ----------------------------
    OUTPUT_DIR.mkdir(exist_ok=True)

    # 1) Distribution of persuasive_power_index
    if "persuasive_power_index" in df.columns:
        plt.figure()
        plt.hist(df["persuasive_power_index"].dropna(), bins=40)
        plt.title("Distribution of Persuasive Power Index (PPI)")
        plt.xlabel("persuasive_power_index")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "ppi_distribution_hist.png", dpi=200)
        plt.show()

    # 2) Boxplot: PPI by content_type
    if {"persuasive_power_index", "content_type"}.issubset(df.columns):
        order_ct = ["informational", "experiential", "promotional"]
        order_ct = [x for x in order_ct if x in df["content_type"].unique()]

        data_ct = [df.loc[df["content_type"] == ct, "persuasive_power_index"].dropna() for ct in order_ct]

        plt.figure()
        plt.boxplot(data_ct, labels=order_ct, showfliers=False)
        plt.title("PPI by Content Type (outliers hidden)")
        plt.xlabel("content_type")
        plt.ylabel("persuasive_power_index")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "ppi_by_content_type_boxplot.png", dpi=200)
        plt.show()

    # 3) Boxplot: PPI by post_format
    if {"persuasive_power_index", "post_format"}.issubset(df.columns):
        order_pf = ["image", "video", "carousel"]
        order_pf = [x for x in order_pf if x in df["post_format"].unique()]

        data_pf = [df.loc[df["post_format"] == pf, "persuasive_power_index"].dropna() for pf in order_pf]

        plt.figure()
        plt.boxplot(data_pf, labels=order_pf, showfliers=False)
        plt.title("PPI by Post Format (outliers hidden)")
        plt.xlabel("post_format")
        plt.ylabel("persuasive_power_index")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "ppi_by_post_format_boxplot.png", dpi=200)
        plt.show()

    # 4) Mean PPI by post-combination (bar chart)
    if {"persuasive_power_index", "content_type", "post_format"}.issubset(df.columns):
        combo_means = (
            df.groupby(["content_type", "post_format"])["persuasive_power_index"]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )
        labels = combo_means["content_type"].astype(str) + " | " + combo_means["post_format"].astype(str)

        plt.figure(figsize=(10, 4))
        plt.bar(labels, combo_means["persuasive_power_index"])
        plt.xticks(rotation=45, ha="right")
        plt.title("Mean PPI by Post Combination")
        plt.xlabel("content_type | post_format")
        plt.ylabel("mean persuasive_power_index")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "ppi_mean_by_combo_bar.png", dpi=200)
        plt.show()

    print(f"\nSaved graphs to: {OUTPUT_DIR}")
    # ----------------------------
    # 9-COMBO "FLARE" VISUALS
    # ----------------------------
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Create combo label for plotting
    df["combo"] = df["content_type"].astype(str) + " | " + df["post_format"].astype(str)

    # Order combos by mean PPI (consistent ordering across all plots)
    combo_order = (
        df.groupby("combo")["persuasive_power_index"]
          .mean()
          .sort_values(ascending=False)
          .index
          .tolist()
    )

    def bar_means_by_combo(metric: str, filename: str, title: str) -> None:
        means = df.groupby("combo")[metric].mean().reindex(combo_order)
        plt.figure(figsize=(10, 4))
        plt.bar(means.index, means.values)
        plt.xticks(rotation=45, ha="right")
        plt.title(title)
        plt.xlabel("content_type | post_format")
        plt.ylabel(f"mean {metric}")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / filename, dpi=200)
        plt.show()

    def boxplot_by_combo(metric: str, filename: str, title: str, showfliers: bool = False) -> None:
        data = [df.loc[df["combo"] == c, metric].dropna() for c in combo_order]
        plt.figure(figsize=(10, 4))
        plt.boxplot(data, labels=combo_order, showfliers=showfliers)
        plt.xticks(rotation=45, ha="right")
        plt.title(title + (" (outliers shown)" if showfliers else " (outliers hidden)"))
        plt.xlabel("content_type | post_format")
        plt.ylabel(metric)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / filename, dpi=200)
        plt.show()

    # (1) Mean-by-combo bar charts for early indicators
    for metric, fname, title in [
        ("early_likes", "combo_mean_early_likes_bar.png", "Mean Early Likes by Post Combination"),
        ("early_comments", "combo_mean_early_comments_bar.png", "Mean Early Comments by Post Combination"),
        ("early_shares", "combo_mean_early_shares_bar.png", "Mean Early Shares by Post Combination"),
    ]:
        if metric in df.columns:
            bar_means_by_combo(metric, fname, title)

    # (2) Boxplots by combo (outliers hidden for readability)
    for metric, fname, title in [
        ("early_likes", "combo_box_early_likes.png", "Early Likes by Post Combination"),
        ("early_comments", "combo_box_early_comments.png", "Early Comments by Post Combination"),
        ("early_shares", "combo_box_early_shares.png", "Early Shares by Post Combination"),
    ]:
        if metric in df.columns:
            boxplot_by_combo(metric, fname, title, showfliers=False)

    # (3) Heatmap: combos × metrics (z-scored means)
    heat_metrics = [
        m for m in ["early_likes", "early_comments", "early_shares", "early_engagement_velocity", "persuasive_power_index"]
        if m in df.columns
    ]

    if len(heat_metrics) >= 3:
        combo_means = df.groupby("combo")[heat_metrics].mean().reindex(combo_order)
        combo_z = (combo_means - combo_means.mean()) / combo_means.std(ddof=0)

        plt.figure(figsize=(8, 5))
        plt.imshow(combo_z.values, aspect="auto")
        plt.yticks(range(len(combo_z.index)), combo_z.index)
        plt.xticks(range(len(combo_z.columns)), combo_z.columns, rotation=45, ha="right")
        plt.title("Combo Flare Profile Heatmap (z-scored means)")
        plt.colorbar(label="z-score (higher = above average)")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "combo_flare_profile_heatmap.png", dpi=200)
        plt.show()

        combo_means.round(4).to_csv(OUTPUT_DIR / "combo_means_early_metrics.csv")
        combo_z.round(4).to_csv(OUTPUT_DIR / "combo_means_early_metrics_z.csv")

    # (4) Optional ratio structure: comments/like and shares/like
    if {"early_likes", "early_comments", "early_shares"}.issubset(df.columns):
        df["comments_per_like"] = df["early_comments"] / df["early_likes"].replace(0, np.nan)
        df["shares_per_like"] = df["early_shares"] / df["early_likes"].replace(0, np.nan)

        for metric, fname, title in [
            ("comments_per_like", "combo_mean_comments_per_like_bar.png", "Mean Comments per Like by Post Combination"),
            ("shares_per_like", "combo_mean_shares_per_like_bar.png", "Mean Shares per Like by Post Combination"),
        ]:
            means = df.groupby("combo")[metric].mean().reindex(combo_order)
            plt.figure(figsize=(10, 4))
            plt.bar(means.index, means.values)
            plt.xticks(rotation=45, ha="right")
            plt.title(title)
            plt.xlabel("content_type | post_format")
            plt.ylabel(f"mean {metric}")
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / fname, dpi=200)
            plt.show()

    print(f"\nSaved combo flare visuals to: {OUTPUT_DIR}")
    
    # ---------- Save outputs ----------
    OUTPUT_DIR.mkdir(exist_ok=True)

    miss.round(3).to_csv(OUTPUT_DIR / "missingness_by_column.csv")
    if combo is not None:
        combo.round(6).to_csv(OUTPUT_DIR / "ppi_by_post_combo.csv", index=False)

    if "influencer_id" in df.columns:
        posts_per_influencer.to_frame("posts_count").to_csv(OUTPUT_DIR / "posts_per_influencer.csv")

    if len(corr_cols) >= 2:
        corr.to_csv(OUTPUT_DIR / "correlations_key_vars.csv")

    print(f"\nSaved: {OUTPUT_DIR / 'missingness_by_column.csv'}")
    if combo is not None:
        print(f"Saved: {OUTPUT_DIR / 'ppi_by_post_combo.csv'}")
    if "influencer_id" in df.columns:
        print(f"Saved: {OUTPUT_DIR / 'posts_per_influencer.csv'}")
    if len(corr_cols) >= 2:
        print(f"Saved: {OUTPUT_DIR / 'correlations_key_vars.csv'}")


if __name__ == "__main__":
    main()