# ## ============================================================
# ## BANA 620 - Research Project Analysis
# ## RQ: Does content design configuration (content type + post format)
# ##     significantly predict early persuasive power?
# ## ============================================================

# import pandas as pd
# import numpy as np
# from scipy import stats
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import r2_score
# import warnings
# warnings.filterwarnings('ignore')

# # ── Load the dataset ──────────────────────────────────────────
# df = pd.read_csv("dataset_algorithmic_persuasion_10000.csv")
# print("Dataset loaded successfully.")
# print(f"Rows: {df.shape[0]}  |  Columns: {df.shape[1]}\n")


# ## ============================================================
# ## FEATURE / CONSTRUCT CREATION
# ## ============================================================

# # 1. content_design_config — combines content_type + post_format into one label
# #    e.g. "informational_image", "promotional_video", etc.
# df['content_design_config'] = df['content_type'] + '_' + df['post_format']

# # 2. early_like_rate — likes earned per minute of the early window
# df['early_like_rate'] = df['early_likes'] / df['early_window_mins']

# # 3. early_share_rate — shares earned per minute of the early window
# df['early_share_rate'] = df['early_shares'] / df['early_window_mins']

# # 4. early_comment_rate — comments earned per minute of the early window
# df['early_comment_rate'] = df['early_comments'] / df['early_window_mins']

# # 5. engagement_momentum — already in dataset as early_engagement_velocity
# # 6. authority_log       — already in dataset (log-transformed follower authority)
# # 7. verified            — already in dataset as 0/1

# print("=== CONSTRUCTS / FEATURES CREATED ===")
# print("1. content_design_config   : content_type + post_format combined (e.g. 'informational_image')")
# print("2. early_like_rate         : early_likes / early_window_mins")
# print("3. early_share_rate        : early_shares / early_window_mins")
# print("4. early_comment_rate      : early_comments / early_window_mins")
# print("5. engagement_momentum     : early_engagement_velocity (already in dataset)")
# print("6. authority_log           : log of follower count (already in dataset)")
# print("7. verified                : 1 = verified account, 0 = not verified (already in dataset)")

# print()
# print("Content Design Config groups:")
# print(df['content_design_config'].value_counts())
# print()


# ## ============================================================
# ## HELPER: ONE-WAY ANOVA + GROUP MEANS  (H1 - H4)
# ## ============================================================

# def run_anova(group_col, outcome_col, outcome_label):
#     print(f"\n{'='*60}")
#     print(f"  Outcome: {outcome_label}")
#     print(f"{'='*60}")

#     # Group means table
#     means = df.groupby(group_col)[outcome_col].agg(['mean', 'std', 'count'])
#     means.columns = ['Mean', 'Std Dev', 'N']
#     print("\nGroup Means:")
#     print(means.round(4).to_string())

#     # Collect each group's values
#     groups = [grp[outcome_col].dropna().values
#               for _, grp in df.groupby(group_col)]

#     # Run one-way ANOVA
#     f_stat, p_value = stats.f_oneway(*groups)

#     print(f"\nOne-Way ANOVA Result:")
#     print(f"  F-statistic = {f_stat:.4f}")
#     print(f"  p-value     = {p_value:.4f}")

#     if p_value < 0.05:
#         print(f"  → SIGNIFICANT  (p < 0.05)")
#         print(f"  → Content design configuration DOES significantly predict {outcome_label}.")
#     else:
#         print(f"  → NOT SIGNIFICANT  (p >= 0.05)")
#         print(f"  → Content design configuration does NOT significantly predict {outcome_label}.")

#     return f_stat, p_value


# ## ============================================================
# ## H1: Content design configuration predicts engagement momentum
# ## ============================================================
# print("\n\n" + "#"*60)
# print("# HYPOTHESIS 1 (H1)")
# print("# H1: Content design configuration significantly predicts")
# print("#     overall engagement momentum (velocity).")
# print("#"*60)

# run_anova('content_design_config', 'early_engagement_velocity', 'Engagement Momentum (Velocity)')


# ## ============================================================
# ## H2: Content design configuration predicts early like rate
# ## ============================================================
# print("\n\n" + "#"*60)
# print("# HYPOTHESIS 2 (H2)")
# print("# H2: Content design configuration significantly predicts")
# print("#     the rate of early likes.")
# print("#"*60)

# run_anova('content_design_config', 'early_like_rate', 'Early Like Rate (likes/min)')


# ## ============================================================
# ## H3: Content design configuration predicts early share rate
# ## ============================================================
# print("\n\n" + "#"*60)
# print("# HYPOTHESIS 3 (H3)")
# print("# H3: Content design configuration significantly predicts")
# print("#     the rate of early shares.")
# print("#"*60)

# run_anova('content_design_config', 'early_share_rate', 'Early Share Rate (shares/min)')


# ## ============================================================
# ## H4: Content design configuration predicts early comment rate
# ## ============================================================
# print("\n\n" + "#"*60)
# print("# HYPOTHESIS 4 (H4)")
# print("# H4: Content design configuration significantly predicts")
# print("#     the rate of early comments.")
# print("#"*60)

# run_anova('content_design_config', 'early_comment_rate', 'Early Comment Rate (comments/min)')


# ## ============================================================
# ## HELPER: LINEAR REGRESSION WITH MODERATION  (H5 - H6)
# ## Uses sklearn LinearRegression + scipy for p-values
# ## ============================================================

# def run_linear_regression(outcome_col, outcome_label, moderator_col, moderator_label):
#     print(f"\n{'='*60}")
#     print(f"  Outcome   : {outcome_label}")
#     print(f"  Moderator : {moderator_label}")
#     print(f"{'='*60}")

#     # --- Build dummy variables ---
#     # content_type: 'informational' is the reference group (dropped)
#     df['ct_promotional']  = (df['content_type'] == 'promotional').astype(int)
#     df['ct_experiential'] = (df['content_type'] == 'experiential').astype(int)

#     # post_format: 'image' is the reference group (dropped)
#     df['pf_video']    = (df['post_format'] == 'video').astype(int)
#     df['pf_carousel'] = (df['post_format'] == 'carousel').astype(int)

#     # --- Standardize the moderator (subtract mean, divide by std) ---
#     # This puts it on a common scale so coefficients are easier to compare
#     mod_mean = df[moderator_col].mean()
#     mod_std  = df[moderator_col].std()
#     df['moderator_z'] = (df[moderator_col] - mod_mean) / mod_std

#     # --- Interaction terms: moderator x each dummy ---
#     df['int_promo_x_mod'] = df['ct_promotional']  * df['moderator_z']
#     df['int_exper_x_mod'] = df['ct_experiential'] * df['moderator_z']
#     df['int_video_x_mod'] = df['pf_video']        * df['moderator_z']
#     df['int_carsl_x_mod'] = df['pf_carousel']     * df['moderator_z']

#     # --- Build the predictor (X) and outcome (y) ---
#     feature_names = [
#         'ct_promotional',
#         'ct_experiential',
#         'pf_video',
#         'pf_carousel',
#         'moderator_z',
#         'int_promo_x_mod',
#         'int_exper_x_mod',
#         'int_video_x_mod',
#         'int_carsl_x_mod',
#     ]

#     X = df[feature_names]
#     y = df[outcome_col]

#     # --- Fit the model using sklearn LinearRegression ---
#     model = LinearRegression()
#     model.fit(X, y)

#     # --- Predictions and R-squared ---
#     y_pred  = model.predict(X)
#     r2      = r2_score(y, y_pred)

#     # Adjusted R-squared (penalizes for adding extra variables)
#     n = len(y)
#     k = len(feature_names)
#     r2_adj = 1 - (1 - r2) * (n - 1) / (n - k - 1)

#     # --- Calculate p-values manually using scipy ---
#     # sklearn does not give p-values, so we compute them from residuals
#     residuals  = y - y_pred
#     RSS        = np.sum(residuals**2)           # sum of squared errors
#     s2         = RSS / (n - k - 1)              # variance of residuals

#     # Add intercept column to X for variance calculation
#     X_with_intercept = np.column_stack([np.ones(n), X])
#     var_beta = s2 * np.linalg.inv(X_with_intercept.T @ X_with_intercept)
#     se_all   = np.sqrt(np.diag(var_beta))       # standard errors

#     # Intercept is index 0; features start at index 1
#     intercept_se = se_all[0]
#     coef_se      = se_all[1:]

#     all_coefs = np.concatenate([[model.intercept_], model.coef_])
#     all_se    = np.concatenate([[intercept_se], coef_se])
#     all_names = ['Intercept'] + feature_names

#     t_stats  = all_coefs / all_se
#     p_values = [2 * (1 - stats.t.cdf(abs(t), df=n-k-1)) for t in t_stats]

#     # --- Overall F-statistic ---
#     TSS    = np.sum((y - y.mean())**2)
#     ESS    = TSS - RSS
#     F_stat = (ESS / k) / (RSS / (n - k - 1))
#     F_pval = 1 - stats.f.cdf(F_stat, dfn=k, dfd=n-k-1)

#     # --- Print coefficient table ---
#     print(f"\nCoefficients (Reference: informational content, image format):")
#     print(f"{'Variable':<22} {'Coefficient':>12} {'Std Error':>10} {'t-stat':>8} {'p-value':>10}  Significant?")
#     print("-" * 72)
#     for i, name in enumerate(all_names):
#         sig = "YES *" if p_values[i] < 0.05 else "no"
#         print(f"{name:<22} {all_coefs[i]:>12.4f} {all_se[i]:>10.4f} {t_stats[i]:>8.3f} {p_values[i]:>10.4f}  {sig}")

#     # --- Print model fit ---
#     print(f"\nModel Fit:")
#     print(f"  R-squared          = {r2:.4f}  ({r2*100:.2f}% of variance explained)")
#     print(f"  Adjusted R-squared = {r2_adj:.4f}")
#     print(f"  F-statistic        = {F_stat:.4f}  (p = {F_pval:.4f})")

#     if F_pval < 0.05:
#         print(f"\n  → Overall model is SIGNIFICANT (p < 0.05)")
#     else:
#         print(f"\n  → Overall model is NOT SIGNIFICANT (p >= 0.05)")

#     # --- Interpret the interaction terms ---
#     print(f"\n  Interaction Terms (Moderation Effect of {moderator_label}):")
#     interaction_indices = [6, 7, 8, 9]  # positions in all_names list
#     any_sig = False
#     for i in interaction_indices:
#         sig = "SIGNIFICANT *" if p_values[i] < 0.05 else "not significant"
#         print(f"    {all_names[i]:<25} p = {p_values[i]:.4f}  → {sig}")
#         if p_values[i] < 0.05:
#             any_sig = True

#     if any_sig:
#         print(f"\n  → {moderator_label} DOES moderate the relationship between")
#         print(f"    content design configuration and {outcome_label}.")
#     else:
#         print(f"\n  → {moderator_label} does NOT significantly moderate the relationship.")


# ## ============================================================
# ## H5: Authority log MODERATES content design → engagement
# ## ============================================================
# print("\n\n" + "#"*60)
# print("# HYPOTHESIS 5 (H5)")
# print("# H5: Influencer authority log moderates the relation between")
# print("#     content design configuration and engagement.")
# print("#"*60)

# run_linear_regression(
#     outcome_col     = 'early_engagement_velocity',
#     outcome_label   = 'Engagement Momentum (Velocity)',
#     moderator_col   = 'authority_log',
#     moderator_label = 'Authority Log'
# )


# ## ============================================================
# ## H6: Verified status MODERATES content design → engagement
# ## ============================================================
# print("\n\n" + "#"*60)
# print("# HYPOTHESIS 6 (H6)")
# print("# H6: Influencer verification status moderates the relation")
# print("#     between content design configuration and engagement.")
# print("#"*60)

# run_linear_regression(
#     outcome_col     = 'early_engagement_velocity',
#     outcome_label   = 'Engagement Momentum (Velocity)',
#     moderator_col   = 'verified',
#     moderator_label = 'Verification Status'
# )

# print("\n\n=== ANALYSIS COMPLETE ===")










## ============================================================
## BANA 620 - Research Project Analysis
## RQ: Does content design configuration (content type + post format)
##     significantly predict early persuasive power?
## ============================================================

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# ── Load the dataset ──────────────────────────────────────────
df = pd.read_csv("dataset_algorithmic_persuasion_10000.csv")
print("Dataset loaded successfully.")
print(f"Rows: {df.shape[0]}  |  Columns: {df.shape[1]}\n")


## ============================================================
## FEATURE / CONSTRUCT CREATION
## ============================================================

# 1. content_design_config — combines content_type + post_format into one label
#    e.g. "informational_image", "promotional_video", etc.
df['content_design_config'] = df['content_type'] + '_' + df['post_format']

# 2. early_like_rate — likes earned per minute of the early window
df['early_like_rate'] = df['early_likes'] / df['early_window_mins']

# 3. early_share_rate — shares earned per minute of the early window
df['early_share_rate'] = df['early_shares'] / df['early_window_mins']

# 4. early_comment_rate — comments earned per minute of the early window
df['early_comment_rate'] = df['early_comments'] / df['early_window_mins']

# 5. engagement_momentum — already in dataset as early_engagement_velocity
# 6. authority_log       — already in dataset (log-transformed follower authority)
# 7. verified            — already in dataset as 0/1

print("=== CONSTRUCTS / FEATURES CREATED ===")
print("1. content_design_config   : content_type + post_format combined (e.g. 'informational_image')")
print("2. early_like_rate         : early_likes / early_window_mins")
print("3. early_share_rate        : early_shares / early_window_mins")
print("4. early_comment_rate      : early_comments / early_window_mins")
print("5. engagement_momentum     : early_engagement_velocity (already in dataset)")
print("6. authority_log           : log of follower count (already in dataset)")
print("7. verified                : 1 = verified account, 0 = not verified (already in dataset)")

print()
print("Content Design Config groups:")
print(df['content_design_config'].value_counts())
print()


## ============================================================
## HELPER: ONE-WAY ANOVA + GROUP MEANS  (H1 - H4)
## ============================================================

def run_anova(group_col, outcome_col, outcome_label):
    print(f"\n{'='*60}")
    print(f"  Outcome: {outcome_label}")
    print(f"{'='*60}")

    # Group means table
    means = df.groupby(group_col)[outcome_col].agg(['mean', 'std', 'count'])
    means.columns = ['Mean', 'Std Dev', 'N']
    print("\nGroup Means:")
    print(means.round(4).to_string())

    # Collect each group's values
    groups = [grp[outcome_col].dropna().values
              for _, grp in df.groupby(group_col)]

    # Run one-way ANOVA
    f_stat, p_value = stats.f_oneway(*groups)

    print(f"\nOne-Way ANOVA Result:")
    print(f"  F-statistic = {f_stat:.4f}")
    print(f"  p-value     = {p_value:.4f}")

    if p_value < 0.05:
        print(f"  → SIGNIFICANT  (p < 0.05)")
        print(f"  → Content design configuration DOES significantly predict {outcome_label}.")
    else:
        print(f"  → NOT SIGNIFICANT  (p >= 0.05)")
        print(f"  → Content design configuration does NOT significantly predict {outcome_label}.")

    return f_stat, p_value


## ============================================================
## H1: Content design configuration predicts engagement momentum
## ============================================================
print("\n\n" + "#"*60)
print("# HYPOTHESIS 1 (H1)")
print("# H1: Content design configuration significantly predicts")
print("#     overall engagement momentum (velocity).")
print("#"*60)

run_anova('content_design_config', 'early_engagement_velocity', 'Engagement Momentum (Velocity)')


## ============================================================
## H2: Content design configuration predicts early like rate
## ============================================================
print("\n\n" + "#"*60)
print("# HYPOTHESIS 2 (H2)")
print("# H2: Content design configuration significantly predicts")
print("#     the rate of early likes.")
print("#"*60)

run_anova('content_design_config', 'early_like_rate', 'Early Like Rate (likes/min)')


## ============================================================
## H3: Content design configuration predicts early share rate
## ============================================================
print("\n\n" + "#"*60)
print("# HYPOTHESIS 3 (H3)")
print("# H3: Content design configuration significantly predicts")
print("#     the rate of early shares.")
print("#"*60)

run_anova('content_design_config', 'early_share_rate', 'Early Share Rate (shares/min)')


## ============================================================
## H4: Content design configuration predicts early comment rate
## ============================================================
print("\n\n" + "#"*60)
print("# HYPOTHESIS 4 (H4)")
print("# H4: Content design configuration significantly predicts")
print("#     the rate of early comments.")
print("#"*60)

run_anova('content_design_config', 'early_comment_rate', 'Early Comment Rate (comments/min)')


## ============================================================
## HELPER: LINEAR REGRESSION WITH MODERATION  (H5 - H6)
## Uses sklearn LinearRegression + scipy for p-values
## ============================================================

def run_linear_regression(outcome_col, outcome_label, moderator_col, moderator_label):
    print(f"\n{'='*60}")
    print(f"  Outcome   : {outcome_label}")
    print(f"  Moderator : {moderator_label}")
    print(f"{'='*60}")

    # --- Build dummy variables ---
    # content_type: 'informational' is the reference group (dropped)
    df['ct_promotional']  = (df['content_type'] == 'promotional').astype(int)
    df['ct_experiential'] = (df['content_type'] == 'experiential').astype(int)

    # post_format: 'image' is the reference group (dropped)
    df['pf_video']    = (df['post_format'] == 'video').astype(int)
    df['pf_carousel'] = (df['post_format'] == 'carousel').astype(int)

    # --- Standardize the moderator (subtract mean, divide by std) ---
    # This puts it on a common scale so coefficients are easier to compare
    mod_mean = df[moderator_col].mean()
    mod_std  = df[moderator_col].std()
    df['moderator_z'] = (df[moderator_col] - mod_mean) / mod_std

    # --- Interaction terms: moderator x each dummy ---
    df['int_promo_x_mod'] = df['ct_promotional']  * df['moderator_z']
    df['int_exper_x_mod'] = df['ct_experiential'] * df['moderator_z']
    df['int_video_x_mod'] = df['pf_video']        * df['moderator_z']
    df['int_carsl_x_mod'] = df['pf_carousel']     * df['moderator_z']

    # --- Build the predictor (X) and outcome (y) ---
    feature_names = [
        'ct_promotional',
        'ct_experiential',
        'pf_video',
        'pf_carousel',
        'moderator_z',
        'int_promo_x_mod',
        'int_exper_x_mod',
        'int_video_x_mod',
        'int_carsl_x_mod',
    ]

    X = df[feature_names]
    y = df[outcome_col]

    # --- Fit the model using sklearn LinearRegression ---
    model = LinearRegression()
    model.fit(X, y)

    # --- Predictions and R-squared ---
    y_pred  = model.predict(X)
    r2      = r2_score(y, y_pred)

    # Adjusted R-squared (penalizes for adding extra variables)
    n = len(y)
    k = len(feature_names)
    r2_adj = 1 - (1 - r2) * (n - 1) / (n - k - 1)

    # --- Calculate p-values manually using scipy ---
    # sklearn does not give p-values, so we compute them from residuals
    residuals  = y - y_pred
    RSS        = np.sum(residuals**2)           # sum of squared errors
    s2         = RSS / (n - k - 1)              # variance of residuals

    # Add intercept column to X for variance calculation
    X_with_intercept = np.column_stack([np.ones(n), X])
    var_beta = s2 * np.linalg.inv(X_with_intercept.T @ X_with_intercept)
    se_all   = np.sqrt(np.diag(var_beta))       # standard errors

    # Intercept is index 0; features start at index 1
    intercept_se = se_all[0]
    coef_se      = se_all[1:]

    all_coefs = np.concatenate([[model.intercept_], model.coef_])
    all_se    = np.concatenate([[intercept_se], coef_se])
    all_names = ['Intercept'] + feature_names

    t_stats  = all_coefs / all_se
    p_values = [2 * (1 - stats.t.cdf(abs(t), df=n-k-1)) for t in t_stats]

    # --- Overall F-statistic ---
    TSS    = np.sum((y - y.mean())**2)
    ESS    = TSS - RSS
    F_stat = (ESS / k) / (RSS / (n - k - 1))
    F_pval = 1 - stats.f.cdf(F_stat, dfn=k, dfd=n-k-1)

    # --- Print coefficient table ---
    print(f"\nCoefficients (Reference: informational content, image format):")
    print(f"{'Variable':<22} {'Coefficient':>12} {'Std Error':>10} {'t-stat':>8} {'p-value':>10}  Significant?")
    print("-" * 72)
    for i, name in enumerate(all_names):
        sig = "YES *" if p_values[i] < 0.05 else "no"
        print(f"{name:<22} {all_coefs[i]:>12.4f} {all_se[i]:>10.4f} {t_stats[i]:>8.3f} {p_values[i]:>10.4f}  {sig}")

    # --- Print model fit ---
    print(f"\nModel Fit:")
    print(f"  R-squared          = {r2:.4f}  ({r2*100:.2f}% of variance explained)")
    print(f"  Adjusted R-squared = {r2_adj:.4f}")
    print(f"  F-statistic        = {F_stat:.4f}  (p = {F_pval:.4f})")

    if F_pval < 0.05:
        print(f"\n  → Overall model is SIGNIFICANT (p < 0.05)")
    else:
        print(f"\n  → Overall model is NOT SIGNIFICANT (p >= 0.05)")

    # --- Interpret the interaction terms ---
    print(f"\n  Interaction Terms (Moderation Effect of {moderator_label}):")
    interaction_indices = [6, 7, 8, 9]  # positions in all_names list
    any_sig = False
    for i in interaction_indices:
        sig = "SIGNIFICANT *" if p_values[i] < 0.05 else "not significant"
        print(f"    {all_names[i]:<25} p = {p_values[i]:.4f}  → {sig}")
        if p_values[i] < 0.05:
            any_sig = True

    if any_sig:
        print(f"\n  → {moderator_label} DOES moderate the relationship between")
        print(f"    content design configuration and {outcome_label}.")
    else:
        print(f"\n  → {moderator_label} does NOT significantly moderate the relationship.")


## ============================================================
## H5: Authority log MODERATES content design → engagement
## ============================================================
print("\n\n" + "#"*60)
print("# HYPOTHESIS 5 (H5)")
print("# H5: Influencer authority log moderates the relation between")
print("#     content design configuration and engagement.")
print("#"*60)

run_linear_regression(
    outcome_col     = 'early_engagement_velocity',
    outcome_label   = 'Engagement Momentum (Velocity)',
    moderator_col   = 'authority_log',
    moderator_label = 'Authority Log'
)


## ============================================================
## H6: Verified status MODERATES content design → engagement
## ============================================================
print("\n\n" + "#"*60)
print("# HYPOTHESIS 6 (H6)")
print("# H6: Influencer verification status moderates the relation")
print("#     between content design configuration and engagement.")
print("#"*60)

run_linear_regression(
    outcome_col     = 'early_engagement_velocity',
    outcome_label   = 'Engagement Momentum (Velocity)',
    moderator_col   = 'verified',
    moderator_label = 'Verification Status'
)

print("\n\n=== ANALYSIS COMPLETE ===")