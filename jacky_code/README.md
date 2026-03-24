testcode_1.py is the overall descriptive file.
What the script does
Loads the dataset (dataset_algorithmic_persuasion_10000.xlsx)
Runs data-quality checks (missing values, duplicates, date range)
Summarizes key variables (early likes/comments/shares, reach, impressions, authority, PPI)
Builds the 9 content-type × format combinations (“post combos”)
Generates “flare” visuals showing how each combo performs across:
early_likes, early_comments, early_shares
persuasive_power_index
early_engagement_velocity
Runs OLS regression (RQ1) and prints hypothesis support
Key descriptive findings (high-level)
Dataset is clean (no missing values, no duplicates)
Early engagement window is standardized (60 minutes)
Highest average PPI appears in Experiential + Video, followed by Promotional + Video, then Experiential + Carousel
Combo “flare” plots show how each post configuration distributes across likes/comments/shares (not just averages)

<img width="640" height="480" alt="Figure_1" src="https://github.com/user-attachments/assets/a59d7b02-bcdc-41b1-bc50-092afea73776" />
