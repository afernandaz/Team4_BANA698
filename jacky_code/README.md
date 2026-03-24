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
<img width="1000" height="400" alt="Figure_14" src="https://github.com/user-attachments/assets/0b13524c-cefc-494a-ae80-c27581662693" />
<img width="1000" height="400" alt="Figure_13" src="https://github.com/user-attachments/assets/c93839da-62da-469a-bdc0-2676fb41b768" />
<img width="800" height="500" alt="Figure_12" src="https://github.com/user-attachments/assets/11eebfc9-0559-412d-95a8-aa8b536eefd3" />
<img width="1000" height="400" alt="Figure_11" src="https://github.com/user-attachments/assets/f6f49241-4cd3-495d-8853-d7e95307a9d8" />
<img width="1000" height="400" alt="Figure_10" src="https://github.com/user-attachments/assets/236284f7-3113-4226-a73e-0982520fb714" />
<img width="1000" height="400" alt="Figure_9" src="https://github.com/user-attachments/assets/cfdd0379-c4b0-489a-b8cb-b71aa463b733" />
<img width="1000" height="400" alt="Figure_8" src="https://github.com/user-attachments/assets/ab328d4c-b0d9-4282-a7aa-5e8b77a96845" />
<img width="1000" height="400" alt="Figure_7" src="https://github.com/user-attachments/assets/d566ec36-42bd-4d04-8c01-63ef75e98924" />
<img width="1000" height="400" alt="Figure_6" src="https://github.com/user-attachments/assets/3b80528b-aa09-43e1-9341-29889ebd639f" />
<img width="1000" height="400" alt="Figure_5" src="https://github.com/user-attachments/assets/1d343f31-5a70-4832-b724-9511d2714e77" />
<img width="640" height="480" alt="Figure_4" src="https://github.com/user-attachments/assets/f08462a4-1a1f-41bf-af89-f094bf2d9a33" />
<img width="640" height="480" alt="Figure_3" src="https://github.com/user-attachments/assets/65961a04-d142-4302-b10d-5ae26bd58cc7" />
<img width="640" height="480" alt="Figure_2" src="https://github.com/user-attachments/assets/335dfe18-d7d6-4f8f-b6ba-c88555097e28" />
