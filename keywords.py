import pandas as pd
from keybert import KeyBERT

# Read CSV file
df = pd.read_csv("youtube_videos_with_metrics.csv")

# https://maartengr.github.io/KeyBERT/
# https://github.com/MaartenGr/KeyBERT
# Extract keywords from video titles
kw_model = KeyBERT()
df["title_keywords"] = df["title"].apply(lambda x: kw_model.extract_keywords(x, keyphrase_ngram_range=(1,2),top_n=3, stop_words="english"))

# Extract keywords from video descriptions
df["description_keywords"] = df["description"].apply(lambda x: kw_model.extract_keywords(x, keyphrase_ngram_range=(1,2), top_n=10, stop_words="english") if isinstance(x, str) and len(x) > 0 else [])

# Add custom stopwords related to cooking
custom_stopwords = ["recipe", "shorts", "food", "cooking", "viral", "trending", "ytshorts", "short", "reels", "youtube"]
df["title_keywords"] = df["title_keywords"].apply(lambda keywords: [kw for kw in keywords if not any(word in custom_stopwords for word in kw[0].lower().split())])
df["description_keywords"] = df["description_keywords"].apply(lambda keywords: [kw for kw in keywords if not any(word in custom_stopwords for word in kw[0].lower().split())])

# Save updated DataFrame to new CSV
df.to_csv("youtube_videos_with_keywords.csv", index=False)
print("Keywords extracted and saved to youtube_videos_with_keywords.csv")