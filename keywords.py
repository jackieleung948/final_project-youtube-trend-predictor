import pandas as pd
import sqlite3
from keybert import KeyBERT


def preprocess_text(text):
    """
    Preprocess the text by replacing # with space to handle hashtags
    """
    if isinstance(text, str):
        return text.replace("#", " ")
    return text


def main():

    # Read from SQLite database to get video titles and descriptions
    conn = sqlite3.connect("youtube_trends.db")
    df = pd.read_sql_query(
        "SELECT video_id, title, description FROM videos", conn)

    # https://maartengr.github.io/KeyBERT/
    # https://github.com/MaartenGr/KeyBERT
    # Extract keywords from video titles
    # Using paraphrase-multilingual-MiniLM-L12-v2 to support multiple languages, as some videos may have non-English titles or descriptions
    # https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
    kw_model = KeyBERT(model="paraphrase-multilingual-MiniLM-L12-v2")
    df["title_keywords"] = df["title"].apply(lambda x: kw_model.extract_keywords(
        preprocess_text(x), keyphrase_ngram_range=(1, 2), top_n=3, stop_words="english"))

    # Extract keywords from video descriptions
    df["description_keywords"] = df["description"].apply(lambda x: kw_model.extract_keywords(preprocess_text(
        x), keyphrase_ngram_range=(1, 2), top_n=5, stop_words="english") if isinstance(x, str) and len(x) > 0 else [])

    # Filter out keywords below a 0.5 relevance score threshold (0.5 was chosen based on a query from 50 runs of data)
    min_relevance_score = 0.5
    df["title_keywords"] = df["title_keywords"].apply(lambda keywords: [(
        kw, score) for kw, score in keywords if score >= min_relevance_score])
    df["description_keywords"] = df["description_keywords"].apply(lambda keywords: [(
        kw, score) for kw, score in keywords if score >= min_relevance_score])
    
    # Add custom stopwords related to cooking as they appear in nearly all videos and don't provide meaningful differentiation for trend prediction
    custom_stopwords = ["recipe", "shorts", "food", "cooking", "viral", "trending",
                        "ytshorts", "short", "reels", "youtube", "shortsfeed",
                        "shortvideo", "youtubeshorts", "homecooking", "easyrecipe",
                        "viralrecipe", "foodie", "homemade", "streetfood", "video", "cookingvideo", "dailyvlog",
                        "funny", "comedy", "ingredients", "recipes",
                        "viralvideo", "lifestyle"]
    df["title_keywords"] = df["title_keywords"].apply(lambda keywords: [
                                                      kw for kw in keywords if not any(word in custom_stopwords for word in kw[0].lower().split())])
    df["description_keywords"] = df["description_keywords"].apply(lambda keywords: [
                                                                  kw for kw in keywords if not any(word in custom_stopwords for word in kw[0].lower().split())])

    # Save keywords to SQLite database
    cursor = conn.cursor()

    for i, row in df.iterrows():
        video_id = row["video_id"]
        collected_at = pd.Timestamp.now(tz='UTC').isoformat()

        # Save title keywords
        for keyword, relevance_score in row["title_keywords"]:
            source = "title"
            cursor.execute('''
                INSERT INTO keywords (video_id, collected_at, source, keyword, relevance_score)
                VALUES (?, ?, ?, ?, ?)
            ''', (video_id, collected_at, source, keyword, relevance_score))

        # Save description keywords
        for keyword, relevance_score in row["description_keywords"]:
            source = "description"
            cursor.execute('''
                INSERT INTO keywords (video_id, collected_at, source, keyword, relevance_score)
                VALUES (?, ?, ?, ?, ?)
            ''', (video_id, collected_at, source, keyword, relevance_score))

    print("Data saved to keywords table in youtube_trends.db")

    conn.commit()
    conn.close()


if __name__ == "__main__":
    main()
