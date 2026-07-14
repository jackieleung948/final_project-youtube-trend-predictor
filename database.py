import sqlite3

# https://docs.python.org/3/library/sqlite3.html
# https://www.geeksforgeeks.org/python/introduction-to-sqlite-in-python/

conn = sqlite3.connect("youtube_trends.db")
cursor = conn.cursor()

# Create a table to store video data
cursor.execute('''
CREATE TABLE IF NOT EXISTS videos (
    video_id TEXT PRIMARY KEY,
    title TEXT,
    published_at TEXT,
    description TEXT,
    channel_title TEXT,
    channel_id TEXT,
    category_id TEXT
)''')

# Create a table to store video metrics
cursor.execute('''
CREATE TABLE IF NOT EXISTS metrics (
    video_id TEXT,
    view_velocity REAL,
    collected_at TEXT,
    engagement_ratio REAL,
    view_count INTEGER,
    like_count INTEGER,
    comment_count INTEGER,
    FOREIGN KEY(video_id) REFERENCES videos(video_id)
)''')

# Create a table to store video keywords
cursor.execute('''
CREATE TABLE IF NOT EXISTS keywords (
    video_id TEXT,
    collected_at TEXT,
    source TEXT,
    keyword TEXT,
    relevance_score REAL,
    FOREIGN KEY(video_id) REFERENCES videos(video_id)
)''')

# Commit changes and close the connection
conn.commit()
conn.close()

print("Database and tables created successfully.")