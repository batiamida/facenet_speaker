import sqlite3

conn = sqlite3.connect("faces.db")
cur = conn.cursor()

cur.execute("""
    CREATE TABLE IF NOT EXISTS user_info (
        user_id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL
    )
""")

cur.execute("""
    CREATE TABLE IF NOT EXISTS user_playlists (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        playlist_id TEXT NOT NULL,
        playlist_name TEXT,
        playlist_genre TEXT,
        playlist_mood TEXT,
        playlist_description TEXT,
        FOREIGN KEY (user_id) REFERENCES user_info(user_id)
    )
""")

cur.execute("""
    CREATE TABLE IF NOT EXISTS face_info (
        faiss_id INTEGER PRIMARY KEY,
        user_id INTEGER NOT NULL,
        FOREIGN KEY (user_id) REFERENCES user_info(user_id)
    )
""")

conn.commit()
conn.close()