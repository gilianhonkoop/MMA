import sqlite3
import os

create_images_table = """
CREATE TABLE IF NOT EXISTS images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    image_path TEXT NOT NULL UNIQUE,
    guidance FLOAT NOT NULL,
    strength FLOAT NOT NULL,
    prompt TEXT NOT NULL,
    prompt_number INTEGER NOT NULL,
    enhanced_prompt BOOLEAN NOT NULL DEFAULT 0,
    selected BOOLEAN NOT NULL DEFAULT 0,
    FOREIGN KEY (user_id) REFERENCES users(id)
);
"""

create_users_table = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE
);
"""

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(script_dir, "mma.db")

    con = sqlite3.connect(db_path)
    cur = con.cursor()

    cur.execute(create_images_table)
    cur.execute(create_users_table)
    
    con.commit()
    con.close()
