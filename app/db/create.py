import sqlite3
import os

## the function create_users_table Stores app users.
# id: auto-incrementing primary key.
# username: must be unique.
# password: stored in plain text (note: consider hashing for real-world use).

create_users_table = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE,
    password TEXT NOT NULL
);
"""

## the function create_chats_table represents a conversation owned by a user_id.

create_chats_table = """
CREATE TABLE IF NOT EXISTS chats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    user_id INTEGER NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id)
);
"""

create_images_table = """
CREATE TABLE IF NOT EXISTS images (
    id TEXT PRIMARY KEY,
    input_prompt_id TEXT,
    output_prompt_id TEXT,
    user_id INTEGER NOT NULL,
    chat_id INTEGER NOT NULL,
    prompt_guidance FLOAT,
    image_guidance FLOAT,
    path TEXT NOT NULL UNIQUE,
    FOREIGN KEY (input_prompt_id) REFERENCES prompts(id),
    FOREIGN KEY (output_prompt_id) REFERENCES prompts(id),
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (chat_id) REFERENCES chats(id)
);
"""

# the function create_prompts_table stores individual prompts used in chats.
# Every prompt is tied to a chat (chat_id) and user (user_id).
# Additional metadata: suggestion used (used_suggestion), image linkages, 
# enhancement status (is_enhanced).
create_prompts_table = """
CREATE TABLE IF NOT EXISTS prompts (
    id TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL,
    chat_id INTEGER NOT NULL,
    prompt TEXT NOT NULL,
    depth INTEGER NOT NULL,
    used_suggestion BOOLEAN NOT NULL DEFAULT 0,
    modified_suggestion BOOLEAN NOT NULL DEFAULT 0,
    suggestion_used TEXT,
    is_enhanced BOOLEAN NOT NULL DEFAULT 0,
    enhanced_prompt TEXT,
    image_in_id INTEGER,
    images_out TEXT,
    FOREIGN KEY (chat_id) REFERENCES chats(id),
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (image_in_id) REFERENCES images(id)
);
"""

create_bertscore_metrics_table = """
CREATE TABLE IF NOT EXISTS bertscore_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prompt_id TEXT NOT NULL,
    previous_prompt_id TEXT,
    user_id INTEGER NOT NULL,
    chat_id INTEGER NOT NULL,
    depth INTEGER NOT NULL,
    bert_novelty FLOAT,
    FOREIGN KEY (prompt_id) REFERENCES prompts(id),
    FOREIGN KEY (previous_prompt_id) REFERENCES prompts(id),
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (chat_id) REFERENCES chats(id)
);
"""

create_lpips_metrics_table = """
CREATE TABLE IF NOT EXISTS lpips_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id TEXT NOT NULL,
    previous_image_id TEXT,
    user_id INTEGER NOT NULL,
    chat_id INTEGER NOT NULL,
    depth INTEGER NOT NULL,
    lpips FLOAT DEFAULT NULL,
    FOREIGN KEY (image_id) REFERENCES images(id),
    FOREIGN KEY (previous_image_id) REFERENCES images(id),
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (chat_id) REFERENCES chats(id)
);
"""

create_guidance_metrics_table = """
CREATE TABLE IF NOT EXISTS guidance_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prompt_id TEXT NOT NULL,
    user_id INTEGER NOT NULL,
    chat_id INTEGER NOT NULL,
    depth INTEGER NOT NULL,
    prompt_guidance FLOAT,
    image_guidance FLOAT,
    FOREIGN KEY (prompt_id) REFERENCES prompts(id),
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (chat_id) REFERENCES chats(id)
);
"""

# pct = percentage
create_functionality_metrics_table = """
CREATE TABLE IF NOT EXISTS functionality_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    chat_id INTEGER NOT NULL,
    used_suggestion_pct FLOAT,
    used_enhancement_pct FLOAT,
    used_both_pct FLOAT,
    no_ai_pct FLOAT,
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (chat_id) REFERENCES chats(id)
);
"""

create_prompt_word_metrics_table = """
CREATE TABLE IF NOT EXISTS prompt_word_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prompt_id TEXT NOT NULL,
    user_id INTEGER,
    chat_id INTEGER,
    depth INTEGER,
    full_text TEXT,
    word_count INTEGER,
    relevant_words TEXT,
    FOREIGN KEY (prompt_id) REFERENCES prompts(id),
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (chat_id) REFERENCES chats(id)
);
"""


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(script_dir, "mma.db")

    con = sqlite3.connect(db_path)
    cur = con.cursor()

    cur.execute(create_users_table)
    cur.execute(create_chats_table)
    cur.execute(create_prompts_table)
    cur.execute(create_images_table)
    cur.execute(create_bertscore_metrics_table)
    cur.execute(create_lpips_metrics_table)
    cur.execute(create_guidance_metrics_table)
    cur.execute(create_functionality_metrics_table)
    cur.execute(create_prompt_word_metrics_table)

    
    con.commit()
    con.close()
