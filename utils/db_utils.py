import sqlite3
from pathlib import Path

DB_PATH = Path("data/data.db")

def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)

def initialize_database():
    conn = get_connection()
    cur = conn.cursor()

    # ----------------------------
    # Table: generations
    # ----------------------------
    # Stores raw LLM outputs (EXTENDED for k-based regimes)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS generations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        question_id INTEGER,
        question TEXT,
        answer TEXT,
        model_name TEXT,
        model_size TEXT,
        temperature REAL,
        generation_index INTEGER,      -- index within k
        k_generations INTEGER,         -- total generations used (3,6,10,12)
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # ----------------------------
    # Table: entropy_features
    # ----------------------------
    cur.execute("""
    CREATE TABLE IF NOT EXISTS entropy_features (
        generation_id INTEGER,
        avg_entropy REAL,
        max_entropy REAL,
        FOREIGN KEY (generation_id) REFERENCES generations(id)
    )
    """)

    # ----------------------------
    # Table: high_level_features
    # ----------------------------
    cur.execute("""
    CREATE TABLE IF NOT EXISTS high_level_features (
        generation_id INTEGER,
        self_consistency REAL,
        semantic_stability REAL,
        linguistic_confidence REAL,
        FOREIGN KEY (generation_id) REFERENCES generations(id)
    )
    """)

    # ----------------------------
    # Table: labels
    # ----------------------------
    cur.execute("""
    CREATE TABLE IF NOT EXISTS labels (
        generation_id INTEGER,
        is_correct INTEGER,
        FOREIGN KEY (generation_id) REFERENCES generations(id)
    )
    """)

    conn.commit()
    conn.close()
    print("[OK] Database initialized successfully")

if __name__ == "__main__":
    initialize_database()
