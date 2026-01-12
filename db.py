import sqlite3

def get_connection():
    return sqlite3.connect("exoplanets.db")

def create_table():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS planets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        pl_rade REAL,
        pl_bmasse REAL,
        pl_orbsmax REAL,
        st_teff REAL,
        st_met REAL,
        st_luminosity REAL,
        pl_luminosity REAL,
        stellar_compatibility_index REAL
    )
    """)

    conn.commit()
    conn.close()

if __name__ == "__main__":
    create_table()
