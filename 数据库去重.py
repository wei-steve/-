import sqlite3
import pandas as pd

def clean_duplicate_timestamps(db_path, coin_name='LEAUSDT', interval='1m'):
    conn = sqlite3.connect(db_path)
    query = f"""
        SELECT timestamp, COUNT(*) as count
        FROM kline_data
        WHERE coin_name = ? AND interval = ?
        GROUP BY timestamp
        HAVING count > 1
    """
    df = pd.read_sql_query(query, conn, params=(coin_name, interval))
    if not df.empty:
        print(f"检测到 {len(df)} 个重复时间戳")
        for ts in df['timestamp']:
            query = f"""
                DELETE FROM kline_data
                WHERE coin_name = ? AND interval = ? AND timestamp = ?
                AND rowid NOT IN (
                    SELECT MIN(rowid)
                    FROM kline_data
                    WHERE coin_name = ? AND interval = ? AND timestamp = ?
                )
            """
            conn.execute(query, (coin_name, interval, ts, coin_name, interval, ts))
        conn.commit()
        print("已清理重复时间戳")
    else:
        print("无重复时间戳")
    conn.close()

if __name__ == '__main__':
    clean_duplicate_timestamps(r'D:\策略研究\kline_db_new\kline_data_LEAUSDT.db')