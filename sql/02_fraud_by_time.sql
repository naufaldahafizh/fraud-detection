-- Asumsikan bahwa Time adalah detik sejak awal (bukan timestamp)
-- Tambahkan kolom hour dari kolom Time
WITH transactions_by_hour AS (
  SELECT 
    FLOOR("Time" / 3600) AS hour,
    "Class",
    COUNT(*) AS total
  FROM fraud.transactions
  GROUP BY hour, Class
)

-- Agregasi: jumlah fraud vs non-fraud per jam
SELECT 
  hour,
  SUM(CASE WHEN Class = 0 THEN total ELSE 0 END) AS non_fraud,
  SUM(CASE WHEN Class = 1 THEN total ELSE 0 END) AS fraud
FROM transactions_by_hour
GROUP BY hour
ORDER BY hour;
