-- Jumlah total transaksi
SELECT class, COUNT(*) FROM fraud.transactions GROUP BY class;

-- Jumlah transaksi fraud vs non-fraud
SELECT 
  Class,
  COUNT(*) AS count,
  ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS percentage
FROM fraud.transactions
GROUP BY Class;

-- Rata-rata dan maksimum Amount pada tiap Class
SELECT 
  Class,
  ROUND(AVG(Amount), 2) AS avg_amount,
  MAX(Amount) AS max_amount
FROM fraud.transactions
GROUP BY Class;