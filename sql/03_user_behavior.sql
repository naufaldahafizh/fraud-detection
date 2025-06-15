-- Statistik transaksi fraud vs non-fraud
SELECT 
  Class,
  ROUND(AVG("Amount"), 2) AS avg_amount,
  ROUND(STDDEV("Amount"), 2) AS std_amount,
  ROUND(MIN("Amount"), 2) AS min_amount,
  ROUND(MAX("Amount"), 2) AS max_amount
FROM fraud.transactions
GROUP BY Class;

-- Korelasi antara amount dan contoh beberapa fitur utama
SELECT 
  CORR("Amount", "V1") AS corr_v1,
  CORR("Amount", "V2") AS corr_v2,
  CORR("Amount", "V3") AS corr_v3
FROM fraud.transactions;

-- Distribusi jumlah transaksi di bawah threshold tertentu
SELECT 
  CASE 
    WHEN Amount < 1 THEN '< 1'
    WHEN Amount < 10 THEN '1-10'
    WHEN Amount < 100 THEN '10-100'
    ELSE '> 100'
  END AS amount_range,
  Class,
  COUNT(*) AS count
FROM fraud.transactions
GROUP BY amount_range, Class
ORDER BY amount_range, Class;
