-- Statistik transaksi fraud vs non-fraud
SELECT 
  Class,
  ROUND(AVG(amount), 2) AS avg_amount,
  ROUND(STDDEV(amount), 2) AS std_amount,
  ROUND(MIN(amount), 2) AS min_amount,
  ROUND(MAX(amount), 2) AS max_amount
FROM fraud.transactions
GROUP BY class;

-- Korelasi antara amount dan contoh beberapa fitur utama
SELECT 
  CORR(amount, v1) AS corr_v1,
  CORR(amount, v2) AS corr_v2,
  CORR(amount, v3) AS corr_v3
FROM fraud.transactions;

-- Distribusi jumlah transaksi di bawah threshold tertentu
SELECT 
  CASE 
    WHEN amount < 1 THEN '< 1'
    WHEN amount < 10 THEN '1-10'
    WHEN amount < 100 THEN '10-100'
    ELSE '> 100'
  END AS amount_range,
  class,
  COUNT(*) AS count
FROM fraud.transactions
GROUP BY amount_range, class
ORDER BY amount_range, class;
