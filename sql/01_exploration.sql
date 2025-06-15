-- Berapa banyak fraud?
SELECT class, COUNT(*) FROM fraud.transactions GROUP BY class;

-- Total amount transaksi fraud vs normal
SELECT class, SUM(amount) FROM fraud.transactions GROUP BY class;