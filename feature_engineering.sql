-- ============================================================
-- FEATURE ENGINEERING (thay cho Dagster/Spark) - chay tren PostgreSQL
-- Tuong ung GD1-GD2 trong README_RecommendationSystem.md hien tai
-- ============================================================


CREATE TABLE IF NOT EXISTS interactions (
    event_time      TIMESTAMP,
    event_type      TEXT,          -- 'view' | 'cart' | 'purchase'
    product_id      BIGINT,
    category_id     BIGINT,
    category_code   TEXT,
    brand           TEXT,
    price           NUMERIC,
    user_id         BIGINT,
    user_session    TEXT
);

-- ============================================================
-- GD1: gan trong so hanh vi (View=1, Cart=2, Purchase=3)
-- ============================================================
CREATE OR REPLACE VIEW interactions_weighted AS
SELECT
    *,
    CASE event_type
        WHEN 'view'     THEN 1
        WHEN 'cart'     THEN 2
        WHEN 'purchase' THEN 3
        ELSE 0
    END AS weight
FROM interactions;

-- ============================================================
-- GD2: feature theo USER (RFM-style, dung cho segment logic
-- cua Hybrid model: loyal / casual / cold-start)
-- ============================================================
CREATE OR REPLACE VIEW user_features AS
SELECT
    user_id,
    COUNT(*)                                                   AS frequency,
    SUM(weight)                                                AS total_weighted_score,
    MAX(event_time)                                            AS last_interaction_at,
    EXTRACT(DAY FROM (NOW() - MAX(event_time)))::INT           AS recency_days,
    SUM(CASE WHEN event_type = 'purchase' THEN price ELSE 0 END) AS monetary,
    COUNT(*) FILTER (WHERE event_type = 'purchase')            AS purchase_count,
    -- segment: khop dung logic Hybrid trong README hien tai
    CASE
        WHEN COUNT(*) >= 2 THEN 'loyal'      -- -> Weighted Hybrid (ALS 60% + Content 40%)
        WHEN COUNT(*) = 1  THEN 'casual'     -- -> Content-Based tren san pham xem gan nhat
        ELSE 'cold_start'                    -- -> Trending
    END AS segment
FROM interactions_weighted
GROUP BY user_id;

-- ============================================================
-- Feature theo PRODUCT (dung lam popularity/trending fallback)
-- ============================================================
CREATE OR REPLACE VIEW product_features AS
SELECT
    product_id,
    category_code,
    brand,
    AVG(price)                                          AS avg_price,
    SUM(weight)                                          AS popularity_score,
    COUNT(*) FILTER (WHERE event_type = 'purchase')      AS purchase_count
FROM interactions_weighted
GROUP BY product_id, category_code, brand
ORDER BY popularity_score DESC;

-- Trending top-10 dung khi user chua co lich su (cold-start)
CREATE OR REPLACE VIEW trending_top10 AS
SELECT product_id, popularity_score
FROM product_features
ORDER BY popularity_score DESC
LIMIT 10;

-- ============================================================
-- San pham xem gan nhat cua tung user (dung cho Content-Based fallback)
-- ============================================================
CREATE OR REPLACE VIEW user_last_viewed_product AS
SELECT DISTINCT ON (user_id)
    user_id, product_id, event_time
FROM interactions_weighted
ORDER BY user_id, event_time DESC;
