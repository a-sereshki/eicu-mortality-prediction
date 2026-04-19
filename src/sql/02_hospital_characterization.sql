-- 02_hospital_characterization.sql
-- Characterize the hospital network in the eICU Demo (v2.0.1)
-- Purpose: identify viable stratification dimensions for cross-hospital analysis
--
-- Key findings:
--   - 186 hospitals contribute patients (full eICU network, not a subset)
--   - Bed count is the cleanest stratification: 4 balanced categories
--   - Teaching status usable but imbalanced (168 non-teaching vs 18 teaching)
--   - Region usable for 3 of 4 categories (Northeast underpowered at 13)
--   - 28 hospitals missing bed category, 18 missing region


-- =============================================================================
-- Query 1: Teaching vs non-teaching
-- Result: 168 non-teaching, 18 teaching (9:1 imbalance)
-- =============================================================================

SELECT 
    COALESCE(teachingstatus::text, '(missing)') AS teaching_status,
    COUNT(*) AS hospital_count
FROM hospital
GROUP BY teachingstatus
ORDER BY hospital_count DESC;


-- =============================================================================
-- Query 2: Bed count category (PRIMARY stratification dimension)
-- Result: 4 categories with balanced counts: 61, 39, 35, 23 (plus 28 null)
-- =============================================================================

SELECT 
    COALESCE(numbedscategory, '(missing)') AS bed_category,
    COUNT(*) AS hospital_count
FROM hospital
GROUP BY numbedscategory
ORDER BY hospital_count DESC;


-- =============================================================================
-- Query 3: Region
-- Result: Midwest 62, South 54, West 39, Northeast 13 (plus 18 null)
-- =============================================================================

SELECT 
    COALESCE(region, '(missing)') AS region,
    COUNT(*) AS hospital_count
FROM hospital
GROUP BY region
ORDER BY hospital_count DESC;
