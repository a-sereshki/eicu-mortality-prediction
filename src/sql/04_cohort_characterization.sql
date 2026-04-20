-- 04_cohort_characterization.sql
-- Demographic and clinical characterization of the analytic cohort (n=1,424)
-- Purpose: describe the patient population and identify key prognostic gradients
--
-- =============================================================================
-- Summary findings:
-- =============================================================================
--   - Mean age 63.8 ± 17.5 years (range 18-90), 60% male, mean BMI 30.2
--   - Mortality rises monotonically with age: 1.82% (<40) to 16.67% (85+)
--   - Mortality varies non-monotonically by hospital size, with mid-large
--     hospitals (250-499 beds) showing 12.5% vs 5-7% elsewhere
--   - Apache IVa score shows 28-fold mortality gradient (1.32% to 37.00%)
--   - 16% of cohort is missing Apache score; will require imputation or
--     direct use of component variables in feature engineering


-- =============================================================================
-- Query 1: Demographics overview (Table 1 statistics)
-- =============================================================================
-- Result: mean age 63.8, sd 17.5, range 18-90; 852 male (60%), 572 female; mean BMI 30.2

SELECT 
    ROUND(AVG(age_numeric)::numeric, 1) AS mean_age,
    ROUND(STDDEV(age_numeric)::numeric, 1) AS sd_age,
    MIN(age_numeric) AS min_age,
    MAX(age_numeric) AS max_age,
    SUM(CASE WHEN gender = 'Male' THEN 1 ELSE 0 END) AS n_male,
    SUM(CASE WHEN gender = 'Female' THEN 1 ELSE 0 END) AS n_female,
    SUM(CASE WHEN gender NOT IN ('Male', 'Female') OR gender IS NULL THEN 1 ELSE 0 END) AS n_other_unknown,
    ROUND(AVG(admissionweight / NULLIF((admissionheight/100.0)*(admissionheight/100.0), 0))::numeric, 1) AS mean_bmi
FROM cohort;


-- =============================================================================
-- Query 2: Mortality stratified by age group
-- =============================================================================
-- Result: monotonic gradient confirms age as strong univariate predictor
--   <40:    165 patients,  3 deaths,  1.82%
--   40-54:  214 patients, 10 deaths,  4.67%
--   55-69:  429 patients, 26 deaths,  6.06%
--   70-84:  472 patients, 55 deaths, 11.65%
--   85+:    144 patients, 24 deaths, 16.67%

SELECT 
    CASE 
        WHEN age_numeric < 40 THEN '1. <40'
        WHEN age_numeric < 55 THEN '2. 40-54'
        WHEN age_numeric < 70 THEN '3. 55-69'
        WHEN age_numeric < 85 THEN '4. 70-84'
        ELSE '5. 85+'
    END AS age_group,
    COUNT(*) AS n_patients,
    SUM(CASE WHEN hospitaldischargestatus = 'Expired' THEN 1 ELSE 0 END) AS deaths,
    ROUND(100.0 * SUM(CASE WHEN hospitaldischargestatus = 'Expired' THEN 1 ELSE 0 END) / COUNT(*), 2) AS mortality_pct
FROM cohort
GROUP BY age_group
ORDER BY age_group;


-- =============================================================================
-- Query 3: Mortality stratified by hospital bed category
-- =============================================================================
-- Result: NON-monotonic pattern with mid-large hospitals as outlier
--   <100:        277 patients, 15 deaths,  5.42%
--   100-249:     465 patients, 32 deaths,  6.88%
--   250-499:     264 patients, 33 deaths, 12.50%   <-- outlier
--   >= 500:      207 patients, 13 deaths,  6.28%
--
-- This pattern motivates the cross-hospital generalization analysis: it
-- raises the question of whether models trained on one hospital size
-- category generalize accurately to others.

SELECT 
    h.numbedscategory AS bed_category,
    COUNT(*) AS n_patients,
    SUM(CASE WHEN c.hospitaldischargestatus = 'Expired' THEN 1 ELSE 0 END) AS deaths,
    ROUND(100.0 * SUM(CASE WHEN c.hospitaldischargestatus = 'Expired' THEN 1 ELSE 0 END) / COUNT(*), 2) AS mortality_pct
FROM cohort c
JOIN hospital h ON c.hospitalid = h.hospitalid
WHERE h.numbedscategory IS NOT NULL
GROUP BY h.numbedscategory
ORDER BY 
    CASE h.numbedscategory
        WHEN '<100' THEN 1
        WHEN '100 - 249' THEN 2
        WHEN '250 - 499' THEN 3
        WHEN '>= 500' THEN 4
    END;


-- =============================================================================
-- Query 4: Mortality stratified by Apache IVa severity score
-- =============================================================================
-- Result: dramatic 28-fold mortality gradient validates Apache as a powerful
-- predictor; 234 missing scores (16%) require handling in feature engineering
--   <30 (low):          152 patients,  2 deaths,  1.32%
--   30-59 (moderate):   608 patients, 25 deaths,  4.11%
--   60-89 (high):       330 patients, 38 deaths, 11.52%
--   90+ (very high):    100 patients, 37 deaths, 37.00%
--   no Apache score:    234 patients, 16 deaths,  6.84%

SELECT 
    CASE 
        WHEN a.apachescore < 30 THEN '1. <30 (low)'
        WHEN a.apachescore < 60 THEN '2. 30-59 (moderate)'
        WHEN a.apachescore < 90 THEN '3. 60-89 (high)'
        WHEN a.apachescore IS NOT NULL THEN '4. 90+ (very high)'
        ELSE '5. (no Apache score)'
    END AS apache_group,
    COUNT(*) AS n_patients,
    SUM(CASE WHEN c.hospitaldischargestatus = 'Expired' THEN 1 ELSE 0 END) AS deaths,
    ROUND(100.0 * SUM(CASE WHEN c.hospitaldischargestatus = 'Expired' THEN 1 ELSE 0 END) / COUNT(*), 2) AS mortality_pct
FROM cohort c
LEFT JOIN apachepatientresult a 
    ON c.patientunitstayid = a.patientunitstayid 
    AND a.apacheversion = 'IVa'
GROUP BY apache_group
ORDER BY apache_group;
