-- 01_dataset_overview.sql
-- Initial exploration of the eICU Collaborative Research Database Demo v2.0.1
-- Purpose: characterize cohort size, hospital coverage, and outcome distribution

-- =============================================================================
-- Query 1: Dataset scale
-- =============================================================================
-- Expected: ~2,174 patients, ~2,520 ICU stays, hospitals in reference table

SELECT 
    COUNT(DISTINCT patienthealthsystemstayid) AS unique_patients,
    COUNT(DISTINCT patientunitstayid) AS unique_icu_stays,
    COUNT(DISTINCT hospitalid) AS hospitals_with_patients
FROM patient;


-- =============================================================================
-- Query 2: Hospital reference table coverage
-- =============================================================================
-- The hospital table contains 186 hospital records (lookup table for full eICU)
-- This query confirms how many hospitals contributed actual patient data

SELECT COUNT(*) AS hospitals_in_reference_table FROM hospital;


-- =============================================================================
-- Query 3: Overall in-hospital mortality distribution
-- =============================================================================
-- Target variable distribution for the mortality prediction model
-- Expected: ~8-12% mortality (typical ICU range)
-- Note: 28 patients have missing discharge status; these are excluded from modeling

SELECT 
    COALESCE(hospitaldischargestatus, '(missing)') AS discharge_status,
    COUNT(*) AS n,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS percentage
FROM patient
GROUP BY hospitaldischargestatus
ORDER BY n DESC;
