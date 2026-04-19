-- 03_cohort_definition.sql
-- Analytic cohort for the ICU mortality prediction model
-- 
-- This file defines a reusable SQL view (`cohort`) that applies the study's
-- inclusion criteria to the eICU Demo dataset. All downstream analyses query
-- FROM cohort rather than FROM patient.
--
-- =============================================================================
-- Inclusion criteria (applied in order):
-- =============================================================================
--   1. First ICU stay per patient only    (prevents data leakage in train/test)
--   2. Adult patients (age >= 18 or > 89) (pediatric physiology is distinct)
--   3. Non-missing hospitaldischargestatus (mortality outcome must be known)
--   4. ICU stay >= 24 hours (1440 min)    (required for 24hr feature window)
--
-- All admitted ICU unit types are retained; only adult ICU types are present
-- in eICU, so no unit-type filter is needed.
--
-- =============================================================================
-- Consort flow (observed patient counts at each filter step):
-- =============================================================================
--   Step 0 — All patient stays                           2,520
--   Step 1 — First ICU stay per patient only            2,174  (-346)
--   Step 2 — Adults only                                 2,166  (-8)
--   Step 3 — Non-missing discharge status                2,141  (-25)
--   Step 4 — ICU stay >= 24 hours (final cohort)         1,424  (-717)
--
-- Final cohort: 1,424 patients across 186 hospitals
-- Mortality rate: 8.29% (118 deaths)
-- Retention: 56.5% of original dataset
--
-- Known limitation: patients who died within 24 hours of ICU admission are
-- excluded. The model is not applicable to very early-death patients.

-- =============================================================================
-- Cohort view definition
-- =============================================================================

DROP VIEW IF EXISTS cohort CASCADE;

CREATE VIEW cohort AS
WITH patient_with_numeric_age AS (
    -- Convert age from text to integer (HIPAA de-identification codes '> 89' for age >= 90)
    SELECT 
        p.*,
        CASE 
            WHEN p.age = '> 89' THEN 90
            WHEN p.age ~ '^[0-9]+$' THEN p.age::integer
            ELSE NULL
        END AS age_numeric
    FROM patient p
),
first_stay_per_patient AS (
    -- Identify the earliest ICU admission for each patient
    SELECT 
        patienthealthsystemstayid,
        MIN(hospitaladmitoffset) AS first_admit_offset
    FROM patient_with_numeric_age
    GROUP BY patienthealthsystemstayid
)
SELECT p.*
FROM patient_with_numeric_age p
INNER JOIN first_stay_per_patient f
    ON p.patienthealthsystemstayid = f.patienthealthsystemstayid
    AND p.hospitaladmitoffset = f.first_admit_offset
WHERE 
    p.age_numeric >= 18                             -- Rule 1: adults
    AND p.hospitaldischargestatus IS NOT NULL       -- Rule 2: outcome known
    AND p.hospitaldischargestatus != ''             -- Rule 2: outcome not blank
    AND p.unitdischargeoffset >= 1440               -- Rule 3: >= 24hr stay
;


-- =============================================================================
-- Cohort size verification
-- =============================================================================

-- Expected: 1,424 patients, 118 deaths, 8.29% mortality, 186 hospitals
SELECT 
    COUNT(*) AS cohort_size,
    SUM(CASE WHEN hospitaldischargestatus = 'Expired' THEN 1 ELSE 0 END) AS deaths,
    ROUND(100.0 * SUM(CASE WHEN hospitaldischargestatus = 'Expired' THEN 1 ELSE 0 END) / COUNT(*), 2) AS mortality_rate_pct,
    COUNT(DISTINCT hospitalid) AS hospitals_represented
FROM cohort;


-- =============================================================================
-- Consort flow diagram data
-- =============================================================================
-- Generates the patient count at each filtering step, for Figure 1 of README

WITH counts AS (
    SELECT 
        'Step 0: All patient stays in eICU Demo' AS step,
        COUNT(*) AS n_patients,
        1 AS sort_order
    FROM patient
    
    UNION ALL
    
    SELECT 
        'Step 1: First ICU stay per patient only' AS step,
        COUNT(DISTINCT patienthealthsystemstayid) AS n_patients,
        2 AS sort_order
    FROM patient
    
    UNION ALL
    
    SELECT 
        'Step 2: Adult patients (age >= 18 or > 89)' AS step,
        COUNT(DISTINCT p.patienthealthsystemstayid) AS n_patients,
        3 AS sort_order
    FROM patient p
    WHERE 
        (p.age = '> 89' OR (p.age ~ '^[0-9]+$' AND p.age::integer >= 18))
    
    UNION ALL
    
    SELECT 
        'Step 3: Non-missing discharge status' AS step,
        COUNT(DISTINCT p.patienthealthsystemstayid) AS n_patients,
        4 AS sort_order
    FROM patient p
    WHERE 
        (p.age = '> 89' OR (p.age ~ '^[0-9]+$' AND p.age::integer >= 18))
        AND p.hospitaldischargestatus IS NOT NULL
        AND p.hospitaldischargestatus != ''
    
    UNION ALL
    
    SELECT 
        'Step 4: ICU stay >= 24 hours (final cohort)' AS step,
        COUNT(*) AS n_patients,
        5 AS sort_order
    FROM cohort
)
SELECT step, n_patients
FROM counts
ORDER BY sort_order;
