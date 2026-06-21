-- ============================================================
--  Seed data — realistic sample for demo & development
-- ============================================================

USE financial_rag;

-- ─── Industries ──────────────────────────────────────────────────────────────

INSERT INTO industry (name, sector, cyclicality, regulated, top_rating, default_rate_pct, disruption_risk, strategic_importance, notes) VALUES
('Renewable Energy', 'Infrastructure', 'Low',    1, 'AA+',  0.80, 'Low',    'Critical', 'Govt backing; long-term PPAs reduce revenue risk'),
('Pharmaceuticals',  'Healthcare',     'Low',    1, 'AAA',  0.50, 'Medium', 'High',     'Price regulation risk; patent cliff concerns'),
('Steel',            'Manufacturing',  'High',   0, 'AA',   2.50, 'Medium', 'High',     'Highly cyclical; dependent on global HRC prices'),
('IT Services',      'Technology',     'Low',    0, 'AAA',  0.20, 'High',   'Medium',   'AI disruption risk rising; low capital intensity'),
('Real Estate',      'Construction',   'High',   1, 'AA-',  3.20, 'Low',    'Medium',   'RERA regulated; inventory risk in residential'),
('NBFC',             'Financial',      'Medium', 1, 'AAA',  1.10, 'Medium', 'Critical', 'RBI regulated; ALM sensitivity');

-- ─── Promoter groups ─────────────────────────────────────────────────────────

INSERT INTO promoter_group (group_name, founded_year, headquarters, credit_rating, rating_agency, listed_domestic, listed_overseas, overseas_exchange, pe_backed, multilateral_financed, total_group_companies, listed_companies, unlisted_companies) VALUES
('Arjun Energy Group',   1985, 'Mumbai',    'AA+',  'CRISIL', 1, 1, 'SGX',  0, 1, 12, 3, 9),
('Mehta Pharma Holdings',1972, 'Ahmedabad', 'AA',   'ICRA',   1, 0, NULL,   1, 0,  8, 2, 6),
('BharatSteel Corp',     1968, 'Jamshedpur','A+',   'CARE',   1, 0, NULL,   0, 0,  6, 1, 5),
('TechFlow Solutions',   2001, 'Bengaluru', 'AAA',  'CRISIL', 1, 1, 'NYSE', 1, 0,  5, 2, 3);

-- ─── Defaults ────────────────────────────────────────────────────────────────

INSERT INTO promoter_default (group_id, default_date, default_type, haircut_pct, resolution_date, lender_names, description) VALUES
(3, '2016-03-15', 'Restructuring', 18.5, '2018-06-30', 'SBI, PNB, BOB', 'CDR restructuring during steel downcycle; lenders took 18.5% haircut. Resolved via asset sale of non-core division.'),
(3, '2019-11-01', 'Payment Default', 0.0, '2020-02-28', 'HDFC Bank',    '30-day payment delay on working capital; regularised without haircut.');

-- ─── Regulatory actions ───────────────────────────────────────────────────────

INSERT INTO regulatory_action (group_id, entity_name, regulator, action_type, action_date, penalty_amount, currency, resolved, description) VALUES
(2, 'Mehta Pharma Ltd',   'USFDA',  'Warning Letter',  '2021-05-10',  0,         'USD', 1, 'Manufacturing site GMP violation; resolved after CAPA submission in 8 months'),
(4, 'TechFlow India Ltd', 'SEBI',   'Penalty',         '2022-11-20',  2500000,   'INR', 1, 'Delayed disclosure of material information; penalty paid; compliance strengthened'),
(1, 'Arjun Power Ltd',    'CERC',   'Show Cause',      '2023-07-01',  0,         'INR', 0, 'Deviation from approved tariff structure under long-term PPA; matter sub-judice');

-- ─── Directors ───────────────────────────────────────────────────────────────

INSERT INTO director (group_id, name, din, designation, compliant, disqualified, appointment_date) VALUES
(1, 'Vikram Arjun',     '00112233', 'Chairman & MD',          1, 0, '2000-04-01'),
(1, 'Priya Arjun',      '00112244', 'Whole-time Director',    1, 0, '2010-06-15'),
(2, 'Rakesh Mehta',     '00223344', 'Chairman',               1, 0, '1995-01-01'),
(3, 'Suresh Bharat',    '00334455', 'MD & CEO',               1, 0, '1990-07-01'),
(3, 'Dinesh Kumar',     '00334466', 'Independent Director',   0, 1, '2015-01-01'),  -- disqualified
(4, 'Anand Krishnan',   '00445566', 'Founder & CEO',          1, 0, '2001-03-20');

-- ─── Companies ───────────────────────────────────────────────────────────────

INSERT INTO company (name, cin, group_id, industry_id, listed, listing_exchange, listing_date, status, governance_rating, esg_compliant, esg_score, debt_restructured, capital_market_access_freq) VALUES
('Arjun Power Ltd',          'L40100MH1990PLC123456', 1, 1, 1, 'NSE/BSE', '2005-03-18', 'Growth',    'AA',  1, 72.5, 0, 'Annual'),
('Mehta Pharmaceuticals Ltd','L24230GJ1978PLC098765', 2, 2, 1, 'BSE',     '1998-09-10', 'Stable',    'A+',  1, 65.0, 0, 'Bi-Annual'),
('BharatSteel Industries',   'L27100JH1970PLC054321', 3, 3, 1, 'NSE',     '1994-01-25', 'Declining', 'B+',  0, 38.0, 1, 'Irregular'),
('TechFlow Technologies',    'L72200KA2001PLC111111', 4, 4, 1, 'NSE/BSE', '2010-07-05', 'Growth',    'AAA', 1, 88.5, 0, 'Annual');

-- ─── Financials (3 years each) ───────────────────────────────────────────────

INSERT INTO company_financials (company_id, fiscal_year, period, revenue, ebitda, ebit, pat, total_debt, total_equity, cash, capex, fcf, dscr, icr, net_debt_ebitda) VALUES
-- Arjun Power
(1, 2022, 'Annual', 45000, 22000, 18500, 9200,  85000, 42000, 3500, 12000, 5800, 1.42, 4.20, 3.70),
(1, 2023, 'Annual', 52000, 26000, 22000, 11500, 92000, 50000, 4200, 14000, 7100, 1.55, 4.85, 3.38),
(1, 2024, 'Annual', 61000, 31000, 26500, 14200, 98000, 62000, 5100, 16000, 9200, 1.68, 5.30, 2.99),
-- Mehta Pharma
(2, 2022, 'Annual', 28000, 7000,  5800,  3500,  18000, 24000, 2200, 3500,  2800, 1.85, 5.60, 2.26),
(2, 2023, 'Annual', 31000, 7800,  6400,  3900,  19500, 26500, 2800, 4200,  2600, 1.92, 5.80, 2.14),
(2, 2024, 'Annual', 35000, 8900,  7400,  4500,  21000, 30000, 3500, 5000,  2900, 1.98, 6.20, 1.97),
-- BharatSteel
(3, 2022, 'Annual', 55000, 8200,  5500,  1200,  42000, 18000, 800,  2000,  -500, 1.05, 2.10, 5.04),
(3, 2023, 'Annual', 48000, 6500,  3800,  -800,  44000, 17200, 500,  1000, -2200, 0.92, 1.65, 6.69),
(3, 2024, 'Annual', 51000, 7200,  4500,  200,   43000, 17400, 700,  800,  -1500, 0.98, 1.80, 5.88),
-- TechFlow
(4, 2022, 'Annual', 18000, 5400,  5100,  3800,   2000, 28000, 8500, 600,   5200, 4.20, 18.5, -3.25),
(4, 2023, 'Annual', 22000, 6800,  6500,  4900,   1500, 34000, 11000, 700,  6500, 5.80, 24.0, -6.33),
(4, 2024, 'Annual', 27000, 8500,  8100,  6200,   1000, 42000, 14000, 900,  7800, 7.20, 32.0, -15.29);

-- ─── Financing mix ───────────────────────────────────────────────────────────

INSERT INTO financing_mix (company_id, fiscal_year, bank_loans_pct, bonds_ncd_pct, cp_pct, ecb_pct, fccb_pct, mld_pct, other_pct, offshore_pct, hedged_pct, hedge_instrument) VALUES
(1, 2024, 35.0, 30.0, 5.0, 20.0, 5.0, 3.0, 2.0, 28.0, 85.0, 'Cross-currency swaps, Forwards'),
(2, 2024, 50.0, 25.0, 10.0, 10.0, 0.0, 5.0, 0.0, 12.0, 70.0, 'Forward contracts'),
(3, 2024, 65.0, 20.0, 0.0,  10.0, 5.0, 0.0, 0.0, 15.0, 40.0, 'Partial forwards only'),
(4, 2024, 20.0, 10.0, 0.0,  0.0,  0.0, 0.0, 70.0, 5.0, 100.0, 'Natural hedge (USD revenues)');

-- ─── Lender diversification ──────────────────────────────────────────────────

INSERT INTO lender_diversification (company_id, fiscal_year, lender_category, exposure_pct, amount, currency) VALUES
(1, 2024, 'PSU Banks',       25.0, 24500, 'INR'),
(1, 2024, 'Private Banks',   20.0, 19600, 'INR'),
(1, 2024, 'Mutual Funds',    15.0, 14700, 'INR'),
(1, 2024, 'Insurance',       12.0, 11760, 'INR'),
(1, 2024, 'Foreign Lenders', 28.0, 27440, 'INR');

-- ─── Instruments ─────────────────────────────────────────────────────────────

INSERT INTO instrument (company_id, isin, instrument_type, instrument_name, issue_date, maturity_date, face_value, coupon_rate, total_issuance, currency, rating, rating_agency, instrument_rating_differs, corporate_rating, complexity_score, listed, end_use) VALUES
-- Arjun Power NCD Series A
(1, 'INE001A07ABC1', 'NCD', 'Arjun Power NCD Series A 2028', '2023-04-15', '2028-04-15', 1000, 8.75, 5000000000, 'INR', 'AA+', 'CRISIL', 0, 'AA+', 1, 1, 'Capex for 500MW solar expansion'),
-- Arjun Power ECB
(1, NULL, 'ECB', 'Arjun Power USD ECB 2026', '2021-09-01', '2026-09-01', NULL, 5.20, 100000000, 'USD', 'AA+', 'CRISIL', 0, 'AA+', 2, 0, 'Offshore capex and refinancing of existing debt'),
-- Mehta Pharma NCD
(2, 'INE002B07DEF2', 'NCD', 'Mehta Pharma NCD 2027', '2022-06-10', '2027-06-10', 1000, 9.10, 2000000000, 'INR', 'AA-', 'ICRA', 1, 'AA', 2, 1, 'Working capital and R&D investment'),
-- BharatSteel Bond
(3, 'INE003C07GHI3', 'Bond', 'BharatSteel Secured Bond 2025', '2020-03-01', '2025-03-01', 1000, 10.50, 3000000000, 'INR', 'A',   'CARE', 0, 'A', 3, 1, 'Debt consolidation and capex'),
-- TechFlow CP
(4, NULL, 'CP', 'TechFlow Commercial Paper Q1 FY25', '2024-04-01', '2024-07-01', 500000, 7.20, 500000000, 'INR', 'A1+', 'CRISIL', 0, 'AAA', 1, 0, 'Working capital');

-- ─── Covenants ───────────────────────────────────────────────────────────────

INSERT INTO instrument_covenant (instrument_id, covenant_type, description, threshold_value, current_headroom, breach_consequence) VALUES
(1, 'Financial',       'Net Debt / EBITDA',           '<= 4.5x',  'Current 3.0x; headroom 1.5x',  'Accelerate repayment, increase coupon by 50bps'),
(1, 'Financial',       'DSCR',                        '>= 1.25x', 'Current 1.68x; headroom 0.43x', 'Event of default; acceleration'),
(1, 'Negative Pledge', 'No additional charge on PPAs', 'N/A',      'Compliant',                     'Accelerated repayment'),
(1, 'Cross Default',   'Cross default above INR 500Mn', '>= 500Mn', 'No defaults',                  'Event of default'),
(3, 'Financial',       'Net Debt / EBITDA',           '<= 6.0x',  'Current 5.88x; headroom 0.12x', 'Step-up coupon + lender consent required'),
(3, 'Financial',       'DSCR',                        '>= 1.0x',  'Current 0.98x; BREACHED',       'Lender consent required for distributions');

-- ─── Repayment schedules ─────────────────────────────────────────────────────

INSERT INTO repayment_schedule (instrument_id, due_date, principal_amount, interest_amount, paid, paid_date, days_delay) VALUES
-- Arjun Power NCD (bullet at maturity + semi-annual interest)
(1, '2023-10-15', 0,           218750000, 1, '2023-10-15', 0),
(1, '2024-04-15', 0,           218750000, 1, '2024-04-15', 0),
(1, '2024-10-15', 0,           218750000, 1, '2024-10-15', 0),
(1, '2028-04-15', 5000000000,  218750000, 0, NULL,         0),
-- BharatSteel Bond (with one historical delay)
(4, '2021-03-01', 1000000000, 157500000, 1, '2021-03-01', 0),
(4, '2022-03-01', 1000000000, 105000000, 1, '2022-04-02', 32),  -- 32 day delay
(4, '2023-03-01', 1000000000,  52500000, 1, '2023-03-01', 0),
(4, '2025-03-01', 0,            52500000, 0, NULL,         0);

-- ─── Prepayment clauses ───────────────────────────────────────────────────────

INSERT INTO prepayment_clause (instrument_id, allowed, preconditions, make_whole, step_down_schedule, escrow_mechanism) VALUES
(1, 1, 'Min 30-day notice; lender consent if within 24 months of issue; no regulatory objection', 0, 'Year 1: 2%, Year 2: 1.5%, Year 3+: 0%', 'Escrow account funded 3 months before prepayment; waterfall: accrued interest → principal → prepayment premium'),
(2, 1, 'RBI prior approval required for ECB prepayment; min USD 5Mn per tranche', 1, NULL, 'No escrow; direct wire to offshore trustee'),
(4, 0, 'No prepayment permitted without 100% lender consent', 0, NULL, NULL);

-- ─── Options ─────────────────────────────────────────────────────────────────

INSERT INTO instrument_option (instrument_id, option_type, exercisable_by, exercise_date, conditions) VALUES
(1, 'Call',       'Issuer',    '2026-04-15', 'Exercisable at par + accrued interest after Year 3; 30-day notice'),
(3, 'Put',        'Investor',  '2023-03-01', 'Put at par exercisable if rating falls below A-; already exercised by 30% investors');

-- ─── Charge creation ─────────────────────────────────────────────────────────

INSERT INTO charge_creation (instrument_id, charge_type, asset_description, scheduled_date, created_date, as_per_schedule) VALUES
(1, 'Mortgage',       'Solar assets at Rajasthan (500MW)',  '2023-05-15', '2023-05-12', 1),
(1, 'Pledge',         '51% shares of SPV subsidiaries',     '2023-05-15', '2023-06-01', 0),  -- delayed
(3, 'Hypothecation',  'Plant & machinery, Jamshedpur plant', '2020-04-01', '2020-05-20', 0); -- delayed

-- ─── Credit rating history ───────────────────────────────────────────────────

INSERT INTO credit_rating_history (entity_type, entity_id, rating, rating_agency, outlook, effective_date) VALUES
('Company',    1, 'AA',   'CRISIL', 'Positive',  '2020-04-01'),
('Company',    1, 'AA+',  'CRISIL', 'Stable',    '2022-10-15'),
('Company',    1, 'AA+',  'CRISIL', 'Stable',    '2024-04-01'),
('Company',    2, 'AA-',  'ICRA',   'Stable',    '2020-01-01'),
('Company',    2, 'AA',   'ICRA',   'Positive',  '2023-06-01'),
('Company',    3, 'AA-',  'CARE',   'Negative',  '2016-01-01'),
('Company',    3, 'A+',   'CARE',   'Negative',  '2017-06-01'),
('Company',    3, 'A',    'CARE',   'Watch',     '2019-03-01'),
('Instrument', 3, 'A',    'CARE',   'Watch',     '2020-03-01'),
('Company',    4, 'AAA',  'CRISIL', 'Stable',    '2021-07-01');
