-- ============================================================
--  Agentic RAG — Reference MySQL Schema
--  Covers: Promoters, Company, Financials, Instruments, Industry
-- ============================================================

CREATE DATABASE IF NOT EXISTS financial_rag CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE financial_rag;

-- ─── 1. INDUSTRY ─────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS industry (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    name            VARCHAR(120) NOT NULL,
    sector          VARCHAR(80),
    cyclicality     ENUM('High','Medium','Low') DEFAULT 'Medium',
    regulated       TINYINT(1) DEFAULT 0,
    top_rating      VARCHAR(10),                 -- e.g. AAA, AA+
    default_rate_pct DECIMAL(5,2),               -- historical default rate %
    disruption_risk ENUM('High','Medium','Low') DEFAULT 'Low',
    strategic_importance ENUM('Critical','High','Medium','Low') DEFAULT 'Medium',
    notes           TEXT,
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at      DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- ─── 2. GROUP / PROMOTER ─────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS promoter_group (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    group_name      VARCHAR(200) NOT NULL,
    founded_year    YEAR,
    business_years  INT GENERATED ALWAYS AS (YEAR(CURDATE()) - founded_year) STORED,
    headquarters    VARCHAR(100),
    credit_rating   VARCHAR(10),
    rating_agency   VARCHAR(50),
    listed_domestic TINYINT(1) DEFAULT 0,
    listed_overseas TINYINT(1) DEFAULT 0,
    overseas_exchange VARCHAR(100),              -- e.g. NYSE, SGX, LSE
    pe_backed       TINYINT(1) DEFAULT 0,
    multilateral_financed TINYINT(1) DEFAULT 0, -- IFC, ADB, DEG, etc.
    total_group_companies INT DEFAULT 0,
    listed_companies INT DEFAULT 0,
    unlisted_companies INT DEFAULT 0,
    notes           TEXT,
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS promoter_default (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    group_id        INT NOT NULL,
    default_date    DATE,
    default_type    ENUM('Payment Default','Technical Default','Restructuring','IBC','OTS') NOT NULL,
    haircut_pct     DECIMAL(5,2),               -- % haircut taken by lenders
    resolution_date DATE,
    lender_names    TEXT,
    description     TEXT,
    FOREIGN KEY (group_id) REFERENCES promoter_group(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS regulatory_action (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    group_id        INT,
    entity_name     VARCHAR(200),
    regulator       VARCHAR(100),               -- SEBI, RBI, SFIO, ED, etc.
    action_type     VARCHAR(100),               -- Penalty, Show Cause, Debarment
    action_date     DATE,
    penalty_amount  DECIMAL(18,2),
    currency        CHAR(3) DEFAULT 'INR',
    resolved        TINYINT(1) DEFAULT 0,
    description     TEXT,
    FOREIGN KEY (group_id) REFERENCES promoter_group(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS director (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    group_id        INT,
    name            VARCHAR(200) NOT NULL,
    din             VARCHAR(20),                -- Director Identification Number
    designation     VARCHAR(100),
    compliant       TINYINT(1) DEFAULT 1,
    disqualified    TINYINT(1) DEFAULT 0,
    disqualification_reason TEXT,
    appointment_date DATE,
    cessation_date  DATE,
    FOREIGN KEY (group_id) REFERENCES promoter_group(id) ON DELETE SET NULL
);

-- ─── 3. COMPANY ──────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS company (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    name            VARCHAR(200) NOT NULL,
    cin             VARCHAR(21),                -- Company Identification Number
    group_id        INT,
    industry_id     INT,
    listed          TINYINT(1) DEFAULT 0,
    listing_exchange VARCHAR(100),
    listing_date    DATE,
    delisted        TINYINT(1) DEFAULT 0,
    delist_date     DATE,
    status          ENUM('Growth','Stable','Declining') DEFAULT 'Stable',
    governance_rating VARCHAR(10),
    esg_compliant   TINYINT(1) DEFAULT 0,
    esg_score       DECIMAL(5,2),
    debt_restructured TINYINT(1) DEFAULT 0,
    restructure_type VARCHAR(100),              -- CDR, IBC, OTS, etc.
    capital_market_access_freq VARCHAR(50),     -- Annual, Bi-Annual, Irregular
    website         VARCHAR(255),
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (group_id)    REFERENCES promoter_group(id) ON DELETE SET NULL,
    FOREIGN KEY (industry_id) REFERENCES industry(id)       ON DELETE SET NULL
);

-- ─── 4. FINANCIALS ───────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS company_financials (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    company_id      INT NOT NULL,
    fiscal_year     YEAR NOT NULL,
    period          ENUM('Annual','H1','H2','Q1','Q2','Q3','Q4') DEFAULT 'Annual',
    revenue         DECIMAL(18,2),
    ebitda          DECIMAL(18,2),
    ebit            DECIMAL(18,2),
    pat             DECIMAL(18,2),              -- Profit After Tax
    total_debt      DECIMAL(18,2),
    total_equity    DECIMAL(18,2),
    cash            DECIMAL(18,2),
    capex           DECIMAL(18,2),
    fcf             DECIMAL(18,2),              -- Free Cash Flow
    dscr            DECIMAL(6,3),               -- Debt Service Coverage Ratio
    icr             DECIMAL(6,3),               -- Interest Coverage Ratio
    net_debt_ebitda DECIMAL(6,3),
    currency        CHAR(3) DEFAULT 'INR',
    FOREIGN KEY (company_id) REFERENCES company(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS financing_mix (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    company_id      INT NOT NULL,
    fiscal_year     YEAR NOT NULL,
    bank_loans_pct  DECIMAL(5,2),
    bonds_ncd_pct   DECIMAL(5,2),
    cp_pct          DECIMAL(5,2),               -- Commercial Paper
    ecb_pct         DECIMAL(5,2),               -- External Commercial Borrowing
    fccb_pct        DECIMAL(5,2),               -- Foreign Currency Convertible Bonds
    mld_pct         DECIMAL(5,2),               -- Market Linked Debentures
    other_pct       DECIMAL(5,2),
    offshore_pct    DECIMAL(5,2),
    hedged_pct      DECIMAL(5,2),               -- % of forex exposure hedged
    hedge_instrument VARCHAR(100),              -- Swaps, Forwards, Options
    FOREIGN KEY (company_id) REFERENCES company(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS lender_diversification (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    company_id      INT NOT NULL,
    fiscal_year     YEAR NOT NULL,
    lender_category VARCHAR(100),               -- PSU Bank, Private Bank, MF, Insurance, FPI
    exposure_pct    DECIMAL(5,2),
    amount          DECIMAL(18,2),
    currency        CHAR(3) DEFAULT 'INR',
    FOREIGN KEY (company_id) REFERENCES company(id) ON DELETE CASCADE
);

-- ─── 5. INSTRUMENTS ──────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS instrument (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    company_id      INT NOT NULL,
    isin            VARCHAR(12),
    instrument_type ENUM('NCD','Bond','CP','ECB','FCCB','Loan','MLD','Debenture','Other') NOT NULL,
    instrument_name VARCHAR(200),
    issue_date      DATE,
    maturity_date   DATE,
    face_value      DECIMAL(18,2),
    coupon_rate     DECIMAL(6,3),               -- % per annum
    total_issuance  DECIMAL(18,2),
    currency        CHAR(3) DEFAULT 'INR',
    rating          VARCHAR(10),
    rating_agency   VARCHAR(50),
    instrument_rating_differs TINYINT(1) DEFAULT 0, -- differs from corporate rating
    corporate_rating VARCHAR(10),
    complexity_score INT CHECK (complexity_score BETWEEN 1 AND 5),
    listed          TINYINT(1) DEFAULT 0,
    delisted        TINYINT(1) DEFAULT 0,
    end_use         TEXT,
    FOREIGN KEY (company_id) REFERENCES company(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS instrument_covenant (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    instrument_id   INT NOT NULL,
    covenant_type   ENUM('Financial','Incurrence','Negative Pledge','Cross Default',
                         'Change of Control','Reporting','Other') NOT NULL,
    description     TEXT NOT NULL,
    threshold_value VARCHAR(100),               -- e.g. "Net Debt/EBITDA <= 3.5x"
    current_headroom VARCHAR(100),
    breach_consequence TEXT,
    FOREIGN KEY (instrument_id) REFERENCES instrument(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS repayment_schedule (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    instrument_id   INT NOT NULL,
    due_date        DATE NOT NULL,
    principal_amount DECIMAL(18,2),
    interest_amount DECIMAL(18,2),
    paid            TINYINT(1) DEFAULT 0,
    paid_date       DATE,
    days_delay      INT DEFAULT 0,
    FOREIGN KEY (instrument_id) REFERENCES instrument(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS prepayment_clause (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    instrument_id   INT NOT NULL,
    allowed         TINYINT(1) DEFAULT 0,
    preconditions   TEXT,
    make_whole      TINYINT(1) DEFAULT 0,
    step_down_schedule TEXT,
    escrow_mechanism TEXT,
    FOREIGN KEY (instrument_id) REFERENCES instrument(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS instrument_option (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    instrument_id   INT NOT NULL,
    option_type     ENUM('Call','Put','Conversion','Warrant') NOT NULL,
    exercisable_by  ENUM('Issuer','Investor','Both') NOT NULL,
    exercise_date   DATE,
    exercise_price  DECIMAL(18,6),
    conditions      TEXT,
    FOREIGN KEY (instrument_id) REFERENCES instrument(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS charge_creation (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    instrument_id   INT NOT NULL,
    charge_type     ENUM('FMG','EMG','Pledge','Hypothecation','Mortgage','Lien','Negative Lien') NOT NULL,
    asset_description TEXT,
    scheduled_date  DATE,
    created_date    DATE,
    days_delay      INT GENERATED ALWAYS AS (
                        CASE WHEN created_date IS NOT NULL AND scheduled_date IS NOT NULL
                             THEN DATEDIFF(created_date, scheduled_date)
                             ELSE NULL END
                    ) STORED,
    as_per_schedule TINYINT(1) DEFAULT 1,
    FOREIGN KEY (instrument_id) REFERENCES instrument(id) ON DELETE CASCADE
);

-- ─── 6. CREDIT RATINGS HISTORY ───────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS credit_rating_history (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    entity_type     ENUM('Company','Instrument','Group') NOT NULL,
    entity_id       INT NOT NULL,
    rating          VARCHAR(10) NOT NULL,
    rating_agency   VARCHAR(50) NOT NULL,
    outlook         ENUM('Stable','Positive','Negative','Watch') DEFAULT 'Stable',
    effective_date  DATE NOT NULL,
    withdrawn       TINYINT(1) DEFAULT 0,
    rationale       TEXT
);

-- ─── Indexes ─────────────────────────────────────────────────────────────────

CREATE INDEX idx_company_name       ON company(name);
CREATE INDEX idx_company_group      ON company(group_id);
CREATE INDEX idx_instrument_company ON instrument(company_id);
CREATE INDEX idx_instrument_isin    ON instrument(isin);
CREATE INDEX idx_financials_cy      ON company_financials(company_id, fiscal_year);
CREATE INDEX idx_repayment_due      ON repayment_schedule(instrument_id, due_date);
