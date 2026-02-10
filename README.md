# Email Security Analyzer Ultimate

<img width="2752" height="1492" alt="unnamed" src="https://github.com/user-attachments/assets/524533d3-2791-4e77-b651-ac7b250bb926" />

A professional desktop application for comprehensive email security analysis, breach detection, and threat intelligence. Built with Python and CustomTkinter.

**Version:** 6.0.2 | **Python:** 3.7+ (optimized for 3.13) | **Platform:** Windows

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![License](https://img.shields.io/badge/License-Private-red)
![Lines of Code](https://img.shields.io/badge/Lines-15%2C900%2B-green)

---

## Features

<img width="3020" height="7137" alt="NotebookLM Mind Map" src="https://github.com/user-attachments/assets/ba74fcb2-c2db-49e4-9037-7547bf0edecd" />

### Core Analysis
- **Email Risk Scoring** - Multi-factor risk assessment (0-100) combining domain reputation, DNS validation, breach history, pattern analysis, and ML predictions
- **Dark Web Breach Detection** - Parallel queries to LeakCheck.io, XposedOrNot, and EmailRep.io APIs with intelligent deduplication and caching
- **DNS Security Validation** - SPF, DMARC, and DKIM record analysis with misconfiguration detection
- **Domain Reputation** - WHOIS age analysis, domain registration patterns, and reputation scoring

### Machine Learning
- **XGBoost + Random Forest Ensemble** - Calibrated classifiers with 99.6% precision targeting
- **44 Real Features** - Email address patterns, domain signals, DNS scores, breach data, disposable detection, typosquatting, Shannon entropy
- **Isolation Forest** - Unsupervised anomaly detection trained on legitimate email patterns
- **Feedback Loop** - User feedback stored in SQLite, auto-retraining at milestones (50/100/250/500/1000 samples)
- **Disposable Email Detection** - 720+ known disposable/temporary email domains
- **Typosquatting Detection** - Levenshtein distance + homoglyph analysis against major providers

### Threat Intelligence

<img width="1367" height="693" alt="image" src="https://github.com/user-attachments/assets/46e8760d-88a2-4ee3-b0e8-e67f5539bdc1" />

- **MITRE ATT&CK Framework** - Maps detected threats to ATT&CK techniques with semantic search via sentence-transformers and FAISS
- **TAXII 2.1 Integration** - Pulls latest ATT&CK data from MITRE TAXII server with GitHub and built-in fallbacks
- **Real-time Monitoring** - Continuous email stream monitoring with live threat detection

### Reporting
- **Enterprise HTML Reports** - Professional dark-themed reports with breach cards, risk gauges, and ML analysis
- **Enterprise Excel Reports** - Multi-sheet workbooks with charts, conditional formatting, and mitigation tracking
- **PDF Export** - Full analysis reports via HTML-to-PDF conversion
- **Bulk Processing** - Concurrent analysis of email lists with aggregated statistics

### UI/UX

<img width="1287" height="701" alt="image" src="https://github.com/user-attachments/assets/4eec8095-0220-4bb5-b0f8-2d9ace70e470" />

<img width="1336" height="703" alt="image" src="https://github.com/user-attachments/assets/11757802-fe6f-41b3-90b5-5853baa9dc71" />

- **Modern Dark Theme** - Glass-morphism effects, gradient buttons, particle animations
- **13-Page Navigation** - Dashboard, Email Analysis, Bulk Scanner, Real-time Monitor, MITRE ATT&CK, Threat Intelligence, ML Models, DNS Security, Analytics, Export Reports, Audit Log, Settings, About
- **Interactive Charts** - Matplotlib-powered analytics: risk distribution, threat heatmaps, MITRE top techniques, ML model performance, confidence distribution
- **Real-time Feedback** - "Safe"/"Risky" buttons on results to improve ML models over time

---

## Architecture

<img width="1288" height="691" alt="image" src="https://github.com/user-attachments/assets/eff23453-0963-41aa-990e-64bb754902a8" />

```
+------------------------------------------------------------------+
|                          main.py                                 |
|           (Entry Point, Dependency Check, Logging)               |
+------------------------------------------------------------------+
                               |
                               v
+------------------------------------------------------------------+
|                 EmailSecurityAnalyzerGUI.py                      |
|            (6,500 lines - Main GUI Application)                  |
|                                                                  |
|  +------------------+  +------------------+  +----------------+  |
|  |    Dashboard     |  | Email Analysis   |  | Bulk Scanner   |  |
|  +------------------+  +------------------+  +----------------+  |
|  +------------------+  +------------------+  +----------------+  |
|  |  RT Monitor      |  | MITRE ATT&CK     |  | Threat Intel   |  |
|  +------------------+  +------------------+  +----------------+  |
|  +------------------+  +------------------+  +----------------+  |
|  |   ML Models      |  |  DNS Security    |  |   Analytics    |  |
|  +------------------+  +------------------+  +----------------+  |
|  +------------------+  +------------------+  +----------------+  |
|  | Export Reports   |  |   Audit Log      |  |   Settings     |  |
|  +------------------+  +------------------+  +----------------+  |
+------------------------------------------------------------------+
          |                    |                       |
          v                    v                       v
+-------------------+  +------------------+  +-------------------+
| EmailSecurity     |  | ThreatIntel      |  | MachineLearning   |
| Analyzer.py       |  | Engine.py        |  | Engine.py         |
| (Risk Scoring,    |  | (Breach APIs,    |  | (XGBoost + RF,    |
|  Feature Extract, |  |  WHOIS, DNS,     |  |  Isolation Forest,|
|  Threat Detect)   |  |  Domain Rep,     |  |  Feedback Loop,   |
|                   |  |  SQLite Cache)   |  |  Auto-Retrain)    |
+-------------------+  +------------------+  +-------------------+
          |                    |                       |
          v                    v                       v
+-------------------+  +------------------+  +-------------------+
| Disposable        |  | MitreAttack      |  | BulkProcessing    |
| EmailDetector.py  |  | Framework.py     |  | Engine.py         |
| (720+ domains)    |  | (ATT&CK mapping, |  | (Concurrent       |
|                   |  |  FAISS search)   |  |  email analysis)  |
| Typosquatting     |  |                  |  |                   |
| Detector.py       |  | MITRETAXIIConn   |  +-------------------+
| (Levenshtein +    |  | ection.py        |
|  homoglyphs)      |  | (STIX2/TAXII)    |         |
+-------------------+  |                  |         v
                        | TechniqueRetr    |  +-------------------+
                        | iever.py         |  | Enterprise        |
                        | (GitHub fallback)|  | ReportGenerator   |
                        +------------------+  | .py (HTML/PDF)    |
                                              |                   |
                                              | EnterpriseExcel   |
                                              | ReportGenerator   |
                                              | .py (XLSX)        |
                                              +-------------------+

+------------------------------------------------------------------+
|                        UI Widgets                                |
|  AnimatedProgressBar | CircularProgress | GradientButton         |
|  ParticleEffect      | RadarChart       |                        |
+------------------------------------------------------------------+

+------------------------------------------------------------------+
|                     ApplicationConfig.py                         |
|          (config.ini parsing, paths, feature flags)              |
+------------------------------------------------------------------+
```

### Data Flow

<img width="1352" height="686" alt="image" src="https://github.com/user-attachments/assets/0d9659b6-7bff-4ec5-8748-21c261f8d079" />

```
User Input (email address)
        |
        v
EmailSecurityAnalyzer.analyze_email()
        |
        +---> ThreadPoolExecutor (parallel)
        |         |
        |         +---> ThreatIntelligenceEngine.check_breaches()
        |         |         +---> SQLite cache check (7-day TTL)
        |         |         +---> LeakCheck.io API
        |         |         +---> XposedOrNot check-email API
        |         |         +---> XposedOrNot breach-analytics API
        |         |         +---> EmailRep.io API
        |         |         +---> Deduplicate + merge results
        |         |         +---> Cache to SQLite
        |         |
        |         +---> ThreatIntelligenceEngine.get_domain_reputation()
        |         |         +---> WHOIS lookup (5s timeout)
        |         |         +---> Domain age scoring
        |         |
        |         +---> ThreatIntelligenceEngine.check_dns_security()
        |         |         +---> SPF record check (2s timeout)
        |         |         +---> DMARC record check
        |         |         +---> DKIM record check
        |         |
        |         +---> ThreatIntelligenceEngine.check_password_breach()
        |                   +---> Keccak-512 k-anonymity API
        |
        +---> extract_features() --> 44 features
        |         +---> DisposableEmailDetector.is_disposable()
        |         +---> TyposquattingDetector.check_typosquatting()
        |         +---> Shannon entropy calculation
        |
        +---> MachineLearningEngine.predict_ensemble()
        |         +---> XGBoost prediction
        |         +---> Random Forest prediction
        |         +---> Isolation Forest anomaly score
        |         +---> Weighted ensemble + threshold check
        |
        +---> Risk Score Calculation
        |         +---> Base score from threats
        |         +---> ML adjustment (max 15%, baseline-subtracted)
        |         +---> Breach penalty
        |         +---> DNS/domain penalties
        |
        v
Analysis Result (dict) --> GUI Display + Reports
```

### Storage

```
%APPDATA%\EmailSecurityUltimate\
    models\
        ml_v2_models.pkl         # Trained ML models (XGBoost + RF + Isolation Forest)
        ml_feedback.db           # User feedback for retraining

<project_dir>\
    data\
        breach_intel.db          # Breach cache (7-day TTL), domain reputation cache
    logs\
        email_analyzer_*.log     # Timestamped application logs
    mitre_cache.json             # MITRE ATT&CK technique cache
    config.ini                   # Application configuration
```

---

## Installation

### Prerequisites
- Python 3.7 or higher (Python 3.13 recommended)
- Windows OS (CustomTkinter GUI)

### Required Packages
```bash
pip install customtkinter pandas numpy scikit-learn matplotlib requests
```

### Optional Packages (enhanced features)
```bash
# DNS validation
pip install dnspython

# Domain WHOIS lookups
pip install python-whois

# ML models (XGBoost)
pip install xgboost

# Excel report generation
pip install openpyxl seaborn

# MITRE ATT&CK semantic search
pip install sentence-transformers faiss-cpu stix2

# Breach password checking (Keccak-512)
pip install pysha3

# Web scraping (for threat intel enrichment)
pip install beautifulsoup4
```

### Install All
```bash
pip install customtkinter pandas numpy scikit-learn matplotlib requests dnspython python-whois xgboost openpyxl seaborn sentence-transformers faiss-cpu stix2 pysha3 beautifulsoup4
```

---

## Usage

<img width="1326" height="682" alt="image" src="https://github.com/user-attachments/assets/196fa9f1-2441-408c-bbbb-96aef23c7981" />

### Quick Start
```bash
python main.py
```

### Single Email Analysis
1. Navigate to **Email Analysis** page
2. Enter an email address
3. Click **Analyze** - results appear in ~2-5 seconds
4. Click **Safe** or **Risky** to provide ML feedback

### Bulk Analysis
1. Navigate to **Bulk Scanner** page
2. Load a text file with one email per line
3. Click **Start Bulk Analysis**
4. Export results as HTML, Excel, or PDF

### Configuration
Edit `config.ini` to customize:
```ini
[ml]
auto_train = True          # Enable ML model training
cross_validation_folds = 5 # CV folds for hyperparameter tuning

[general]
max_workers = 12           # Thread pool size for parallel operations
cache_ttl = 3600           # Cache time-to-live in seconds
```

---

## Module Reference

<img width="1346" height="727" alt="image" src="https://github.com/user-attachments/assets/a2dc4b06-ae5c-403c-990a-1f3fdb60c742" />

| Module | Lines | Description |
|--------|-------|-------------|
| `EmailSecurityAnalyzerGUI.py` | 6,498 | Main GUI application (13 pages, charts, exports) |
| `EnterpriseReportGenerator.py` | 2,698 | HTML/PDF enterprise report generation |
| `EnterpriseExcelReportGenerator.py` | 1,101 | Multi-sheet Excel workbook generation |
| `MachineLearningEngine.py` | 1,059 | XGBoost + RF ensemble, Isolation Forest, feedback loop |
| `ThreatIntelligenceEngine.py` | 904 | Breach APIs, WHOIS, DNS, domain reputation, SQLite cache |
| `EmailSecurityAnalyzer.py` | 788 | Core analysis orchestrator, feature extraction, risk scoring |
| `DisposableEmailDetector.py` | 666 | 720+ disposable email domain detection |
| `MitreAttackFramework.py` | 441 | MITRE ATT&CK technique mapping with FAISS |
| `MITRETAXIIConnection.py` | 337 | STIX2/TAXII 2.1 ATT&CK data retrieval |
| `TyposquattingDetector.py` | 327 | Levenshtein distance + homoglyph analysis |
| `TechniqueRetriever.py` | 224 | GitHub fallback for ATT&CK techniques |
| `ApplicationConfig.py` | 195 | Configuration parsing and path management |
| `BulkProcessingEngine.py` | 176 | Concurrent bulk email analysis |
| `AnimatedProgressBar.py` | 149 | Animated gradient progress bar widget |
| `CircularProgress.py` | 92 | Circular progress indicator widget |
| `ParticleEffect.py` | 79 | Background particle animation effect |
| `GradientButton.py` | 77 | Gradient hover-effect button widget |
| `RadarChart.py` | 71 | Radar/spider chart for security metrics |

---

## API Integrations

<img width="1321" height="702" alt="image" src="https://github.com/user-attachments/assets/63236a45-ad3f-4b2d-b819-335058e34c85" />

| API | Purpose | Rate Limit | Auth Required |
|-----|---------|------------|---------------|
| [LeakCheck.io](https://leakcheck.io) | Breach names, dates, data classes | 10/day (free) | No |
| [XposedOrNot](https://xposedornot.com) | Breach detection + analytics | 100/day | No |
| [EmailRep.io](https://emailrep.io) | Email reputation scoring | Varies | API key |
| MITRE TAXII 2.1 | ATT&CK technique database | Unlimited | No |

---

## Tech Stack

- **GUI:** CustomTkinter + Matplotlib
- **ML:** scikit-learn, XGBoost, CalibratedClassifierCV
- **NLP:** sentence-transformers, FAISS (MITRE semantic search)
- **Data:** pandas, numpy, SQLite3
- **Network:** requests, dnspython, python-whois
- **Reports:** openpyxl (Excel), HTML/CSS (web reports)
- **Security:** STIX2, TAXII 2.1, Keccak-512 (k-anonymity)
