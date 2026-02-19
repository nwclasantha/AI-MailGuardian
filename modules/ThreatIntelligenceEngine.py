import copy
import re
import sqlite3
import hashlib
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

import requests
import urllib3
from urllib.parse import quote

# Keccak-512 support for XposedOrNot password breach checking
_keccak_method = None  # 'hashlib' or 'pycryptodome'
KECCAK_AVAILABLE = False
try:
    hashlib.new('keccak-512', b'test')
    _keccak_method = 'hashlib'
    KECCAK_AVAILABLE = True
except (ValueError, TypeError):
    try:
        from Crypto.Hash import keccak as _keccak_module
        _keccak_method = 'pycryptodome'
        KECCAK_AVAILABLE = True
    except ImportError:
        try:
            import sha3  # pysha3 package fallback
            _keccak_method = 'hashlib'
            KECCAK_AVAILABLE = True
        except ImportError:
            KECCAK_AVAILABLE = False

# DNS Resolution (optional)
try:
    import dns.resolver
    DNS_AVAILABLE = True
except ImportError:
    DNS_AVAILABLE = False
    print("Warning: dnspython not available. DNS checks will be limited.")

# WHOIS Lookup (optional)
try:
    import whois
    WHOIS_AVAILABLE = True
except ImportError:
    WHOIS_AVAILABLE = False

# Library modules should not configure the root logger — main.py owns that
logger = logging.getLogger(__name__)

# Import module dependencies
from .ApplicationConfig import ApplicationConfig

class ThreatIntelligenceEngine:
    """Complete threat intelligence and analysis"""

    def __init__(self, config: ApplicationConfig):
        self.config = config
        self.threat_feeds = {}
        self.breach_db = None
        self._db_lock = threading.Lock()
        self._cache_lock = threading.Lock()
        self.reputation_cache = {}
        self.dns_cache = {}
        self.initialize()

    def initialize(self):
        """Initialize threat intelligence components"""
        self.init_database()
        self.load_threat_feeds()

    def init_database(self):
        """Initialize breach database"""
        db_path = self.config.data_dir / "breach_intel.db"
        self.breach_db = sqlite3.connect(str(db_path), check_same_thread=False)

        with self._db_lock:
            cursor = self.breach_db.cursor()
            cursor.execute("""
                        CREATE TABLE IF NOT EXISTS breaches (
                            email TEXT PRIMARY KEY,
                            breach_count INTEGER,
                            breach_names TEXT,
                            severity TEXT,
                            last_checked TIMESTAMP,
                            details_json TEXT,
                            data_classes_json TEXT
                        )
                    """)

            cursor.execute("""
                        CREATE TABLE IF NOT EXISTS domains (
                            domain TEXT PRIMARY KEY,
                            reputation_score REAL,
                            category TEXT,
                            whois_data TEXT,
                            dns_records TEXT,
                            last_checked TIMESTAMP
                        )
                    """)

            cursor.execute("""
                        CREATE TABLE IF NOT EXISTS threat_indicators (
                            indicator TEXT PRIMARY KEY,
                            indicator_type TEXT,
                            threat_type TEXT,
                            confidence REAL,
                            source TEXT,
                            last_seen TIMESTAMP
                        )
                    """)

            # Migrate existing DB: add new columns if they don't exist yet
            # Check each migration column independently (handles partial migrations from crashes)
            for col in ('details_json', 'data_classes_json'):
                try:
                    cursor.execute(f"SELECT {col} FROM breaches LIMIT 1")
                except sqlite3.OperationalError:
                    try:
                        cursor.execute(f"ALTER TABLE breaches ADD COLUMN {col} TEXT")
                        logger.info(f"Migrated breaches table: added {col} column")
                    except Exception as migrate_err:
                        logger.debug(f"Migration of {col} skipped: {migrate_err}")

            # Clean up stale cache entries that have no details (from before v2 migration)
            try:
                deleted = cursor.execute(
                    "DELETE FROM breaches WHERE details_json IS NULL AND breach_count > 0"
                ).rowcount
                if deleted:
                    logger.info(f"Cleared {deleted} stale breach cache entries (missing details)")
            except Exception:
                pass

            self.breach_db.commit()

    def load_threat_feeds(self):
        """Load threat intelligence feeds"""
        self.threat_feeds = {
            'phishing_domains': set(),
            'malware_domains': set(),
            'spam_domains': set(),
            'typosquatting': set(),
            'compromised_emails': set(),
            'suspicious_ips': set(),
            'malicious_hashes': set(),
            'suspicious_keywords': set()
        }

        # Add known bad domains
        self.threat_feeds['phishing_domains'].update([
            'phishing-example.tk', 'suspicious-site.ml', 'fake-bank.ga',
            'secure-update.cf', 'account-verify.tk', 'payment-confirm.ml'
        ])

        self.threat_feeds['malware_domains'].update([
            'malware-host.tk', 'virus-download.ml', 'trojan-server.ga'
        ])

        self.threat_feeds['suspicious_keywords'].update([
            'urgent', 'verify', 'suspend', 'click here', 'act now',
            'limited time', 'winner', 'congratulations', 'free money'
        ])

        # Try to load online feeds (optional)
        if getattr(self.config, "enable_threat_feeds", True) and getattr(self.config, "auto_update_feeds", True):
            threading.Thread(target=self.update_online_feeds, daemon=True, name="ThreatFeedUpdater").start()

    def update_online_feeds(self):
        """Update from online threat feeds"""
        feed_urls = {
            'phishing': 'https://raw.githubusercontent.com/mitchellkrogza/Phishing.Database/master/phishing-domains-ACTIVE.txt',
            'malware': 'https://urlhaus.abuse.ch/downloads/text/',
            'spam': 'https://raw.githubusercontent.com/matomo-org/referrer-spam-blacklist/master/spammers.txt'
        }

        for feed_type, url in feed_urls.items():
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    domains = response.text.strip().split('\n')[:1000]  # Limit for performance
                    if feed_type == 'phishing':
                        self.threat_feeds['phishing_domains'].update(domains)
                    elif feed_type == 'malware':
                        self.threat_feeds['malware_domains'].update(domains)
                    elif feed_type == 'spam':
                        self.threat_feeds['spam_domains'].update(domains)
            except Exception:
                pass

    def check_email_breach(self, email: str, *, enabled: Optional[bool] = None) -> Dict:
        """Check if email is in breach databases with detailed breach information"""
        breach_info = {
            'found': False,
            'count': 0,
            'breaches': [],
            'severity': 'low',
            'details': [],
            'mitigation_steps': [],
            'data_classes': set(),
            'breach_dates': []
        }

        enabled = self.config.enable_breach_check if enabled is None else bool(enabled)
        if not enabled:
            breach_info['data_classes'] = []
            breach_info['mitigation_steps'] = ["Breach detection is disabled."]
            return breach_info

        # Check local database first for recent cache (within 7 days)
        with self._db_lock:
            cursor = self.breach_db.cursor()
            cursor.execute("SELECT * FROM breaches WHERE email = ?", (email.lower(),))
            row = cursor.fetchone()

        # Check if cached data is recent (within 7 days)
        use_cache = False
        if row and row[4]:  # row[4] is last_checked timestamp
            try:
                last_checked = datetime.fromisoformat(str(row[4]))
                # Ensure both datetimes are naive for safe comparison
                if last_checked.tzinfo is not None:
                    last_checked = last_checked.replace(tzinfo=None)
                now = datetime.now()
                if now - last_checked < timedelta(days=7):
                    use_cache = True
                    logger.info(f"Using cached breach data for {email} (age: {(now - last_checked).days} days)")
            except Exception:
                pass

        # Use cache if valid and recent
        if use_cache and row:
            breach_info['found'] = (row[1] or 0) > 0  # Only set found=True if count > 0
            breach_info['count'] = row[1] or 0
            breach_info['severity'] = row[3] or 'low'
            # Restore breach names from cache — try JSON first, fall back to comma-split for legacy data
            import json as _json
            try:
                breach_info['breaches'] = _json.loads(row[2]) if row[2] else []
            except (ValueError, TypeError):
                # Legacy format: comma-separated string
                breach_info['breaches'] = [b.strip() for b in row[2].split(',') if b.strip()] if row[2] else []
            # Restore details and data_classes from cache (columns 5 and 6)
            try:
                cached_details = row[5] if len(row) > 5 and row[5] else None
                breach_info['details'] = _json.loads(cached_details) if cached_details else []
            except Exception:
                breach_info['details'] = []
            try:
                cached_classes = row[6] if len(row) > 6 and row[6] else None
                breach_info['data_classes'] = _json.loads(cached_classes) if cached_classes else []
            except Exception:
                breach_info['data_classes'] = []
            # Rebuild breach_dates from cached details (not stored as own column)
            breach_info['breach_dates'] = [
                d.get('breach_date', '')
                for d in breach_info['details']
                if isinstance(d, dict) and d.get('breach_date') and d.get('breach_date') != 'Unknown'
            ]
            breach_info['mitigation_steps'] = self._generate_mitigation_steps(breach_info)
            return breach_info  # Return cached result

        # Check multiple FREE breach APIs in parallel
        if enabled:
            def _leakcheck():
                """LeakCheck.io FREE API"""
                lc = {'breaches': [], 'details': [], 'dates': [], 'classes': set(), 'count': 0, 'found': False}
                try:
                    logger.info(f"Checking LeakCheck.io FREE API for: {email}")
                    response = requests.get(
                        f'https://leakcheck.io/api/public?check={quote(email)}',
                        headers={'User-Agent': 'EmailSecurityAnalyzer/1.0'},
                        timeout=8
                    )
                    if response.status_code == 200:
                        data = response.json()
                        if data.get('success') and data.get('found'):
                            lc['found'] = True
                            lc['count'] = data.get('sources_count', 1)
                            for source in data.get('sources', []):
                                source_name = source.get('name', 'Unknown Breach')
                                source_date = source.get('date', 'Unknown')
                                description_parts = []
                                if source_name and source_name != 'Unknown Breach':
                                    description_parts.append(f"Breach: {source_name}")
                                if source.get('origin'):
                                    description_parts.append(f"Origin: {source.get('origin')}")
                                description = " | ".join(description_parts) if description_parts else f"Found in {source_name}"
                                breach_detail = {
                                    'name': source_name, 'title': source_name,
                                    'domain': source.get('origin', ''),
                                    'breach_date': source_date, 'description': description,
                                    'data_classes': ['Email addresses'] + (source.get('fields', []) if isinstance(source.get('fields'), list) else []),
                                    'pwn_count': source.get('entries', 0), 'is_verified': True
                                }
                                lc['breaches'].append(source_name)
                                lc['details'].append(breach_detail)
                                if source_date:
                                    lc['dates'].append(source_date)
                                if breach_detail.get('data_classes'):
                                    lc['classes'].update(breach_detail['data_classes'])
                            logger.info(f"LeakCheck found {lc['count']} breaches")
                except Exception as e:
                    logger.warning(f"LeakCheck API error: {e}")
                return lc

            def _xposedornot():
                """XposedOrNot FREE API"""
                xon = {'breaches': [], 'details': [], 'classes': set(), 'found': False}
                try:
                    logger.info(f"Checking XposedOrNot API for: {email}")
                    response = requests.get(
                        f'https://api.xposedornot.com/v1/check-email/{quote(email, safe="")}',
                        headers={'User-Agent': 'EmailSecurityAnalyzer/1.0'},
                        timeout=8
                    )
                    if response.status_code == 200:
                        data = response.json()
                        breaches_list = data.get('breaches', [])
                        if breaches_list and isinstance(breaches_list, list):
                            flat_breaches = []
                            for item in breaches_list:
                                if isinstance(item, list):
                                    flat_breaches.extend(item)
                                elif isinstance(item, str):
                                    flat_breaches.append(item)
                            if flat_breaches:
                                xon['found'] = True
                                for breach_name in flat_breaches:
                                    xon['breaches'].append(breach_name)
                                    xon['details'].append({
                                        'name': breach_name, 'title': breach_name,
                                        'domain': '', 'breach_date': 'Unknown',
                                        'description': f'Found in {breach_name} breach (via XposedOrNot)',
                                        'data_classes': ['Email addresses'],
                                        'pwn_count': 0, 'is_verified': True
                                    })
                                    xon['classes'].add('Email addresses')
                                logger.info(f"XposedOrNot found {len(flat_breaches)} breaches")
                    elif response.status_code == 404:
                        logger.info(f"XposedOrNot: No breaches found for {email}")
                except Exception as e:
                    logger.warning(f"XposedOrNot API error: {e}")
                return xon

            def _xon_analytics():
                """XposedOrNot breach-analytics (FREE, richer data - dates, data classes, descriptions)"""
                xa = {'breach_details': {}, 'found': False}
                try:
                    logger.info(f"Checking XposedOrNot breach-analytics for: {email}")
                    response = requests.get(
                        f'https://api.xposedornot.com/v1/breach-analytics?email={email}',
                        headers={'User-Agent': 'EmailSecurityAnalyzer/1.0'},
                        timeout=8
                    )
                    if response.status_code == 200:
                        data = response.json()
                        # breach-analytics returns detailed per-breach info
                        exposures = data.get('ExposedBreaches') or {}
                        breaches_details = exposures.get('breaches_details') or []
                        if isinstance(breaches_details, list) and breaches_details:
                            xa['found'] = True
                            for breach in breaches_details:
                                name = breach.get('breach', '') or breach.get('name', '')
                                if name:
                                    xa['breach_details'][name] = {
                                        'domain': breach.get('domain', ''),
                                        'breach_date': breach.get('xposed_date', 'Unknown'),
                                        'data_classes': [dc.strip() for dc in re.split(r'[,;]\s*', breach.get('xposed_data', '')) if dc.strip()] if breach.get('xposed_data') else ['Email addresses'],
                                        'description': breach.get('details', '') or f'Found in {name} breach',
                                        'industry': breach.get('industry', ''),
                                        'password_risk': breach.get('password_risk', ''),
                                        'searchable': breach.get('searchable', False),
                                        'xposed_records': breach.get('xposed_records', 0)
                                    }
                            logger.info(f"XposedOrNot analytics found {len(xa['breach_details'])} breach details")
                    elif response.status_code == 404:
                        logger.info(f"XposedOrNot analytics: No data for {email}")
                except Exception as e:
                    logger.warning(f"XposedOrNot analytics error: {e}")
                return xa

            def _emailrep():
                """EmailRep.io API (FREE, 10/day)"""
                er = {'credentials_leaked': False, 'data_breach': False, 'reputation': 'none', 'suspicious': False, 'references': 0}
                try:
                    logger.info(f"Checking EmailRep.io for: {email}")
                    response = requests.get(
                        f'https://emailrep.io/{quote(email, safe="")}',
                        headers={
                            'User-Agent': 'EmailSecurityAnalyzer/1.0',
                            'Accept': 'application/json'
                        },
                        timeout=8
                    )
                    if response.status_code == 200:
                        data = response.json()
                        details = data.get('details') or {}
                        er['credentials_leaked'] = bool(details.get('credentials_leaked', False))
                        er['data_breach'] = bool(details.get('data_breach', False))
                        er['reputation'] = data.get('reputation', 'none')
                        er['suspicious'] = bool(data.get('suspicious', False))
                        er['references'] = data.get('references', 0)
                        logger.info(f"EmailRep: reputation={er['reputation']}, breach={er['data_breach']}, leaked={er['credentials_leaked']}")
                    elif response.status_code == 401:
                        logger.warning("EmailRep: API key required (free unauthenticated API disabled)")
                    elif response.status_code == 429:
                        logger.info("EmailRep: Rate limited (10/day free)")
                    else:
                        logger.debug(f"EmailRep returned status {response.status_code}")
                except Exception as e:
                    logger.warning(f"EmailRep API error: {e}")
                return er

            try:
                # Run all 4 breach APIs in parallel
                # Each .result() is wrapped individually so one timeout doesn't drop all data
                def _safe_result(fut, default, api_name):
                    try:
                        return fut.result(timeout=12)
                    except Exception as e:
                        logger.warning(f"Breach API {api_name} failed: {e}")
                        return default

                with ThreadPoolExecutor(max_workers=4) as pool:
                    fut_lc = pool.submit(_leakcheck)
                    fut_xon = pool.submit(_xposedornot)
                    fut_xa = pool.submit(_xon_analytics)
                    fut_er = pool.submit(_emailrep)
                    lc_result = _safe_result(fut_lc, {'breaches': [], 'details': [], 'dates': [], 'classes': set(), 'count': 0, 'found': False}, 'LeakCheck')
                    xon_result = _safe_result(fut_xon, {'breaches': [], 'details': [], 'classes': set(), 'found': False}, 'XposedOrNot')
                    xa_result = _safe_result(fut_xa, {'breach_details': {}, 'found': False}, 'XON-Analytics')
                    er_result = _safe_result(fut_er, {'credentials_leaked': False, 'data_breach': False, 'reputation': 'none', 'suspicious': False, 'references': 0}, 'EmailRep')

                # Merge LeakCheck results first (most authoritative count), dedup by name
                if lc_result['found']:
                    breach_info['found'] = True
                    breach_info['count'] = lc_result['count']
                    seen = set()
                    for i, name in enumerate(lc_result['breaches']):
                        if name.lower() not in seen:
                            breach_info['breaches'].append(name)
                            if i < len(lc_result['details']):
                                breach_info['details'].append(lc_result['details'][i])
                            seen.add(name.lower())
                    breach_info['breach_dates'].extend(lc_result['dates'])
                    breach_info['data_classes'].update(lc_result['classes'])

                # Merge XposedOrNot breach names (de-duplicate)
                if xon_result['found']:
                    breach_info['found'] = True
                    existing = set(b.lower() for b in breach_info['breaches'])
                    for name, detail in zip(xon_result['breaches'], xon_result['details']):
                        if name.lower() not in existing:
                            breach_info['breaches'].append(name)
                            breach_info['details'].append(detail)
                            existing.add(name.lower())
                    breach_info['data_classes'].update(xon_result['classes'])

                # Enrich with XposedOrNot breach-analytics (dates, data classes, descriptions)
                if xa_result['found']:
                    breach_info['found'] = True
                    # Build case-insensitive lookup for analytics details
                    xa_details_ci = {k.lower(): v for k, v in xa_result['breach_details'].items()}
                    for detail in breach_info['details']:
                        name = detail.get('name', '')
                        xa_info = xa_details_ci.get(name.lower())
                        if xa_info:
                            # Enrich with richer data from analytics
                            if detail.get('breach_date', 'Unknown') == 'Unknown' and xa_info.get('breach_date', 'Unknown') != 'Unknown':
                                detail['breach_date'] = xa_info['breach_date']
                                breach_info['breach_dates'].append(xa_info['breach_date'])
                            xa_classes = xa_info.get('data_classes', [])
                            if xa_classes and xa_classes != ['Email addresses']:
                                existing_classes = set(detail.get('data_classes', []))
                                detail['data_classes'] = list(existing_classes.union(xa_classes))
                                breach_info['data_classes'].update(xa_classes)
                            if xa_info.get('description') and not detail.get('description'):
                                detail['description'] = xa_info['description']
                            if xa_info.get('domain') and not detail.get('domain'):
                                detail['domain'] = xa_info['domain']
                    # Add any breaches found in analytics but not in check-email
                    existing = set(b.lower() for b in breach_info['breaches'])
                    for name, info in xa_result['breach_details'].items():
                        if name.lower() not in existing:
                            breach_info['breaches'].append(name)
                            breach_info['details'].append({
                                'name': name, 'title': name,
                                'domain': info.get('domain', ''),
                                'breach_date': info.get('breach_date', 'Unknown'),
                                'description': info.get('description', f'Found in {name} breach'),
                                'data_classes': info.get('data_classes', ['Email addresses']),
                                'pwn_count': info.get('xposed_records', 0),
                                'is_verified': True
                            })
                            existing.add(name.lower())
                            if info.get('breach_date', 'Unknown') != 'Unknown':
                                breach_info['breach_dates'].append(info['breach_date'])
                            breach_info['data_classes'].update(info.get('data_classes', []))

                # Merge EmailRep (adds reputation context, not individual breach names)
                if er_result['credentials_leaked'] or er_result['data_breach']:
                    breach_info['found'] = True
                    # If no named breaches from other APIs, ensure minimum count for severity calc
                    if breach_info['count'] == 0 and len(breach_info['breaches']) == 0:
                        breach_info['count'] = 1
                    breach_info['emailrep'] = {
                        'credentials_leaked': er_result['credentials_leaked'],
                        'data_breach': er_result['data_breach'],
                        'reputation': er_result['reputation'],
                        'suspicious': er_result['suspicious'],
                        'references': er_result['references']
                    }

                # Set count to actual number of named breaches for accuracy
                # Only use API count as fallback when EmailRep detected breach but no names found
                actual_names = len(breach_info['breaches'])
                if breach_info['found'] and actual_names == 0:
                    breach_info['count'] = 1  # EmailRep-only: at least 1, no named breaches
                else:
                    breach_info['count'] = actual_names

            except Exception as e:
                logger.error(f"Unexpected error in breach check: {e}")

        # Determine severity based on breach count and data types
        dc_lower = {str(d).lower() for d in breach_info['data_classes']}
        if breach_info['count'] >= 5 or 'passwords' in dc_lower:
            breach_info['severity'] = 'critical'
        elif breach_info['count'] >= 3 or any(dc in dc_lower
                                               for dc in ['credit cards', 'bank account numbers', 'social security numbers']):
            breach_info['severity'] = 'high'
        elif breach_info['count'] >= 1:
            breach_info['severity'] = 'medium'

        # Convert set to list for JSON serialization
        breach_info['data_classes'] = list(breach_info['data_classes'])

        # Cache the result AFTER severity is calculated (so cached severity is correct)
        if enabled:
            try:
                import json as _json
                details_json = _json.dumps(breach_info.get('details', []), default=str)
                data_classes_json = _json.dumps(breach_info.get('data_classes', []), default=str)
                breach_names_json = _json.dumps([str(b) for b in breach_info['breaches']])
                with self._db_lock:
                    cursor = self.breach_db.cursor()
                    cursor.execute("""
                        INSERT OR REPLACE INTO breaches
                        (email, breach_count, breach_names, severity, last_checked, details_json, data_classes_json)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        email.lower(),
                        breach_info['count'],
                        breach_names_json,
                        breach_info['severity'],
                        datetime.now(),
                        details_json,
                        data_classes_json
                    ))
                    self.breach_db.commit()
            except Exception as cache_err:
                logger.warning(f"Could not cache result: {cache_err}")

        # Add mitigation steps based on findings
        breach_info['mitigation_steps'] = self._generate_mitigation_steps(breach_info)

        return breach_info

    def check_password_breach(self, password: str) -> Dict:
        """Check if a password has been exposed in known data breaches.

        Uses XposedOrNot password API with Keccak-512 hash + k-anonymity.
        Only the first 10 characters of the hash are sent to the API.
        The full password never leaves the local machine.
        """
        result = {
            'found': False,
            'details': '',
            'recommendation': '',
            'source': 'XposedOrNot',
            'hash_algorithm': 'Keccak-512',
            'privacy_note': 'Only first 10 chars of hash sent (k-anonymity)'
        }

        if not password or not password.strip():
            result['details'] = 'No password provided'
            return result

        if not KECCAK_AVAILABLE:
            result['details'] = 'Keccak-512 hashing not available. Install pysha3: pip install pysha3'
            result['recommendation'] = 'Cannot check password without Keccak-512 support.'
            logger.warning("Keccak-512 not available for password breach check")
            return result

        try:
            # Hash the password locally with Keccak-512
            if _keccak_method == 'pycryptodome':
                k = _keccak_module.new(digest_bits=512)
                k.update(password.encode('utf-8'))
                keccak_hash = k.hexdigest()
            else:
                keccak_hash = hashlib.new('keccak-512', password.encode('utf-8')).hexdigest()

            # Send only first 10 characters (k-anonymity)
            partial_hash = keccak_hash[:10]

            logger.info(f"Checking password breach via XposedOrNot (partial hash: {partial_hash}...)")

            response = requests.get(
                f'https://passwords.xposedornot.com/api/v1/pass/anon/{partial_hash}',
                headers={'User-Agent': 'EmailSecurityAnalyzer/1.0'},
                timeout=8
            )

            if response.status_code == 200:
                result['found'] = True
                result['details'] = 'This password has been found in known data breaches.'
                result['recommendation'] = (
                    'CRITICAL: This password is compromised. '
                    'Change it immediately on all accounts where it is used. '
                    'Use a unique, strong password (12+ characters with mixed case, numbers, and symbols). '
                    'Consider using a password manager.'
                )
                logger.info("Password found in breach database")

            elif response.status_code == 404:
                result['found'] = False
                result['details'] = 'This password was NOT found in known breach databases.'
                result['recommendation'] = (
                    'Good news! This password has not been detected in known breaches. '
                    'Continue following best practices: use unique passwords per account '
                    'and enable two-factor authentication.'
                )
                logger.info("Password not found in breach database (safe)")

            else:
                result['details'] = f'Password breach check returned status {response.status_code}'
                logger.warning(f"XposedOrNot password API returned unexpected status: {response.status_code}")

        except requests.exceptions.Timeout:
            result['details'] = 'Password breach check timed out. Try again later.'
            logger.warning("XposedOrNot password API timed out")
        except requests.exceptions.RequestException as e:
            result['details'] = f'Network error during password breach check: {str(e)}'
            logger.warning(f"XposedOrNot password API network error: {e}")
        except Exception as e:
            result['details'] = f'Error checking password: {str(e)}'
            logger.error(f"Unexpected error in password breach check: {e}")

        return result

    def _generate_mitigation_steps(self, breach_info: Dict) -> list:
        """Generate specific mitigation steps based on breach details"""
        steps = []

        if not breach_info['found']:
            return ["No breaches detected. Continue following security best practices."]

        # Critical steps
        steps.append("IMMEDIATE ACTIONS REQUIRED:")
        steps.append("1. Change passwords immediately for all accounts using this email")

        dc_lower = {str(d).lower() for d in breach_info['data_classes']}
        if 'passwords' in dc_lower:
            steps.append("2. CRITICAL: Your passwords were exposed! Change passwords on ALL accounts immediately")
            steps.append("3. Enable Two-Factor Authentication (2FA) on all important accounts")

        if any(dc in dc_lower for dc in ['credit cards', 'bank account numbers']):
            steps.append("4. URGENT: Contact your bank immediately to monitor for fraudulent activity")
            steps.append("5. Consider placing a fraud alert on your credit reports")

        if 'social security numbers' in dc_lower:
            steps.append("6. CRITICAL: Consider identity theft protection services")
            steps.append("7. Monitor credit reports for suspicious activity")

        # Standard recommendations
        steps.append("\nSTANDARD SECURITY MEASURES:")
        steps.append("- Use unique, strong passwords for each account (minimum 12 characters)")
        steps.append("- Use a password manager to generate and store complex passwords")
        steps.append("- Enable 2FA/MFA wherever available")
        steps.append("- Monitor accounts regularly for suspicious activity")
        steps.append("- Be cautious of phishing emails related to these breaches")

        if breach_info['count'] >= 3:
            steps.append("\nRECOMMENDED:")
            steps.append("- Consider creating a new email address for sensitive accounts")
            steps.append("- Review account permissions and connected apps")
            steps.append("- Enable breach notification services")

        # Specific breach information
        if breach_info['details']:
            steps.append(f"\nBREACH SUMMARY:")
            steps.append(f"- Total breaches: {breach_info['count']}")
            if breach_info['breach_dates']:
                valid_dates = [d for d in breach_info['breach_dates'] if d and d != 'Unknown']
                if valid_dates:
                    earliest = min(valid_dates)
                    latest = max(valid_dates)
                    steps.append(f"- Date range: {earliest} to {latest}")
            if breach_info['data_classes']:
                steps.append(f"- Compromised data types: {', '.join(str(d) for d in breach_info['data_classes'][:10])}")

        steps.append("\nMONITORING:")
        steps.append("- Check https://xposedornot.com regularly for new breaches")
        steps.append("- Sign up for breach notifications at XposedOrNot")
        steps.append("- Review this email in security audits periodically")

        return steps

    def check_domain_reputation(self, domain: str, *, enabled: Optional[bool] = None, enable_whois: Optional[bool] = None) -> Dict:
        """Check domain reputation"""
        reputation = {
            'score': 50,
            'category': 'unknown',
            'age': None,
            'flags': [],
            'whois': {},
            'registrar': None,
            'creation_date': None
        }

        enabled = getattr(self.config, "enable_threat_feeds", True) if enabled is None else bool(enabled)
        if not enabled:
            return reputation

        enable_whois = self.config.enable_whois if enable_whois is None else bool(enable_whois)

        # Check cache — return a copy so callers can't corrupt shared state
        with self._cache_lock:
            if domain in self.reputation_cache:
                cache_entry = self.reputation_cache[domain]
                if (datetime.now() - cache_entry['timestamp']).total_seconds() < 3600:
                    return copy.deepcopy(cache_entry['data'])

        # Check threat feeds
        if domain in self.threat_feeds['phishing_domains']:
            reputation['score'] -= 40
            reputation['flags'].append('phishing')
            reputation['category'] = 'malicious'

        if domain in self.threat_feeds['malware_domains']:
            reputation['score'] -= 40
            reputation['flags'].append('malware')
            reputation['category'] = 'malicious'

        if domain in self.threat_feeds['spam_domains']:
            reputation['score'] -= 20
            reputation['flags'].append('spam')

        # Check TLD reputation
        suspicious_tlds = ['.tk', '.ml', '.ga', '.cf', '.click', '.download', '.win', '.bid']
        if any(domain.endswith(tld) for tld in suspicious_tlds):
            reputation['score'] -= 20
            reputation['flags'].append('suspicious_tld')

        # WHOIS check if available (run with timeout to avoid hanging)
        if WHOIS_AVAILABLE and enable_whois:
            try:
                # Run WHOIS in a thread with 5s timeout to prevent hangs
                whois_result = [None]
                def _whois_lookup():
                    try:
                        whois_result[0] = whois.whois(domain)
                    except Exception:
                        pass
                t = threading.Thread(target=_whois_lookup, daemon=True)
                t.start()
                t.join(timeout=5)
                w = whois_result[0]
                if w is None:
                    logger.debug(f"WHOIS lookup for {domain} timed out after 5s")
                else:
                    if w.creation_date:
                        if isinstance(w.creation_date, list):
                            creation = w.creation_date[0]
                        else:
                            creation = w.creation_date

                        # Type-check: python-whois can return strings in edge cases
                        if not isinstance(creation, datetime):
                            try:
                                from dateutil import parser as date_parser
                                creation = date_parser.parse(str(creation))
                            except Exception:
                                creation = None

                        if creation is not None:
                            # Ensure both datetimes are naive for safe subtraction
                            creation_naive = creation.replace(tzinfo=None) if hasattr(creation, 'tzinfo') and creation.tzinfo else creation
                            age_days = (datetime.now() - creation_naive).days
                            reputation['age'] = age_days
                            reputation['creation_date'] = creation.isoformat()

                            if age_days < 30:
                                reputation['score'] -= 30
                                reputation['flags'].append('very_new_domain')
                            elif age_days < 180:
                                reputation['score'] -= 15
                                reputation['flags'].append('new_domain')
                            elif age_days > 3650 and reputation.get('category') != 'malicious':
                                reputation['score'] += 10
                                reputation['flags'].append('established_domain')

                    reputation['registrar'] = w.registrar
                    # Handle expiration_date and updated_date which can be list or single value
                    exp_date = w.expiration_date
                    if isinstance(exp_date, list):
                        exp_date = exp_date[0] if exp_date else None
                    upd_date = w.updated_date
                    if isinstance(upd_date, list):
                        upd_date = upd_date[0] if upd_date else None

                    reputation['whois'] = {
                        'registrar': w.registrar,
                        'creation_date': reputation.get('creation_date'),
                        'expiration_date': exp_date.isoformat() if exp_date and hasattr(exp_date, 'isoformat') else None,
                        'updated_date': upd_date.isoformat() if upd_date and hasattr(upd_date, 'isoformat') else None
                    }
            except Exception:
                pass

        # Normalize score
        reputation['score'] = max(0, min(100, reputation['score']))

        # Cache result — store a copy so returned dict can't corrupt cache
        with self._cache_lock:
            self.reputation_cache[domain] = {
                'data': copy.deepcopy(reputation),
                'timestamp': datetime.now()
            }

        return reputation

    def check_dns_security(self, domain: str, *, enabled: Optional[bool] = None) -> Dict:
        """Comprehensive DNS security checks"""
        dns_security = {
            'spf': False,
            'dmarc': False,
            'dkim': False,
            'mx': False,
            'dnssec': False,
            'bimi': False,
            'mta_sts': False,
            'tls_rpt': False,
            'issues': [],
            'records': {},
            'score': 0
        }

        enabled = self.config.enable_dns if enabled is None else bool(enabled)
        if not DNS_AVAILABLE or not enabled:
            return dns_security

        # Check cache — return a copy so callers can't corrupt shared state
        with self._cache_lock:
            if domain in self.dns_cache:
                cache_entry = self.dns_cache[domain]
                if (datetime.now() - cache_entry['timestamp']).total_seconds() < 3600:
                    return copy.deepcopy(cache_entry['data'])

        try:
            resolver = dns.resolver.Resolver()
            resolver.timeout = 2
            resolver.lifetime = 5

            # SPF Check
            try:
                txt_records = resolver.resolve(domain, 'TXT')
                spf_records = []
                for record in txt_records:
                    record_str = str(record).strip('"')
                    if 'v=spf1' in record_str:
                        dns_security['spf'] = True
                        spf_records.append(record_str)
                        dns_security['score'] += 20

                dns_security['records']['spf'] = spf_records
            except Exception:
                dns_security['issues'].append('No SPF record')

            # DMARC Check
            try:
                dmarc = resolver.resolve(f'_dmarc.{domain}', 'TXT')
                dmarc_records = []
                for record in dmarc:
                    record_str = str(record).strip('"')
                    if 'v=DMARC1' in record_str:
                        dns_security['dmarc'] = True
                        dmarc_records.append(record_str)
                        dns_security['score'] += 25

                        # Check DMARC policy
                        if 'p=reject' in record_str:
                            dns_security['score'] += 10
                        elif 'p=quarantine' in record_str:
                            dns_security['score'] += 5

                dns_security['records']['dmarc'] = dmarc_records
            except Exception:
                dns_security['issues'].append('No DMARC policy')

            # MX Records
            try:
                mx_records = resolver.resolve(domain, 'MX')
                if mx_records:
                    dns_security['mx'] = True
                    dns_security['score'] += 15
                    dns_security['records']['mx'] = [str(mx) for mx in mx_records]
            except Exception:
                dns_security['issues'].append('No MX records')

            # DKIM Check — probe common selectors and validate record content
            dkim_selectors = ['default', 'google', 'mail', 'dkim', 'k1', 'selector1', 'selector2', 's1', 's2']
            for selector in dkim_selectors:
                try:
                    dkim_records = resolver.resolve(f'{selector}._domainkey.{domain}', 'TXT')
                    for rdata in dkim_records:
                        txt = str(rdata).strip('"')
                        # Must contain DKIM version or a public key, and the key must not be empty (revoked)
                        if 'v=DKIM1' in txt or 'p=' in txt:
                            # Check for revoked key: p= with empty value means key is revoked
                            p_match = re.search(r'p=([^;\s]*)', txt)
                            if p_match and p_match.group(1):
                                # Non-empty public key — valid DKIM
                                dns_security['dkim'] = True
                                dns_security['score'] += 10
                                break
                            elif not p_match:
                                # No p= tag but has v=DKIM1 — still counts
                                dns_security['dkim'] = True
                                dns_security['score'] += 10
                                break
                            # else: p= with empty value — revoked key, skip
                    if dns_security['dkim']:
                        break
                except Exception:
                    pass

            # DNSSEC
            try:
                resolver.resolve(domain, 'DNSKEY')
                dns_security['dnssec'] = True
                dns_security['score'] += 20
            except Exception:
                pass

            # BIMI Check (Brand Indicators for Message Identification)
            if getattr(self.config, 'enable_bimi', True):
                try:
                    bimi_records = resolver.resolve(f'default._bimi.{domain}', 'TXT')
                    for record in bimi_records:
                        txt = str(record).strip('"')
                        if 'v=BIMI1' in txt:
                            dns_security['bimi'] = True
                            dns_security['score'] += 5
                            dns_security['records']['bimi'] = [txt]
                            break
                except Exception:
                    pass

            # MTA-STS Check (Mail Transfer Agent Strict Transport Security)
            if getattr(self.config, 'enable_mta_sts', True):
                try:
                    mta_sts_records = resolver.resolve(f'_mta-sts.{domain}', 'TXT')
                    for record in mta_sts_records:
                        txt = str(record).strip('"')
                        if 'v=STSv1' in txt:
                            dns_security['mta_sts'] = True
                            dns_security['score'] += 5
                            dns_security['records']['mta_sts'] = [txt]
                            break
                except Exception:
                    pass

            # TLS-RPT Check (TLS Reporting)
            if getattr(self.config, 'enable_tls_rpt', True):
                try:
                    tls_rpt_records = resolver.resolve(f'_smtp._tls.{domain}', 'TXT')
                    for record in tls_rpt_records:
                        txt = str(record).strip('"')
                        if 'v=TLSRPTv1' in txt:
                            dns_security['tls_rpt'] = True
                            dns_security['score'] += 5
                            dns_security['records']['tls_rpt'] = [txt]
                            break
                except Exception:
                    pass

            # A Records
            try:
                a_records = resolver.resolve(domain, 'A')
                dns_security['records']['a'] = [str(ip) for ip in a_records]
            except Exception:
                pass

        except Exception as e:
            dns_security['issues'].append(f'DNS resolution failed: {str(e)}')

        # Calculate final score
        dns_security['score'] = min(100, dns_security['score'])

        # Cache result — store a copy so returned dict can't corrupt cache
        with self._cache_lock:
            self.dns_cache[domain] = {
                'data': copy.deepcopy(dns_security),
                'timestamp': datetime.now()
            }

        return dns_security

    # ------------------------------------------------------------------ #
    #  New Advanced Security Checks                                       #
    # ------------------------------------------------------------------ #

    def check_dnsbl(self, domain: str, *, enabled: Optional[bool] = None) -> Dict:
        """Check domain mail-server IPs against DNS-based blacklists (DNSBL).

        Resolves MX records -> A records -> queries 10 DNSBL zones.
        """
        result = {
            'listed': False,
            'listings': [],
            'total_checked': 0,
            'listed_count': 0,
            'checked_ips': [],
        }

        enabled = getattr(self.config, 'enable_dnsbl', True) if enabled is None else bool(enabled)
        if not DNS_AVAILABLE or not enabled:
            return result

        # Check cache
        cache_key = f'dnsbl_{domain}'
        with self._cache_lock:
            if cache_key in self.reputation_cache:
                entry = self.reputation_cache[cache_key]
                if (datetime.now() - entry['timestamp']).total_seconds() < 3600:
                    return copy.deepcopy(entry['data'])

        dnsbl_zones = [
            'zen.spamhaus.org',
            'bl.spamcop.net',
            'b.barracudacentral.org',
            'dnsbl.sorbs.net',
            'spam.dnsbl.sorbs.net',
            'cbl.abuseat.org',
            'dnsbl-1.uceprotect.net',
            'psbl.surriel.com',
            'dyna.spamrats.com',
            'all.s5h.net',
        ]

        try:
            resolver = dns.resolver.Resolver()
            resolver.timeout = 2
            resolver.lifetime = 3

            # Step 1: Get MX server hostnames
            mx_hosts = []
            try:
                mx_records = resolver.resolve(domain, 'MX')
                for mx in mx_records:
                    mx_hosts.append(str(mx.exchange).rstrip('.'))
            except Exception:
                # Fallback: try domain itself
                mx_hosts = [domain]

            # Step 2: Resolve MX hostnames to IPs
            ips = set()
            for host in mx_hosts[:3]:  # Limit to 3 MX hosts
                try:
                    a_records = resolver.resolve(host, 'A')
                    for rr in a_records:
                        ips.add(str(rr))
                except Exception:
                    pass

            if not ips:
                # Try domain A record as last resort
                try:
                    a_records = resolver.resolve(domain, 'A')
                    for rr in a_records:
                        ips.add(str(rr))
                except Exception:
                    pass

            result['checked_ips'] = list(ips)

            # Step 3: Query each IP against each DNSBL (IPv4 only — DNSBL uses reversed octets)
            for ip in list(ips)[:3]:  # Limit IPs checked
                if ':' in ip:
                    continue  # Skip IPv6 — DNSBL reverse lookup only works for IPv4
                reversed_ip = '.'.join(reversed(ip.split('.')))
                for zone in dnsbl_zones:
                    result['total_checked'] += 1
                    try:
                        query = f'{reversed_ip}.{zone}'
                        resolver.resolve(query, 'A')
                        # A response means the IP IS listed
                        result['listed'] = True
                        result['listed_count'] += 1
                        result['listings'].append({
                            'zone': zone,
                            'ip': ip,
                        })
                    except (dns.resolver.NXDOMAIN, dns.resolver.NoAnswer):
                        # NXDOMAIN/NoAnswer = not listed (expected for clean IPs)
                        pass
                    except Exception:
                        # Timeout or other resolver error — result unknown
                        pass

        except Exception as e:
            logger.debug(f"DNSBL check failed for {domain}: {e}")

        # Cache result
        with self._cache_lock:
            self.reputation_cache[cache_key] = {
                'data': copy.deepcopy(result),
                'timestamp': datetime.now()
            }

        return result

    def check_cert_transparency(self, domain: str, *, enabled: Optional[bool] = None) -> Dict:
        """Query Certificate Transparency logs via crt.sh for domain certificates."""
        result = {
            'found': False,
            'cert_count': 0,
            'recent_certs': [],
            'issuers': [],
            'first_seen': None,
            'last_seen': None,
        }

        enabled = getattr(self.config, 'enable_cert_transparency', True) if enabled is None else bool(enabled)
        if not enabled:
            return result

        # Check cache
        cache_key = f'ct_{domain}'
        with self._cache_lock:
            if cache_key in self.reputation_cache:
                entry = self.reputation_cache[cache_key]
                if (datetime.now() - entry['timestamp']).total_seconds() < 3600:
                    return copy.deepcopy(entry['data'])

        try:
            response = requests.get(
                f'https://crt.sh/?q=%.{domain}&output=json',
                headers={'User-Agent': 'EmailSecurityAnalyzer/1.0'},
                timeout=10,
                verify=getattr(self.config, 'ssl_verify', True),
            )

            if response.status_code == 200 and 'json' in response.headers.get('Content-Type', ''):
                try:
                    certs = response.json()
                except ValueError:
                    logger.debug(f"crt.sh returned non-JSON response for {domain}")
                    certs = []
                if certs:
                    result['found'] = True
                    result['cert_count'] = len(certs)

                    # Collect unique issuers
                    issuers = set()
                    dates = []
                    for cert in certs:
                        issuer = cert.get('issuer_name', '')
                        if issuer:
                            for part in issuer.split(','):
                                part = part.strip()
                                if part.startswith('CN='):
                                    issuers.add(part[3:])
                                    break
                        entry_date = cert.get('entry_timestamp', '')
                        if entry_date:
                            dates.append(entry_date)

                    result['issuers'] = sorted(issuers)[:10]

                    if dates:
                        dates.sort()
                        result['first_seen'] = dates[0][:10] if dates[0] else None
                        result['last_seen'] = dates[-1][:10] if dates[-1] else None

                    # Recent certificates (last 5, deduplicated by key fields)
                    sorted_certs = sorted(certs, key=lambda c: c.get('entry_timestamp', ''), reverse=True)
                    seen_certs = set()
                    for cert in sorted_certs:
                        key = (cert.get('common_name', ''), cert.get('not_before', ''), cert.get('not_after', ''))
                        if key in seen_certs:
                            continue
                        seen_certs.add(key)
                        result['recent_certs'].append({
                            'common_name': cert.get('common_name', ''),
                            'issuer': cert.get('issuer_name', ''),
                            'not_before': (cert.get('not_before') or '')[:10],
                            'not_after': (cert.get('not_after') or '')[:10],
                        })
                        if len(result['recent_certs']) >= 5:
                            break

        except requests.exceptions.Timeout:
            logger.debug(f"crt.sh timeout for {domain}")
        except Exception as e:
            logger.debug(f"Certificate transparency check failed for {domain}: {e}")

        # Cache result
        with self._cache_lock:
            self.reputation_cache[cache_key] = {
                'data': copy.deepcopy(result),
                'timestamp': datetime.now()
            }

        return result

    def check_gravatar(self, email: str, *, enabled: Optional[bool] = None) -> Dict:
        """Check if an email has a Gravatar profile (indicates real user account)."""
        result = {
            'has_profile': False,
            'hash': '',
        }

        enabled = getattr(self.config, 'enable_gravatar_check', True) if enabled is None else bool(enabled)
        if not enabled:
            return result

        # Check cache
        cache_key = f'gravatar_{email.lower().strip()}'
        with self._cache_lock:
            if cache_key in self.reputation_cache:
                entry = self.reputation_cache[cache_key]
                if (datetime.now() - entry['timestamp']).total_seconds() < 3600:
                    return copy.deepcopy(entry['data'])

        try:
            email_hash = hashlib.md5(email.lower().strip().encode('utf-8')).hexdigest()
            result['hash'] = email_hash

            response = requests.head(
                f'https://gravatar.com/avatar/{email_hash}?d=404',
                headers={'User-Agent': 'EmailSecurityAnalyzer/1.0'},
                timeout=5,
                allow_redirects=True,
            )

            result['has_profile'] = response.status_code == 200

        except Exception as e:
            logger.debug(f"Gravatar check failed for {email}: {e}")

        # Cache result
        with self._cache_lock:
            self.reputation_cache[cache_key] = {
                'data': copy.deepcopy(result),
                'timestamp': datetime.now()
            }

        return result

    def check_threatfox(self, domain: str, *, enabled: Optional[bool] = None) -> Dict:
        """Check domain against abuse.ch ThreatFox IOC database."""
        result = {
            'found': False,
            'ioc_count': 0,
            'iocs': [],
        }

        enabled = getattr(self.config, 'enable_threatfox', True) if enabled is None else bool(enabled)
        if not enabled:
            return result

        # Check cache
        cache_key = f'threatfox_{domain}'
        with self._cache_lock:
            if cache_key in self.reputation_cache:
                entry = self.reputation_cache[cache_key]
                if (datetime.now() - entry['timestamp']).total_seconds() < 3600:
                    return copy.deepcopy(entry['data'])

        try:
            response = requests.post(
                'https://threatfox-api.abuse.ch/api/v1/',
                json={'query': 'search_ioc', 'search_term': domain},
                headers={'User-Agent': 'EmailSecurityAnalyzer/1.0'},
                timeout=10,
                verify=getattr(self.config, 'ssl_verify', True),
            )

            if response.status_code == 200:
                data = response.json()
                query_status = data.get('query_status', '')

                if query_status == 'ok' and data.get('data'):
                    iocs = data['data']
                    result['found'] = True
                    result['ioc_count'] = len(iocs)

                    for ioc in iocs[:10]:  # Limit to 10
                        result['iocs'].append({
                            'ioc_type': ioc.get('ioc_type', ''),
                            'threat_type': ioc.get('threat_type', ''),
                            'malware': ioc.get('malware_printable', ''),
                            'first_seen': (ioc.get('first_seen_utc') or '')[:10],
                            'confidence': ioc.get('confidence_level', 0),
                            'tags': ioc.get('tags', []),
                        })
                elif query_status not in ('ok', 'no_result'):
                    logger.warning(f"ThreatFox unexpected status '{query_status}' for {domain}")

        except requests.exceptions.Timeout:
            logger.debug(f"ThreatFox timeout for {domain}")
        except Exception as e:
            logger.debug(f"ThreatFox check failed for {domain}: {e}")

        # Cache result
        with self._cache_lock:
            self.reputation_cache[cache_key] = {
                'data': copy.deepcopy(result),
                'timestamp': datetime.now()
            }

        return result

    def check_parked_domain(self, domain: str, *, enabled: Optional[bool] = None) -> Dict:
        """Check if a domain appears to be parked (domain for sale / placeholder)."""
        result = {
            'is_parked': False,
            'indicators': [],
            'status_code': 0,
        }

        enabled = getattr(self.config, 'enable_parked_detection', True) if enabled is None else bool(enabled)
        if not enabled:
            return result

        # Check cache
        cache_key = f'parked_{domain}'
        with self._cache_lock:
            if cache_key in self.reputation_cache:
                entry = self.reputation_cache[cache_key]
                if (datetime.now() - entry['timestamp']).total_seconds() < 3600:
                    return copy.deepcopy(entry['data'])

        parking_keywords = [
            'buy this domain', 'domain for sale', 'this domain is for sale',
            'domain is parked', 'parked by', 'parked domain', 'parking page',
            'godaddy', 'sedo', 'dan.com', 'afternic', 'hugedomains',
            'domain may be for sale', 'inquire about this domain',
            'this webpage was generated', 'coming soon', 'under construction',
            'domain expired', 'this domain has expired',
        ]

        parking_services = [
            'sedoparking.com', 'parkingcrew.net', 'bodis.com',
            'above.com', 'domainsponsor.com', 'domainpark.com',
        ]

        try:
            ssl_verify = getattr(self.config, 'ssl_verify', True)
            if not ssl_verify:
                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            response = requests.get(
                f'http://{domain}',
                headers={'User-Agent': 'Mozilla/5.0 (compatible; EmailSecurityAnalyzer/1.0)'},
                timeout=5,
                allow_redirects=True,
                verify=ssl_verify,  # Many parked domains have invalid certs; respects config
            )

            result['status_code'] = response.status_code

            body = response.text.lower()[:5000]  # Only check first 5KB

            # Check for parking keywords (deduplicate)
            seen_indicators = set()
            for keyword in parking_keywords:
                if keyword in body and keyword not in seen_indicators:
                    result['indicators'].append(keyword)
                    seen_indicators.add(keyword)

            # Check if redirected to known parking service
            final_url = str(response.url).lower()
            for service in parking_services:
                indicator = f'redirect to {service}'
                if service in final_url and indicator not in seen_indicators:
                    result['indicators'].append(indicator)
                    seen_indicators.add(indicator)

            # Very small pages with generic content are suspicious
            if len(response.text) < 1000 and len(result['indicators']) >= 2:
                result['indicators'].append('minimal page content')

        except requests.exceptions.Timeout:
            logger.debug(f"Parked domain check timeout for {domain}")
        except requests.exceptions.ConnectionError:
            # Can't connect — might be a dead/parked domain
            result['indicators'].append('connection refused')
        except Exception as e:
            logger.debug(f"Parked domain check failed for {domain}: {e}")

        # Limit indicators to 5 most relevant
        result['indicators'] = result['indicators'][:5]
        # Evaluate is_parked after all code paths (including exception handlers)
        result['is_parked'] = len(result['indicators']) >= 2

        # Cache result
        with self._cache_lock:
            self.reputation_cache[cache_key] = {
                'data': copy.deepcopy(result),
                'timestamp': datetime.now()
            }

        return result
