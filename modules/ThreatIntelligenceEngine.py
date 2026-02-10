import sqlite3
import hashlib
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

import requests

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

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('email_analyzer_ultimate.log'),
        logging.StreamHandler()
    ]
)
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
            try:
                cursor.execute("SELECT details_json FROM breaches LIMIT 1")
            except sqlite3.OperationalError:
                try:
                    cursor.execute("ALTER TABLE breaches ADD COLUMN details_json TEXT")
                    cursor.execute("ALTER TABLE breaches ADD COLUMN data_classes_json TEXT")
                    # Clear stale entries that lack detail data — force fresh API calls
                    cursor.execute("DELETE FROM breaches WHERE details_json IS NULL")
                    logger.info("Migrated breaches table: added details_json and data_classes_json columns, cleared stale entries")
                except Exception as migrate_err:
                    logger.debug(f"Migration skipped: {migrate_err}")

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
            self.update_online_feeds()

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
            breach_info['found'] = row[1] > 0  # Only set found=True if count > 0
            breach_info['count'] = row[1]
            breach_info['severity'] = row[3]
            breach_info['breaches'] = [b for b in row[2].split(',') if b] if row[2] else []
            # Restore details and data_classes from cache (columns 5 and 6)
            import json as _json
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
                        f'https://leakcheck.io/api/public?check={email}',
                        headers={'User-Agent': 'EmailSecurityAnalyzer/1.0'},
                        timeout=8
                    )
                    if response.status_code == 200:
                        data = response.json()
                        if data.get('success') and data.get('found'):
                            lc['found'] = True
                            lc['count'] = data.get('sources_count', 1)
                            for source in data.get('sources', [])[:10]:
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
                        f'https://api.xposedornot.com/v1/check-email/{email}',
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
                                for breach_name in flat_breaches[:15]:
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
                        exposures = data.get('ExposedBreaches', {})
                        breaches_details = exposures.get('breaches_details', [])
                        if isinstance(breaches_details, list) and breaches_details:
                            xa['found'] = True
                            for breach in breaches_details:
                                name = breach.get('breach', '') or breach.get('name', '')
                                if name:
                                    xa['breach_details'][name] = {
                                        'domain': breach.get('domain', ''),
                                        'breach_date': breach.get('xposed_date', 'Unknown'),
                                        'data_classes': breach.get('xposed_data', '').split(', ') if breach.get('xposed_data') else ['Email addresses'],
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
                        f'https://emailrep.io/{email}',
                        headers={
                            'User-Agent': 'EmailSecurityAnalyzer/1.0',
                            'Accept': 'application/json'
                        },
                        timeout=8
                    )
                    if response.status_code == 200:
                        data = response.json()
                        details = data.get('details', {})
                        er['credentials_leaked'] = bool(details.get('credentials_leaked', False))
                        er['data_breach'] = bool(details.get('data_breach', False))
                        er['reputation'] = data.get('reputation', 'none')
                        er['suspicious'] = bool(data.get('suspicious', False))
                        er['references'] = data.get('references', 0)
                        logger.info(f"EmailRep: reputation={er['reputation']}, breach={er['data_breach']}, leaked={er['credentials_leaked']}")
                    elif response.status_code == 429:
                        logger.info("EmailRep: Rate limited (10/day free)")
                except Exception as e:
                    logger.warning(f"EmailRep API error: {e}")
                return er

            try:
                # Run all 4 breach APIs in parallel
                with ThreadPoolExecutor(max_workers=4) as pool:
                    fut_lc = pool.submit(_leakcheck)
                    fut_xon = pool.submit(_xposedornot)
                    fut_xa = pool.submit(_xon_analytics)
                    fut_er = pool.submit(_emailrep)
                    lc_result = fut_lc.result(timeout=12)
                    xon_result = fut_xon.result(timeout=12)
                    xa_result = fut_xa.result(timeout=12)
                    er_result = fut_er.result(timeout=12)

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
                    for i, name in enumerate(xon_result['breaches']):
                        if name.lower() not in existing:
                            breach_info['breaches'].append(name)
                            breach_info['details'].append(xon_result['details'][i])
                            existing.add(name.lower())
                    breach_info['data_classes'].update(xon_result['classes'])

                # Enrich with XposedOrNot breach-analytics (dates, data classes, descriptions)
                if xa_result['found']:
                    breach_info['found'] = True
                    for detail in breach_info['details']:
                        name = detail.get('name', '')
                        xa_info = xa_result['breach_details'].get(name)
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
                    breach_info['emailrep'] = {
                        'credentials_leaked': er_result['credentials_leaked'],
                        'data_breach': er_result['data_breach'],
                        'reputation': er_result['reputation'],
                        'suspicious': er_result['suspicious'],
                        'references': er_result['references']
                    }

                # Always update count to match actual unique breach names from merged APIs
                breach_info['count'] = len(breach_info['breaches'])

            except Exception as e:
                logger.error(f"Unexpected error in breach check: {e}")

        # Determine severity based on breach count and data types
        if breach_info['count'] >= 5 or 'Passwords' in breach_info['data_classes']:
            breach_info['severity'] = 'critical'
        elif breach_info['count'] >= 3 or any(dc in breach_info['data_classes']
                                               for dc in ['Credit cards', 'Bank account numbers', 'Social security numbers']):
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
                with self._db_lock:
                    cursor = self.breach_db.cursor()
                    cursor.execute("""
                        INSERT OR REPLACE INTO breaches
                        (email, breach_count, breach_names, severity, last_checked, details_json, data_classes_json)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        email.lower(),
                        breach_info['count'],
                        ','.join(breach_info['breaches']),
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

        if 'Passwords' in breach_info['data_classes']:
            steps.append("2. CRITICAL: Your passwords were exposed! Change passwords on ALL accounts immediately")
            steps.append("3. Enable Two-Factor Authentication (2FA) on all important accounts")

        if any(dc in breach_info['data_classes'] for dc in ['Credit cards', 'Bank account numbers']):
            steps.append("4. URGENT: Contact your bank immediately to monitor for fraudulent activity")
            steps.append("5. Consider placing a fraud alert on your credit reports")

        if 'Social security numbers' in breach_info['data_classes']:
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
                steps.append(f"- Compromised data types: {', '.join(breach_info['data_classes'][:10])}")

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

        # Check cache
        with self._cache_lock:
            if domain in self.reputation_cache:
                cache_entry = self.reputation_cache[domain]
                if (datetime.now() - cache_entry['timestamp']).total_seconds() < 3600:
                    return cache_entry['data']

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
                            elif age_days > 3650:
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

        # Cache result
        with self._cache_lock:
            self.reputation_cache[domain] = {
                'data': reputation,
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
            'issues': [],
            'records': {},
            'score': 0
        }

        enabled = self.config.enable_dns if enabled is None else bool(enabled)
        if not DNS_AVAILABLE or not enabled:
            return dns_security

        # Check cache
        with self._cache_lock:
            if domain in self.dns_cache:
                cache_entry = self.dns_cache[domain]
                if (datetime.now() - cache_entry['timestamp']).total_seconds() < 3600:
                    return cache_entry['data']

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

            # DNSSEC
            try:
                resolver.resolve(domain, 'DNSKEY')
                dns_security['dnssec'] = True
                dns_security['score'] += 20
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

        # Cache result
        with self._cache_lock:
            self.dns_cache[domain] = {
                'data': dns_security,
                'timestamp': datetime.now()
            }

        return dns_security
