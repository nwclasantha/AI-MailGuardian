import os
import json
import shutil
import logging
from datetime import datetime, timezone
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Library modules should not configure the root logger — main.py owns that
logger = logging.getLogger(__name__)

# Check for STIX2 support (optional)
try:
    from stix2 import MemoryStore
    STIX2_AVAILABLE = True
except ImportError:
    STIX2_AVAILABLE = False
    logger.warning("stix2 not available. MITRE data handling will be limited.")

# Import module dependencies
from .ApplicationConfig import ApplicationConfig

class MITRETAXIIConnection:
    """Handles connection to the MITRE ATT&CK TAXII Collection with local caching."""

    def __init__(self, config: ApplicationConfig):
        self.config = config
        self.collection_url = config.mitre_collection_url
        self.cache_file = config.mitre_cache_dir / "mitre_cache.json"
        self.last_updated_file = config.mitre_cache_dir / "last_updated.txt"
        self.timeout = config.timeout
        self.use_github_fallback = config.mitre_github_fallback
        self.memory_store = None
        # Ensure cache directory exists
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self.session = self.create_session_with_retries()
        self.load_data()

    def create_session_with_retries(self):
        """Create a session with retry logic and timeout"""
        session = requests.Session()

        # already imported at module level

        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        # Check for proxy settings
        http_proxy = os.getenv('HTTP_PROXY') or os.getenv('http_proxy')
        https_proxy = os.getenv('HTTPS_PROXY') or os.getenv('https_proxy')

        if http_proxy or https_proxy:
            proxies = {}
            if http_proxy:
                proxies['http'] = http_proxy
            if https_proxy:
                proxies['https'] = https_proxy
            session.proxies.update(proxies)
            logger.info(f"Using proxy settings: {proxies}")

        # Respect SSL verification from config if available
        try:
            if hasattr(self.config, 'ssl_verify'):
                session.verify = bool(self.config.ssl_verify)
        except Exception:
            pass

        return session

    def get_last_updated(self):
        """Retrieve the last updated timestamp from file."""
        if self.last_updated_file.exists():
            return self.last_updated_file.read_text(encoding="utf-8").strip()
        return "2000-01-01T00:00:00Z"

    def set_last_updated(self, timestamp):
        """Store the last updated timestamp."""
        self.last_updated_file.write_text(timestamp, encoding="utf-8")

    def _bootstrap_cache_from_repo(self) -> None:
        """If no cache exists yet, try to copy a packaged mitre_cache.json from the repo."""
        try:
            if self.cache_file.exists():
                return

            candidates = [
                Path.cwd() / "mitre_cache.json",
                Path(__file__).resolve().parent / "mitre_cache.json",
            ]

            for src in candidates:
                if not src.exists():
                    continue

                try:
                    self.cache_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, self.cache_file)
                    logger.info(f"Bootstrapped MITRE cache from: {src}")
                    try:
                        self.set_last_updated(datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
                    except Exception:
                        pass
                    return
                except Exception as e:
                    logger.warning(f"Could not bootstrap MITRE cache from {src}: {e}")

        except Exception as e:
            logger.debug(f"MITRE cache bootstrap skipped: {e}")

    def load_data(self):
        """Loads data from cache or fetches from the TAXII server."""
        self._bootstrap_cache_from_repo()
        if self.cache_file.exists():
            try:
                logger.info("Loading MITRE data from local cache.")
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    stix_data = json.load(f)

                if stix_data:
                    if STIX2_AVAILABLE:
                        self.memory_store = MemoryStore(stix_data=stix_data)
                    else:
                        self.memory_store = stix_data  # Fallback to raw data
                    logger.info(f"Successfully loaded {len(stix_data)} objects from cache.")

                    # Try to update cache with new data in background (truly non-blocking)
                    import threading
                    def _bg_update():
                        try:
                            self.update_cache_if_needed()
                        except Exception as e:
                            logger.warning(f"Could not update MITRE cache: {e}")
                    threading.Thread(target=_bg_update, daemon=True).start()
                    return
                else:
                    logger.warning("Cache file is empty, will fetch new data.")
            except (json.JSONDecodeError, UnicodeDecodeError, OSError) as e:
                logger.error(f"Cache file is invalid, will fetch new data: {e}")
                try:
                    self.cache_file.unlink()
                except Exception as unlink_err:
                    logger.warning(f"Could not remove invalid cache file: {unlink_err}")

        logger.info("Cache not found or invalid. Fetching data from MITRE ATT&CK.")
        self.fetch_and_cache_data()

    def update_cache_if_needed(self):
        """Try to update cache with new data if connected."""
        if not self.collection_url:
            return

        try:
            last_updated = self.get_last_updated()
            last_update_time = datetime.fromisoformat(last_updated.replace('Z', '+00:00'))
            if last_update_time.tzinfo is None:
                last_update_time = last_update_time.replace(tzinfo=timezone.utc)
            current_time = datetime.now(timezone.utc)

            if (current_time - last_update_time).total_seconds() >= 86400:
                logger.info("MITRE cache is more than 24 hours old, attempting to update...")
                self.fetch_incremental_updates()
        except Exception as e:
            logger.debug(f"Could not check for updates: {e}")

    def fetch_incremental_updates(self):
        """Fetch only new updates since last fetch."""
        if not self.collection_url:
            return

        try:
            last_updated = self.get_last_updated()
            url = f"{self.collection_url}/objects?added_after={last_updated}"

            headers = {"Accept": "application/vnd.oasis.stix+json; version=2.0"}

            # Use shorter timeout for background updates (don't block for 30s x 3 retries)
            update_timeout = min(self.timeout, 10)
            response = self.session.get(url, headers=headers, timeout=update_timeout)
            response.raise_for_status()

            new_objects = response.json().get("objects", [])

            # Handle STIX Bundle dict format (mirrors existing_data guard below)
            if isinstance(new_objects, dict) and 'objects' in new_objects:
                new_objects = new_objects['objects']

            if new_objects:
                try:
                    with open(self.cache_file, 'r', encoding='utf-8') as f:
                        existing_data = json.load(f)
                except (FileNotFoundError, json.JSONDecodeError):
                    existing_data = []  # Start fresh if cache is missing or corrupt

                # Handle both list and STIX Bundle dict formats
                if isinstance(existing_data, dict) and 'objects' in existing_data:
                    existing_data = existing_data['objects']
                existing_data.extend(new_objects)

                # Deduplicate by STIX ID, keeping the newest version
                seen = {}
                for obj in existing_data:
                    obj_id = obj.get('id', '')
                    if not obj_id:
                        continue
                    existing_mod = seen.get(obj_id, {}).get('modified', '')
                    if obj.get('modified', '') >= existing_mod:
                        seen[obj_id] = obj
                existing_data = list(seen.values())

                with open(self.cache_file, 'w', encoding='utf-8') as f:
                    json.dump(existing_data, f)

                if STIX2_AVAILABLE:
                    self.memory_store = MemoryStore(stix_data=existing_data)
                else:
                    self.memory_store = existing_data

                current_time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                self.set_last_updated(current_time)

                logger.info(f"Added {len(new_objects)} new MITRE objects to cache.")

        except Exception as e:
            logger.debug(f"Could not fetch incremental updates: {e}")

    def fetch_from_github(self):
        """Fallback method to fetch ATT&CK data from GitHub."""
        logger.info("Attempting to fetch MITRE data from GitHub (fallback)...")

        github_urls = [
            "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json",
            "https://raw.githubusercontent.com/mitre-attack/attack-stix-data/master/enterprise-attack/enterprise-attack.json"
        ]

        for url in github_urls:
            try:
                response = self.session.get(url, timeout=self.timeout)
                response.raise_for_status()

                attack_data = response.json()
                stix_objects = attack_data.get("objects", [])

                if stix_objects:
                    with open(self.cache_file, 'w', encoding='utf-8') as f:
                        json.dump(stix_objects, f)

                    if STIX2_AVAILABLE:
                        self.memory_store = MemoryStore(stix_data=stix_objects)
                    else:
                        self.memory_store = stix_objects

                    current_time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                    self.set_last_updated(current_time)

                    logger.info(f"Successfully fetched {len(stix_objects)} objects from GitHub.")
                    return True

            except Exception as e:
                logger.warning(f"Failed to fetch from {url}: {e}")
                continue

        return False

    def fetch_and_cache_data(self):
        """Fetches data with multiple fallback options."""
        # Try TAXII server first if URL is provided
        if self.collection_url:
            try:
                last_updated = self.get_last_updated()
                url = f"{self.collection_url}/objects?added_after={last_updated}"

                headers = {"Accept": "application/vnd.oasis.stix+json; version=2.0"}

                logger.info(f"Fetching from TAXII server: {url}")

                response = self.session.get(url, headers=headers, timeout=self.timeout)
                response.raise_for_status()

                stix_data = response.json().get("objects", [])

                if stix_data:
                    if STIX2_AVAILABLE:
                        self.memory_store = MemoryStore(stix_data=stix_data)
                    else:
                        self.memory_store = stix_data

                    with open(self.cache_file, 'w', encoding='utf-8') as f:
                        json.dump(stix_data, f)

                    current_time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                    self.set_last_updated(current_time)

                    logger.info(f"Successfully fetched {len(stix_data)} objects from TAXII server.")
                    return

            except requests.exceptions.Timeout:
                logger.error(f"Timeout connecting to TAXII server (waited {self.timeout}s)")
            except requests.exceptions.ConnectionError as e:
                logger.error(f"Connection error to TAXII server: {e}")
            except Exception as e:
                logger.error(f"Failed to fetch from TAXII server: {e}")

        # Try GitHub fallback
        if self.use_github_fallback:
            if self.fetch_from_github():
                return

        # If all else fails, use built-in data
        logger.warning("Could not fetch data from any source. Using built-in MITRE data.")
        self.memory_store = self.get_builtin_techniques()

    def get_builtin_techniques(self):
        """Return built-in MITRE techniques as fallback"""
        # Return the basic techniques from the original implementation
        return {
            'T1566': {'name': 'Phishing', 'type': 'attack-pattern',
                      'description': 'Adversaries send phishing messages to gain access'},
            'T1566.001': {'name': 'Spearphishing Attachment', 'type': 'attack-pattern',
                          'description': 'Phishing with malicious attachments'},
            'T1566.002': {'name': 'Spearphishing Link', 'type': 'attack-pattern',
                          'description': 'Phishing with malicious links'},
            'T1566.003': {'name': 'Spearphishing via Service', 'type': 'attack-pattern',
                          'description': 'Phishing through third-party services'},
            'T1598': {'name': 'Phishing for Information', 'type': 'attack-pattern',
                      'description': 'Phishing to collect information'},
            'T1190': {'name': 'Exploit Public-Facing Application', 'type': 'attack-pattern',
                      'description': 'Exploiting internet-facing applications'},
            'T1204': {'name': 'User Execution', 'type': 'attack-pattern',
                      'description': 'User executes malicious content'},
            'T1204.001': {'name': 'Malicious Link', 'type': 'attack-pattern',
                          'description': 'User clicks malicious link'},
            'T1204.002': {'name': 'Malicious File', 'type': 'attack-pattern',
                          'description': 'User opens malicious file'},
            'T1078': {'name': 'Valid Accounts', 'type': 'attack-pattern',
                      'description': 'Using legitimate credentials'},
            'T1110': {'name': 'Brute Force', 'type': 'attack-pattern', 'description': 'Password guessing attacks'},
            'T1486': {'name': 'Data Encrypted for Impact', 'type': 'attack-pattern',
                      'description': 'Ransomware encryption'},
        }
