import os
import sys
import platform
import logging
import warnings
from pathlib import Path
from dataclasses import dataclass
import configparser

warnings.filterwarnings('ignore')

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

# Python 3.13 compatibility check
if sys.version_info >= (3, 13):
    logger.warning("Python 3.13 detected - some features may have compatibility issues")
    PYTHON_313_COMPAT = True
else:
    PYTHON_313_COMPAT = False

# Check for sentence transformers availability (for semantic MITRE mapping)
try:
    import importlib.util as _ilutil
    SENTENCE_TRANSFORMERS_AVAILABLE = _ilutil.find_spec('sentence_transformers') is not None
except Exception:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

@dataclass
class ApplicationConfig:
    """Enhanced application configuration"""

    # System
    os_type: str = platform.system()
    is_windows: bool = platform.system() == "Windows"
    is_linux: bool = platform.system() == "Linux"
    is_mac: bool = platform.system() == "Darwin"

    # Paths
    base_dir: Path = None
    models_dir: Path = None
    data_dir: Path = None
    reports_dir: Path = None
    cache_dir: Path = None
    logs_dir: Path = None
    mitre_cache_dir: Path = None  # New: MITRE cache directory

    # Processing
    max_workers: int = 10
    batch_size: int = 100
    timeout: int = 30
    max_retries: int = 3
    cache_size: int = 100  # Cache size in MB

    # Analysis
    enable_ml: bool = True
    enable_dns: bool = True
    enable_whois: bool = True
    enable_breach_check: bool = True
    enable_password_breach_check: bool = True
    hibp_api_key: str = ""  # DEPRECATED - HIBP no longer used; breach checks use LeakCheck + XposedOrNot
    enable_threat_feeds: bool = True
    enable_animations: bool = True
    enable_semantic_mitre: bool = SENTENCE_TRANSFORMERS_AVAILABLE  # New: Semantic MITRE mapping
    ssl_verify: bool = True
    force_enable_ml: bool = False

    # ML Settings
    ml_threshold: float = 0.7
    anomaly_contamination: float = 0.1

    # UI Settings
    theme: str = "dark"
    theme_dark: bool = True  # GUI-friendly boolean (saved in settings.json)
    window_width: int = 1600
    window_height: int = 900
    enable_particles: bool = False
    enable_sounds: bool = False
    deep_scan_default: bool = True

    # Threat Intel / Updates
    auto_update_feeds: bool = True

    # ML UI Settings
    model_update_freq: str = "Weekly"

    # MITRE Settings
    mitre_collection_url: str = ""  # Disabled — use local mitre_cache.json (MITRE updates quarterly)
    mitre_github_fallback: bool = True
    mitre_cache_timeout: int = 86400  # 24 hours

    def __post_init__(self):
        """Initialize paths and apply overrides"""
        # Allow environment variable to force-enable ML on Python 3.13+
        try:
            if str(os.getenv('FORCE_ENABLE_ML', 'false')).lower() in ['1', 'true', 'yes']:
                self.force_enable_ml = True
        except Exception:
            pass

        if self.base_dir is None:
            if self.is_windows:
                self.base_dir = Path(os.environ.get('APPDATA', '.')) / "EmailSecurityUltimate"
            else:
                self.base_dir = Path.home() / ".email_security_ultimate"

        # Defaults based on base_dir
        self.models_dir = self.base_dir / "models"
        self.data_dir = self.base_dir / "data"
        self.reports_dir = self.base_dir / "reports"
        self.cache_dir = self.base_dir / "cache"
        self.logs_dir = self.base_dir / "logs"
        self.mitre_cache_dir = self.base_dir / "mitre_cache"

        # Attempt to load overrides from config.ini at project root
        try:
            project_root = Path.cwd()
            ini_path = project_root / "config.ini"
            if ini_path.exists():
                parser = configparser.ConfigParser()
                parser.read(ini_path)

                # [general]
                general = parser["general"] if parser.has_section("general") else {}
                self.max_workers = int(general.get("max_workers", self.max_workers))
                self.batch_size = int(general.get("batch_size", self.batch_size))

                # Optional directory overrides (absolute or relative)
                log_dir_value = general.get("log_dir") if general else None
                data_dir_value = general.get("data_dir") if general else None

                if log_dir_value:
                    self.logs_dir = Path(log_dir_value).expanduser()
                    if not self.logs_dir.is_absolute():
                        self.logs_dir = project_root / self.logs_dir

                if data_dir_value:
                    self.data_dir = Path(data_dir_value).expanduser()
                    if not self.data_dir.is_absolute():
                        self.data_dir = project_root / self.data_dir

                # [database]
                if parser.has_section("database"):
                    db = parser["database"]
                    self.timeout = int(db.get("timeout", self.timeout))

                # [ml]
                if parser.has_section("ml"):
                    ml = parser["ml"]
                    # Allow toggling ML (still forced off on 3.13 above)
                    self.enable_ml = self.enable_ml and ml.get("auto_train", str(self.enable_ml)).lower() not in ["false", "0", "no"]
                    if ml.get("force_enable", "false").lower() in ["1", "true", "yes"]:
                        self.force_enable_ml = True

                # [security]
                if parser.has_section("security"):
                    sec = parser["security"]
                    if 'ssl_verify' in sec:
                        self.ssl_verify = str(sec.get('ssl_verify')).lower() in ['1', 'true', 'yes']

                # [performance]
                if parser.has_section("performance"):
                    # Reserved for future performance tuning options
                    _ = parser["performance"]

        except Exception as e:
            logger.warning(f"Failed to read config.ini overrides: {e}")

        # Keep theme and theme_dark in sync (settings.json may override later)
        try:
            if isinstance(self.theme, str) and self.theme.lower() in ["dark", "light"]:
                self.theme_dark = self.theme.lower() == "dark"
            self.theme = "dark" if bool(self.theme_dark) else "light"
        except Exception:
            pass

        # ML v2 engine (RF + XGBoost) is fully Python 3.13 compatible — keep ML enabled
        if self.force_enable_ml:
            self.enable_ml = True

        # Ensure directories exist after overrides
        for directory in [self.models_dir, self.data_dir, self.reports_dir,
                          self.cache_dir, self.logs_dir, self.mitre_cache_dir]:
            try:
                directory.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.warning(f"Could not create directory {directory}: {e}")
