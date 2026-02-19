#!/usr/bin/env python3
"""
================================================================================
ULTIMATE EMAIL SECURITY ANALYZER - PREMIUM ENHANCED GUI EDITION
================================================================================
Complete Professional Email Security Platform with Stunning Modern UI
Version: 6.0.2 ULTIMATE (Python 3.13 Compatible + Advanced MITRE ATT&CK)
Build: Full Feature Set + Ultra Beautiful GUI + Semantic MITRE Mapping
================================================================================

Features:
- Stunning Modern Dark UI with Animations
- Glass-morphism and Gradient Effects
- Real-time Interactive Visualizations
- ADVANCED MITRE ATT&CK Framework with Semantic Search
- 15+ Machine Learning Algorithms
- Deep Learning with PyTorch Support
- DNS Security Validation (SPF, DMARC, DKIM)
- Breach Detection with HIBP
- Domain Reputation Analysis
- Threat Intelligence Feeds
- Multi-threaded Bulk Processing
- Premium Animated Components
- Advanced Security Dashboards
- Semantic MITRE Technique Matching with Embeddings
"""

import os
import sys
import warnings
import logging
import traceback
import subprocess
from pathlib import Path
from typing import List, Tuple
from datetime import datetime
from enum import Enum

# Suppress warnings
warnings.filterwarnings('ignore')


class LogLevel(Enum):
    """Enumeration for log levels"""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


class ApplicationError(Exception):
    """Base exception for application errors"""
    pass


class DependencyError(ApplicationError):
    """Exception raised for dependency-related errors"""
    pass


class ConfigurationError(ApplicationError):
    """Exception raised for configuration errors"""
    pass


class InitializationError(ApplicationError):
    """Exception raised during application initialization"""
    pass


class EmailSecurityAnalyzerApp:
    """Main application class for Email Security Analyzer"""

    VERSION = "6.0.2"
    MIN_PYTHON_VERSION = (3, 7)
    PYTHON_313_VERSION = (3, 13)

    REQUIRED_PACKAGES = [
        # customtkinter is checked separately via _check_customtkinter()
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('sklearn', 'scikit-learn'),
        ('matplotlib', 'matplotlib'),
        ('requests', 'requests')
    ]

    OPTIONAL_PACKAGES = [
        ('dns', 'dnspython'),
        ('bs4', 'beautifulsoup4'),
        ('whois', 'python-whois'),
        ('torch', 'torch'),
        ('xgboost', 'xgboost'),
        ('lightgbm', 'lightgbm'),
        ('catboost', 'catboost'),
        ('openpyxl', 'openpyxl'),
        ('seaborn', 'seaborn'),
        ('sentence_transformers', 'sentence-transformers'),
        ('faiss', 'faiss-cpu'),
        ('stix2', 'stix2')
    ]

    def __init__(self, log_level: LogLevel = LogLevel.INFO):
        """
        Initialize the Email Security Analyzer Application

        Args:
            log_level: Logging level for the application
        """
        self.log_level = log_level
        self.logger = None
        self.log_dir = Path("logs")
        self.log_file = None
        self.python_313_compat = False
        self.missing_required = []
        self.missing_optional = []
        self.gui_app = None
        self.ctk_available = False

        # Initialize logging first
        self._setup_logging()
        self.logger.info(f"Initializing Email Security Analyzer v{self.VERSION}")

        # Setup environment
        self._setup_environment()

    def _setup_logging(self) -> None:
        """Configure application logging with file and console handlers"""
        try:
            # Create logs directory if it doesn't exist
            self.log_dir.mkdir(parents=True, exist_ok=True)

            # Generate log filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_file = self.log_dir / f"email_analyzer_{timestamp}.log"

            # Configure logging format
            log_format = logging.Formatter(
                '%(asctime)s - %(name)s - [%(levelname)s] - %(funcName)s:%(lineno)d - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )

            # Configure the ROOT logger so all module loggers inherit handlers
            root_logger = logging.getLogger()
            root_logger.setLevel(self.log_level.value)

            # Remove any existing handlers (e.g., from basicConfig)
            root_logger.handlers.clear()

            # File handler with detailed logging
            file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)  # Always log everything to file
            file_handler.setFormatter(log_format)

            # Console handler with configurable level
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.log_level.value)
            console_format = logging.Formatter(
                '%(asctime)s - [%(levelname)s] - %(message)s',
                datefmt='%H:%M:%S'
            )
            console_handler.setFormatter(console_format)

            # Add handlers to root logger — all module loggers propagate to this
            root_logger.addHandler(file_handler)
            root_logger.addHandler(console_handler)

            # Also keep a local reference for main.py usage
            self.logger = logging.getLogger(__name__)

            self.logger.debug(f"Logging initialized. Log file: {self.log_file}")

        except Exception as e:
            print(f"Failed to setup logging: {e}")
            sys.exit(1)

    def _setup_environment(self) -> None:
        """Setup environment variables and configurations"""
        try:
            self.logger.debug("Setting up environment variables")

            # Disable SSL warnings for development
            try:
                import urllib3
                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            except ImportError:
                pass  # urllib3 not available yet; will be installed with requests

            # Force sentence-transformers to use PyTorch only
            os.environ["USE_TF"] = "0"
            os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

            # Set other environment variables as needed
            os.environ["PYTHONIOENCODING"] = "utf-8"

            self.logger.debug("Environment setup completed")

        except Exception as e:
            self.logger.error(f"Failed to setup environment: {e}", exc_info=True)
            raise ConfigurationError(f"Environment setup failed: {e}")

    def _check_python_version(self) -> None:
        """Check Python version compatibility"""
        try:
            current_version = sys.version_info
            self.logger.info(f"Python version: {sys.version}")

            # Check minimum version
            if current_version < self.MIN_PYTHON_VERSION:
                error_msg = f"Python {self.MIN_PYTHON_VERSION[0]}.{self.MIN_PYTHON_VERSION[1]}+ required, found {current_version.major}.{current_version.minor}"
                self.logger.critical(error_msg)
                raise InitializationError(error_msg)

            # Check for Python 3.13 compatibility
            if current_version >= self.PYTHON_313_VERSION:
                self.python_313_compat = True
                self.logger.info("Python 3.13+ detected - ML Engine v2 fully supported")

            self.logger.debug("Python version check completed")

        except InitializationError:
            raise
        except Exception as e:
            self.logger.error(f"Error checking Python version: {e}", exc_info=True)
            raise InitializationError(f"Python version check failed: {e}")

    def _check_customtkinter(self) -> None:
        """Check if customtkinter is available"""
        try:
            import importlib
            if importlib.util.find_spec('customtkinter') is None:
                raise ImportError('customtkinter not found')
            self.ctk_available = True
            self.logger.debug("customtkinter is available")
        except ImportError:
            self.ctk_available = False
            error_msg = "customtkinter is required. Install with: pip install customtkinter"
            self.logger.critical(error_msg)
            print(f"Error: {error_msg}")
            raise DependencyError(error_msg)

    def _check_dependencies(self) -> Tuple[bool, bool]:
        """
        Check for required and optional dependencies

        Returns:
            Tuple of (all_required_met, has_optional)
        """
        try:
            self.logger.info("Checking application dependencies...")
            print("\nChecking dependencies...")

            # Check required packages
            for import_name, package_name in self.REQUIRED_PACKAGES:
                try:
                    __import__(import_name)
                    self.logger.debug(f"[OK] Required package found: {package_name}")
                    print(f"[OK] {package_name}")
                except ImportError:
                    self.missing_required.append(package_name)
                    self.logger.warning(f"[MISSING] Missing required package: {package_name}")
                    print(f"[MISSING] {package_name} (REQUIRED)")

            # Check optional packages
            for import_name, package_name in self.OPTIONAL_PACKAGES:
                try:
                    __import__(import_name)
                    self.logger.debug(f"[OK] Optional package found: {package_name}")
                    print(f"[OK] {package_name} (optional)")
                except ImportError:
                    self.missing_optional.append(package_name)
                    self.logger.debug(f"[SKIP] Optional package not found: {package_name}")
                    print(f"[SKIP] {package_name} (optional)")

            all_required_met = len(self.missing_required) == 0
            has_optional = len(self.missing_optional) < len(self.OPTIONAL_PACKAGES)

            if self.missing_required:
                self.logger.error(f"Missing required packages: {', '.join(self.missing_required)}")

            if self.missing_optional:
                self.logger.info(f"Missing optional packages: {', '.join(self.missing_optional)}")

            return all_required_met, has_optional

        except Exception as e:
            self.logger.error(f"Error checking dependencies: {e}", exc_info=True)
            raise DependencyError(f"Dependency check failed: {e}")

    def _install_packages(self, packages: List[str]) -> bool:
        """
        Install missing packages using pip

        Args:
            packages: List of package names to install

        Returns:
            True if installation successful, False otherwise
        """
        try:
            self.logger.info(f"Attempting to install packages: {', '.join(packages)}")

            for pkg in packages:
                print(f"Installing {pkg}...")
                self.logger.debug(f"Running: {sys.executable} -m pip install {pkg}")

                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", pkg],
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minute timeout
                )

                if result.returncode != 0:
                    self.logger.error(f"Failed to install {pkg}: {result.stderr}")
                    print(f"[FAIL] Failed to install {pkg}")
                    return False

                self.logger.info(f"Successfully installed {pkg}")
                print(f"[OK] Installed {pkg}")

            return True

        except subprocess.TimeoutExpired:
            self.logger.error("Package installation timed out")
            print("[FAIL] Installation timed out")
            return False
        except Exception as e:
            self.logger.error(f"Error installing packages: {e}", exc_info=True)
            print(f"[FAIL] Installation error: {e}")
            return False

    def _handle_missing_dependencies(self) -> None:
        """Handle missing required dependencies in non-interactive mode.

        Behavior:
        - If env AUTO_INSTALL_DEPS=true, attempt auto-install and exit(0) on success.
        - Otherwise, print instructions and exit(1) without prompting.
        """
        try:
            if not self.missing_required:
                return

            print(f"\nWARNING: Missing required packages: {', '.join(self.missing_required)}")
            print(f"Install with: pip install {' '.join(self.missing_required)}")

            auto_install = os.getenv('AUTO_INSTALL_DEPS', 'false').lower() in ['1', 'true', 'yes']
            if auto_install:
                self.logger.info("AUTO_INSTALL_DEPS enabled. Attempting automatic installation.")
                if self._install_packages(self.missing_required):
                    print("\n[OK] Packages installed! Please restart the application.")
                    self.logger.info("Package installation completed successfully")
                    raise DependencyError("Packages installed — please restart the application.")
                else:
                    print("\n[FAIL] Package installation failed. Please install manually.")
                    self.logger.error("Package installation failed")
                    raise DependencyError(f"Failed to install required packages: {', '.join(self.missing_required)}")
            else:
                self.logger.error("Required packages missing and auto-install disabled. Exiting.")
                raise DependencyError(f"Missing required packages: {', '.join(self.missing_required)}")

        except DependencyError:
            raise  # propagate intentional dependency errors unchanged
        except Exception as e:
            self.logger.error(f"Error handling missing dependencies: {e}", exc_info=True)
            raise DependencyError(f"Failed to handle missing dependencies: {e}")

    def _setup_gui_framework(self) -> None:
        """Setup GUI framework and theming"""
        try:
            self.logger.debug("Setting up GUI framework")

            # Import and configure customtkinter
            import customtkinter as ctk

            # Configure matplotlib backend
            import matplotlib
            matplotlib.use('TkAgg')

            # Configure GUI theme
            ctk.set_appearance_mode("dark")
            ctk.set_default_color_theme("dark-blue")

            self.logger.info("GUI framework configured successfully")

        except ImportError as e:
            self.logger.critical("customtkinter not available")
            raise DependencyError("customtkinter is required but not installed")
        except Exception as e:
            self.logger.error(f"Failed to setup GUI framework: {e}", exc_info=True)
            raise InitializationError(f"GUI framework setup failed: {e}")

    def _import_modules(self) -> None:
        """Import custom application modules in correct order"""
        try:
            self.logger.info("Importing application modules...")

            # Module import order with dependencies
            module_imports = [
                # 1. Base configuration (no dependencies)
                ("ApplicationConfig", "modules.ApplicationConfig"),

                # 2. UI components (no dependencies on other custom modules)
                ("AnimatedProgressBar", "modules.AnimatedProgressBar"),
                ("CircularProgress", "modules.CircularProgress"),
                ("GradientButton", "modules.GradientButton"),
                ("ParticleEffect", "modules.ParticleEffect"),
                ("RadarChart", "modules.RadarChart"),

                # 3. Detection modules (no dependencies on other custom modules)
                ("DisposableEmailDetector", "modules.DisposableEmailDetector"),
                ("TyposquattingDetector", "modules.TyposquattingDetector"),
                ("DGADetector", "modules.DGADetector"),

                # 4. Core components that only depend on ApplicationConfig
                ("MITRETAXIIConnection", "modules.MITRETAXIIConnection"),
                ("TechniqueRetriever", "modules.TechniqueRetriever"),
                ("MachineLearningEngine", "modules.MachineLearningEngine"),
                ("ThreatIntelligenceEngine", "modules.ThreatIntelligenceEngine"),

                # 4. MITRE Framework
                ("MitreAttackFramework", "modules.MitreAttackFramework"),

                # 5. Email Analyzer
                ("EmailSecurityAnalyzer", "modules.EmailSecurityAnalyzer"),

                # 6. Bulk Processor
                ("BulkProcessingEngine", "modules.BulkProcessingEngine"),

                # 7. Main GUI
                ("EmailSecurityAnalyzerGUI", "modules.EmailSecurityAnalyzerGUI")
            ]

            # Import each module
            for module_name, module_path in module_imports:
                try:
                    self.logger.debug(f"Importing {module_path}")
                    module = __import__(module_path, fromlist=[module_name])

                    # Store the GUI class for later use
                    if module_name == "EmailSecurityAnalyzerGUI":
                        self.gui_class = getattr(module, module_name)

                except ImportError as e:
                    self.logger.error(f"Failed to import {module_name}: {e}")
                    raise InitializationError(f"Module import failed: {module_name} - {e}")
                except AttributeError as e:
                    self.logger.error(f"Module {module_name} not found in {module_path}: {e}")
                    raise InitializationError(f"Module not found: {module_name}")

            self.logger.info("All modules imported successfully")

        except InitializationError:
            raise
        except Exception as e:
            self.logger.error(f"Error importing modules: {e}", exc_info=True)
            raise InitializationError(f"Module import failed: {e}")

    def _display_banner(self) -> None:
        """Display application banner"""
        banner = """
        ================================================================

           EMAIL SECURITY ANALYZER
           ULTIMATE - Premium Edition v{version}
        """

        if self.python_313_compat:
            banner += """           Python 3.13+ Compatible Edition
        """

        banner += """
        ================================================================
        """

        # Use safe encoding for Windows console
        try:
            print(banner.format(version=self.VERSION), flush=True)
        except UnicodeEncodeError:
            # Fallback to ASCII-only banner
            simple_banner = f"\n{'='*64}\nEMAIL SECURITY ANALYZER v{self.VERSION}\n{'='*64}\n"
            print(simple_banner, flush=True)

        self.logger.debug("Banner displayed")

    def _initialize_application(self) -> None:
        """Initialize the main application"""
        try:
            self.logger.info("Initializing main application...")
            print("\n" + "=" * 70)
            print("Launching GUI...")
            print("=" * 70 + "\n")

            # Create GUI instance
            if not hasattr(self, 'gui_class'):
                raise InitializationError("GUI class not imported")

            self.gui_app = self.gui_class()
            self.logger.info("GUI application created successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize application: {e}", exc_info=True)
            raise InitializationError(f"Application initialization failed: {e}")

    def _cleanup(self) -> None:
        """Cleanup resources before exit"""
        try:
            self.logger.info("Performing cleanup...")

            # Close GUI if exists
            if self.gui_app:
                try:
                    if hasattr(self.gui_app, 'destroy'):
                        self.gui_app.destroy()
                    elif hasattr(self.gui_app, 'quit'):
                        self.gui_app.quit()
                except Exception as e:
                    self.logger.debug(f"Error closing GUI: {e}")

            # Flush log handlers (root logger owns the handlers, not child loggers)
            for handler in logging.getLogger().handlers:
                handler.flush()

            self.logger.info("Cleanup completed")

        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}", exc_info=True)

    def run(self) -> int:
        """
        Main application entry point

        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        exit_code = 0

        try:
            # Display banner
            self._display_banner()

            # Check Python version
            self._check_python_version()

            # Check if customtkinter is available first
            self._check_customtkinter()

            # Setup GUI framework early
            self._setup_gui_framework()

            # Print startup message
            print("\n" + "=" * 70)
            print(f"Starting Email Security Analyzer Ultimate v{self.VERSION}...")
            if self.python_313_compat:
                print("Running on Python 3.13+ with ML Engine v2")
            print("=" * 70 + "\n")

            # Check dependencies
            all_required_met, _ = self._check_dependencies()

            # Handle missing required dependencies
            if not all_required_met:
                self._handle_missing_dependencies()

            # Display optional package info
            if self.missing_optional:
                print(f"\nINFO: Optional packages not installed: {', '.join(self.missing_optional)}")
                print("Some advanced features may be limited.")
                self.logger.info(f"Optional packages missing: {', '.join(self.missing_optional)}")

            print("\n[OK] All required dependencies satisfied!")

            # Import application modules
            self._import_modules()

            # Initialize and run application
            self._initialize_application()

            # Run the GUI
            self.logger.info("Starting GUI main loop")
            if hasattr(self.gui_app, 'run'):
                self.gui_app.run()
            elif hasattr(self.gui_app, 'mainloop'):
                self.gui_app.mainloop()
            else:
                raise InitializationError("GUI app has no run() or mainloop() method")

            self.logger.info("Application terminated normally")

        except KeyboardInterrupt:
            self.logger.info("Application terminated by user (Ctrl+C)")
            print("\n\nApplication terminated by user.")
            exit_code = 130  # Standard exit code for SIGINT

        except InitializationError as e:
            self.logger.critical(f"Initialization error: {e}")
            print(f"\n[ERROR] Initialization Error: {e}")
            exit_code = 1

        except DependencyError as e:
            self.logger.critical(f"Dependency error: {e}")
            print(f"\n[ERROR] Dependency Error: {e}")
            exit_code = 2

        except ConfigurationError as e:
            self.logger.critical(f"Configuration error: {e}")
            print(f"\n[ERROR] Configuration Error: {e}")
            exit_code = 3

        except ApplicationError as e:
            self.logger.critical(f"Application error: {e}")
            print(f"\n[ERROR] Application Error: {e}")
            exit_code = 4

        except Exception as e:
            self.logger.critical(f"Unexpected error: {e}", exc_info=True)
            print(f"\n[ERROR] Unexpected Error: {e}")
            print("\nFor details, check the log file:")
            if self.log_file:
                print(f"  {self.log_file}")
            traceback.print_exc()
            exit_code = 5

        finally:
            # Cleanup resources
            self._cleanup()

            # Log final status
            if exit_code == 0:
                self.logger.info(f"Application exited successfully (code: {exit_code})")
            else:
                self.logger.error(f"Application exited with error (code: {exit_code})")

            return exit_code


def main():
    """Main entry point for the application"""
    # Parse command line arguments for log level (optional enhancement)
    log_level = LogLevel.INFO

    if len(sys.argv) > 1:
        level_arg = sys.argv[1].upper()
        if level_arg in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
            log_level = LogLevel[level_arg]
            print(f"Log level set to: {level_arg}")

    # Create and run application
    app = EmailSecurityAnalyzerApp(log_level=log_level)
    exit_code = app.run()

    # Exit with appropriate code
    sys.exit(exit_code)


if __name__ == "__main__":
    main()