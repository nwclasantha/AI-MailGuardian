import html as html_mod
import os
import sys
import platform
import json
import time
import random
import threading
import logging
import warnings
from datetime import datetime
from collections import Counter
import tkinter as tk
from tkinter import filedialog, messagebox

# Suppress warnings
warnings.filterwarnings('ignore')

# CustomTkinter for modern UI
try:
    import customtkinter as ctk

    CTK_AVAILABLE = True
except ImportError:
    CTK_AVAILABLE = False
    print("Error: customtkinter is required. Install with: pip install customtkinter")
    sys.exit(1)

# Data Processing (for export functionality)
import pandas as pd
import numpy as np

# Check for Excel support (for export) without importing the module symbol
try:
    import importlib.util
    EXCEL_AVAILABLE = importlib.util.find_spec('openpyxl') is not None
except Exception:
    EXCEL_AVAILABLE = False

# Check for PyTorch availability (for status display only)
try:
    import torch

    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
    DEVICE = torch.device("cuda" if CUDA_AVAILABLE else "cpu")
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False
    DEVICE = "cpu"

# Matplotlib for charts
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Configure GUI theme
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

# Color scheme — Modern Dark Pro (GitHub/Discord-inspired)
COLORS = {
    'bg_primary': '#0f1117',
    'bg_secondary': '#161b22',
    'bg_tertiary': '#1c2128',
    'bg_quaternary': '#272d37',
    'accent_primary': '#4f8ff7',
    'accent_secondary': '#3fb950',
    'accent_tertiary': '#a78bfa',
    'danger': '#f85149',
    'warning': '#d29922',
    'success': '#3fb950',
    'info': '#58a6ff',
    'text_primary': '#e6edf3',
    'text_secondary': '#8b949e',
    'text_tertiary': '#6e7681',
    'gradient_start': '#4f8ff7',
    'gradient_mid': '#a78bfa',
    'gradient_end': '#3fb950',
    'glass': 'rgba(255,255,255,0.03)',
    'shadow': 'rgba(0,0,0,0.4)',
    'highlight': '#f0f6fc',
    'border': '#30363d',
    'sidebar_active': '#1f2937',
    'sidebar_hover': '#1a2332',
}

# Configure matplotlib dark style
plt.style.use('dark_background')
plt.rcParams.update({
    'figure.facecolor': '#0f1117',
    'axes.facecolor': '#161b22',
    'text.color': '#e6edf3',
    'axes.labelcolor': '#e6edf3',
    'xtick.color': '#8b949e',
    'ytick.color': '#8b949e',
    'grid.color': '#30363d',
    'grid.alpha': 0.4,
    'font.size': 10,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 10,
    'figure.titlesize': 15
})

# Library modules should not configure the root logger — main.py owns that
logger = logging.getLogger(__name__)

# Python 3.13 compatibility check
if sys.version_info >= (3, 13):
    logger.info("Python 3.13 detected - ML Engine v2 fully supported")
    PYTHON_313_COMPAT = True
else:
    PYTHON_313_COMPAT = False

# Import module dependencies
from .ApplicationConfig import ApplicationConfig
from .EmailSecurityAnalyzer import EmailSecurityAnalyzer
from .BulkProcessingEngine import BulkProcessingEngine
from .AnimatedProgressBar import AnimatedProgressBar
from .CircularProgress import CircularProgress
from .GradientButton import GradientButton
from .ParticleEffect import ParticleEffect
from .RadarChart import RadarChart


class EmailSecurityAnalyzerGUI:
    """Ultimate beautiful GUI application"""

    def __init__(self):
        self.root = ctk.CTk()
        self.root.title("AI-MailArmor Ultimate ✨")

        # Initialize configuration
        self.config = ApplicationConfig()

        # Load saved settings
        self.load_settings()

        # Apply appearance mode from config (startup)
        try:
            ctk.set_appearance_mode("dark" if getattr(self.config, "theme_dark", True) else "light")
        except Exception:
            pass

        self.root.geometry(f"{self.config.window_width}x{self.config.window_height}")

        # Set window properties
        self.root.configure(bg=COLORS['bg_primary'])

        # Center window
        self.center_window()

        # Initialize components
        self.analyzer = None
        self.bulk_processor = None
        self.current_results = []
        self.current_report = None
        self.is_processing = False
        self.animations_running = []
        self.monitor_running = False  # Add this for monitor control

        # Setup UI
        self.setup_ui()

        # Initialize analyzer in background
        self.initialize_analyzer()

        # Start animations
        if self.config.enable_animations:
            self.start_animations()

    def ui(self, func, *args, **kwargs):
        """Thread-safe UI updater: schedules func on main Tk loop."""
        try:
            if hasattr(self, 'root') and self.root.winfo_exists():
                self.root.after(0, lambda: func(*args, **kwargs))
        except Exception:
            pass

    def center_window(self):
        """Center window on screen"""
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (self.config.window_width // 2)
        y = (self.root.winfo_screenheight() // 2) - (self.config.window_height // 2)
        self.root.geometry(f"{self.config.window_width}x{self.config.window_height}+{x}+{y}")

    def setup_ui(self):
        """Setup the ultimate beautiful UI"""
        # Main container
        self.main_container = ctk.CTkFrame(self.root, fg_color=COLORS['bg_primary'])
        self.main_container.pack(fill="both", expand=True)

        # Add particle effect background
        if self.config.enable_particles:
            self.particle_bg = ParticleEffect(self.main_container)
            self.particle_bg.place(x=0, y=0, relwidth=1, relheight=1)

        # Create header
        self.create_header()

        # Create sidebar
        self.create_sidebar()

        # Create content area
        self.create_content_area()

        # Create status bar
        self.create_status_bar()

        # Show dashboard
        self.show_dashboard()

    def create_header(self):
        """Create refined header"""
        header_frame = ctk.CTkFrame(self.main_container, height=72, fg_color=COLORS['bg_secondary'],
                                     border_width=1, border_color=COLORS['border'])
        header_frame.pack(fill="x", padx=0, pady=0)
        header_frame.pack_propagate(False)

        # Content container
        content_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
        content_frame.pack(fill="both", expand=True, padx=24, pady=8)

        # Logo and title section
        logo_section = ctk.CTkFrame(content_frame, fg_color="transparent")
        logo_section.pack(side="left", fill="y")

        self.logo_label = ctk.CTkLabel(logo_section, text="🛡️", font=ctk.CTkFont(size=36))
        self.logo_label.pack(side="left", padx=(0, 16))

        title_frame = ctk.CTkFrame(logo_section, fg_color="transparent")
        title_frame.pack(side="left", fill="y", expand=True)

        self.title_label = ctk.CTkLabel(
            title_frame,
            text="AI-MailArmor",
            font=ctk.CTkFont(size=28, weight="bold"),
            text_color=COLORS['text_primary']
        )
        self.title_label.pack(anchor="w")

        self.subtitle_label = ctk.CTkLabel(
            title_frame,
            text="Ultimate Security Suite  •  Real-time Protection  •  AI-Powered",
            font=ctk.CTkFont(size=12),
            text_color=COLORS['text_secondary']
        )
        self.subtitle_label.pack(anchor="w")

        # Stats section
        stats_frame = ctk.CTkFrame(content_frame, fg_color="transparent")
        stats_frame.pack(side="right", fill="y")

        self.header_stats = []
        model_count = 0
        if self.config.models_dir.exists():
            model_count = len(list(self.config.models_dir.glob("*.pkl")))

        stats_data = [
            ("Emails Analyzed", "0", COLORS['accent_primary']),
            ("Threats Detected", "0", COLORS['danger']),
            ("Models Active", str(model_count), COLORS['success'])
        ]

        for label, value, color in stats_data:
            stat_container = ctk.CTkFrame(stats_frame, fg_color="transparent")
            stat_container.pack(side="left", padx=16)

            value_label = ctk.CTkLabel(
                stat_container,
                text=value,
                font=ctk.CTkFont(size=20, weight="bold"),
                text_color=color
            )
            value_label.pack()

            label_label = ctk.CTkLabel(
                stat_container,
                text=label,
                font=ctk.CTkFont(size=11),
                text_color=COLORS['text_secondary']
            )
            label_label.pack()

            self.header_stats.append({'value': value_label, 'label': label})

    def create_sidebar(self):
        """Create refined sidebar navigation"""
        sidebar_container = ctk.CTkFrame(self.main_container, fg_color=COLORS['bg_secondary'],
                                          border_width=1, border_color=COLORS['border'])
        sidebar_container.pack(side="left", fill="y", padx=0, pady=0)

        self.sidebar = ctk.CTkScrollableFrame(sidebar_container, width=220, fg_color=COLORS['bg_secondary'])
        self.sidebar.pack(fill="both", expand=True, padx=0, pady=0)

        # Navigation sections
        nav_sections = [
            {
                'title': 'MAIN',
                'items': [
                    ("Dashboard", "📊", self.show_dashboard),
                    ("Email Analysis", "🔍", self.show_analysis),
                    ("Bulk Scanner", "📁", self.show_bulk),
                    ("Real-time Monitor", "📡", self.show_monitor)
                ]
            },
            {
                'title': 'SECURITY',
                'items': [
                    ("MITRE ATT&CK", "🎯", self.show_mitre),
                    ("Threat Intelligence", "🌐", self.show_threat_intel),
                    ("ML Models", "🤖", self.show_ml),
                    ("DNS Security", "🔒", self.show_dns)
                ]
            },
            {
                'title': 'REPORTS',
                'items': [
                    ("Analytics", "📈", self.show_analytics),
                    ("Export Reports", "📄", self.show_reports),
                    ("Audit Log", "📝", self.show_audit)
                ]
            },
            {
                'title': 'SYSTEM',
                'items': [
                    ("Settings", "⚙️", self.show_settings),
                    ("About", "ℹ️", self.show_about)
                ]
            }
        ]

        self.nav_buttons = {}

        for section in nav_sections:
            section_label = ctk.CTkLabel(
                self.sidebar,
                text=section['title'],
                font=ctk.CTkFont(size=11, weight="bold"),
                text_color=COLORS['text_tertiary']
            )
            section_label.pack(pady=(16, 6), padx=16, anchor="w")

            for text, icon, command in section['items']:
                btn = ctk.CTkButton(
                    self.sidebar,
                    text=f" {icon}  {text}",
                    command=command,
                    height=38,
                    font=ctk.CTkFont(size=13),
                    anchor="w",
                    fg_color="transparent",
                    hover_color=COLORS['sidebar_hover'],
                    text_color=COLORS['text_secondary'],
                    corner_radius=8
                )
                btn.pack(fill="x", padx=8, pady=1)
                self.nav_buttons[text] = btn

    def create_content_area(self):
        """Create main content area"""
        content_container = ctk.CTkFrame(self.main_container, fg_color=COLORS['bg_primary'])
        content_container.pack(side="left", fill="both", expand=True, padx=0, pady=0)

        # Create all pages
        self.pages = {}
        self.create_all_pages(content_container)

    def create_status_bar(self):
        """Create refined status bar"""
        status_frame = ctk.CTkFrame(self.main_container, height=32, fg_color=COLORS['bg_secondary'],
                                     border_width=1, border_color=COLORS['border'])
        status_frame.pack(side="bottom", fill="x", padx=0, pady=0)
        status_frame.pack_propagate(False)

        status_content = ctk.CTkFrame(status_frame, fg_color="transparent")
        status_content.pack(fill="both", expand=True, padx=16)

        self.status_label = ctk.CTkLabel(
            status_content,
            text="Ready",
            font=ctk.CTkFont(size=11),
            text_color=COLORS['text_tertiary']
        )
        self.status_label.pack(side="left", pady=6)

        self.connection_indicator = ctk.CTkLabel(
            status_content,
            text="● Connected",
            font=ctk.CTkFont(size=11),
            text_color=COLORS['success']
        )
        self.connection_indicator.pack(side="right", pady=6, padx=(0, 12))

        self.status_progress = AnimatedProgressBar(status_content)
        self.status_progress.pack(side="right", fill="x", expand=True, padx=16, pady=6)
        self.status_progress.set_progress(0)

    def create_all_pages(self, parent):
        """Create all application pages"""

        # Dashboard
        self.pages['dashboard'] = self.create_dashboard_page(parent)

        # Email Analysis
        self.pages['analysis'] = self.create_analysis_page(parent)

        # Bulk Scanner
        self.pages['bulk'] = self.create_bulk_page(parent)

        # Real-time Monitor
        self.pages['monitor'] = self.create_monitor_page(parent)

        # MITRE ATT&CK
        self.pages['mitre'] = self.create_mitre_page(parent)

        # Threat Intelligence
        self.pages['threat_intel'] = self.create_threat_intel_page(parent)

        # ML Models
        self.pages['ml'] = self.create_ml_page(parent)

        # DNS Security
        self.pages['dns'] = self.create_dns_page(parent)

        # Analytics
        self.pages['analytics'] = self.create_analytics_page(parent)

        # Reports
        self.pages['reports'] = self.create_reports_page(parent)

        # Audit Log
        self.pages['audit'] = self.create_audit_page(parent)

        # Settings
        self.pages['settings'] = self.create_settings_page(parent)

        # About
        self.pages['about'] = self.create_about_page(parent)

    def create_dashboard_page(self, parent):
        """Create dashboard with live visualizations"""
        page = ctk.CTkScrollableFrame(parent, fg_color=COLORS['bg_primary'])

        # Dashboard header
        header_frame = ctk.CTkFrame(page, fg_color="transparent")
        header_frame.pack(fill="x", padx=24, pady=(16, 16))

        ctk.CTkLabel(
            header_frame,
            text="Security Dashboard",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color=COLORS['text_primary']
        ).pack(side="left")

        refresh_btn = ctk.CTkButton(
            header_frame,
            text="🔄  Refresh",
            command=self.refresh_dashboard,
            width=110,
            height=36,
            fg_color=COLORS['bg_tertiary'],
            hover_color=COLORS['bg_quaternary'],
            corner_radius=8
        )
        refresh_btn.pack(side="right")

        # Stats cards row
        stats_container = ctk.CTkFrame(page, fg_color="transparent")
        stats_container.pack(fill="x", padx=24, pady=(0, 16))

        self.dashboard_cards = []
        card_data = [
            ("Total Scans", "0", COLORS['accent_primary'], "📊", "total_scans"),
            ("Critical Risks", "0", COLORS['danger'], "🚨", "critical_risks"),
            ("Active Threats", "0", COLORS['warning'], "⚠️", "active_threats"),
            ("Safe Emails", "0", COLORS['success'], "✅", "safe_emails"),
            ("Avg Risk Score", "0", COLORS['info'], "📈", "avg_risk"),
            ("ML Accuracy", "0%", COLORS['accent_tertiary'], "🤖", "ml_accuracy")
        ]

        for i, (title, value, color, icon, key) in enumerate(card_data):
            card = self.create_animated_stat_card(stats_container, title, value, color, icon)
            card.grid(row=i // 3, column=i % 3, padx=6, pady=6, sticky="ew")
            card.key = key
            self.dashboard_cards.append(card)
            stats_container.columnconfigure(i % 3, weight=1)

        # Charts section
        charts_container = ctk.CTkFrame(page, fg_color="transparent")
        charts_container.pack(fill="both", expand=True, padx=24, pady=(0, 16))

        # Risk distribution chart
        risk_frame = ctk.CTkFrame(charts_container, fg_color=COLORS['bg_secondary'], corner_radius=12,
                                   border_width=1, border_color=COLORS['border'])
        risk_frame.grid(row=0, column=0, padx=6, pady=6, sticky="nsew")

        ctk.CTkLabel(
            risk_frame,
            text="Risk Distribution",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(16, 8))

        self.risk_chart_fig = Figure(figsize=(6, 4), facecolor='#161b22')
        self.risk_chart_ax = self.risk_chart_fig.add_subplot(111)
        self.risk_chart_ax.set_facecolor('#161b22')

        self.risk_chart_canvas = FigureCanvasTkAgg(self.risk_chart_fig, risk_frame)
        self.risk_chart_canvas.get_tk_widget().pack(fill="both", expand=True, padx=16, pady=(0, 16))

        # Threat timeline
        timeline_frame = ctk.CTkFrame(charts_container, fg_color=COLORS['bg_secondary'], corner_radius=12,
                                       border_width=1, border_color=COLORS['border'])
        timeline_frame.grid(row=0, column=1, padx=6, pady=6, sticky="nsew")

        ctk.CTkLabel(
            timeline_frame,
            text="Threat Timeline",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(16, 8))

        self.timeline_fig = Figure(figsize=(6, 4), facecolor='#161b22')
        self.timeline_ax = self.timeline_fig.add_subplot(111)
        self.timeline_ax.set_facecolor('#161b22')

        self.timeline_canvas = FigureCanvasTkAgg(self.timeline_fig, timeline_frame)
        self.timeline_canvas.get_tk_widget().pack(fill="both", expand=True, padx=16, pady=(0, 16))

        charts_container.columnconfigure(0, weight=1)
        charts_container.columnconfigure(1, weight=1)

        # Radar chart
        radar_frame = ctk.CTkFrame(charts_container, fg_color=COLORS['bg_secondary'], corner_radius=12,
                                    border_width=1, border_color=COLORS['border'])
        radar_frame.grid(row=1, column=0, padx=6, pady=6, sticky="nsew")

        ctk.CTkLabel(
            radar_frame,
            text="Threat Categories",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(16, 8))

        self.radar_chart = RadarChart(
            radar_frame,
            ['Phishing', 'Malware', 'Spam', 'Breach', 'DNS', 'Domain'],
            fg_color='#161b22'
        )
        self.radar_chart.pack(fill="both", expand=True, padx=16, pady=(0, 16))

        # Recent activity feed
        activity_frame = ctk.CTkFrame(charts_container, fg_color=COLORS['bg_secondary'], corner_radius=12,
                                       border_width=1, border_color=COLORS['border'])
        activity_frame.grid(row=1, column=1, padx=6, pady=6, sticky="nsew")

        ctk.CTkLabel(
            activity_frame,
            text="Recent Activity",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(16, 8))

        self.activity_feed = ctk.CTkTextbox(
            activity_frame,
            height=300,
            fg_color=COLORS['bg_tertiary'],
            text_color=COLORS['text_primary'],
            font=ctk.CTkFont(family="Consolas", size=11),
            corner_radius=8
        )
        self.activity_feed.pack(fill="both", expand=True, padx=16, pady=(0, 16))

        charts_container.rowconfigure(0, weight=1)
        charts_container.rowconfigure(1, weight=1)

        return page

    def create_animated_stat_card(self, parent, title, value, color, icon):
        """Create stat card"""
        card = ctk.CTkFrame(parent, fg_color=COLORS['bg_secondary'], corner_radius=12, height=120,
                             border_width=1, border_color=COLORS['border'])
        card.pack_propagate(False)

        content = ctk.CTkFrame(card, fg_color="transparent")
        content.pack(expand=True)

        icon_label = ctk.CTkLabel(
            content,
            text=icon,
            font=ctk.CTkFont(size=28),
            text_color=color
        )
        icon_label.pack(pady=(8, 4))

        value_label = ctk.CTkLabel(
            content,
            text=value,
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color=COLORS['text_primary']
        )
        value_label.pack()

        title_label = ctk.CTkLabel(
            content,
            text=title,
            font=ctk.CTkFont(size=12),
            text_color=COLORS['text_secondary']
        )
        title_label.pack(pady=(2, 8))

        card.value_label = value_label
        card.title = title
        card.color = color

        return card

    def create_analysis_page(self, parent):
        """Create email analysis page"""
        page = ctk.CTkFrame(parent, fg_color=COLORS['bg_primary'])

        # Header
        header_frame = ctk.CTkFrame(page, fg_color="transparent")
        header_frame.pack(fill="x", padx=24, pady=(16, 12))

        ctk.CTkLabel(
            header_frame,
            text="Email Analysis",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color=COLORS['text_primary']
        ).pack(anchor="w")

        ml_status = "AI-powered" if self.config.enable_ml else "Rule-based (ML disabled)"
        ctk.CTkLabel(
            header_frame,
            text=f"{ml_status} analysis with real-time threat detection",
            font=ctk.CTkFont(size=12),
            text_color=COLORS['text_secondary']
        ).pack(anchor="w")

        # Main content with two columns
        content_frame = ctk.CTkFrame(page, fg_color="transparent")
        content_frame.pack(fill="both", expand=True, padx=24, pady=(0, 16))

        # Left column - Input
        left_column = ctk.CTkFrame(content_frame, fg_color=COLORS['bg_secondary'], corner_radius=12,
                                    border_width=1, border_color=COLORS['border'])
        left_column.pack(side="left", fill="both", expand=True, padx=(0, 8))

        input_frame = ctk.CTkFrame(left_column, fg_color="transparent")
        input_frame.pack(fill="x", padx=24, pady=24)

        ctk.CTkLabel(
            input_frame,
            text="Enter Email Address",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(0, 12))

        self.email_entry = ctk.CTkEntry(
            input_frame,
            placeholder_text="example@domain.com",
            height=44,
            font=ctk.CTkFont(size=14),
            fg_color=COLORS['bg_tertiary'],
            border_color=COLORS['border'],
            border_width=1,
            corner_radius=8
        )
        self.email_entry.pack(fill="x", pady=8)

        # Password input
        password_label_frame = ctk.CTkFrame(input_frame, fg_color="transparent")
        password_label_frame.pack(fill="x", pady=(12, 0))

        ctk.CTkLabel(
            password_label_frame,
            text="Check Password (Optional)",
            font=ctk.CTkFont(size=13, weight="bold")
        ).pack(side="left")

        ctk.CTkLabel(
            password_label_frame,
            text="Hashed locally, partial hash sent",
            font=ctk.CTkFont(size=10),
            text_color=COLORS['text_tertiary']
        ).pack(side="right")

        self.password_entry = ctk.CTkEntry(
            input_frame,
            placeholder_text="Enter password to check against breach databases",
            height=40,
            font=ctk.CTkFont(size=13),
            fg_color=COLORS['bg_tertiary'],
            border_color=COLORS['border'],
            border_width=1,
            corner_radius=8,
            show="*"
        )
        self.password_entry.pack(fill="x", pady=(4, 0))

        self.show_password_var = ctk.BooleanVar(value=False)

        def toggle_password_visibility():
            if self.show_password_var.get():
                self.password_entry.configure(show="")
            else:
                self.password_entry.configure(show="*")

        ctk.CTkCheckBox(
            input_frame,
            text="Show password",
            variable=self.show_password_var,
            command=toggle_password_visibility,
            font=ctk.CTkFont(size=11),
            height=20,
            checkbox_width=16,
            checkbox_height=16
        ).pack(anchor="w", pady=(3, 0))

        # Quick fill buttons
        quick_fill_frame = ctk.CTkFrame(input_frame, fg_color="transparent")
        quick_fill_frame.pack(fill="x", pady=8)

        ctk.CTkLabel(
            quick_fill_frame,
            text="Quick Test:",
            font=ctk.CTkFont(size=11),
            text_color=COLORS['text_tertiary']
        ).pack(side="left", padx=(0, 8))

        test_emails = [
            ("Safe", "john.doe@gmail.com"),
            ("Suspicious", "urgent.verify@suspicious-site.tk"),
            ("Phishing", "security-alert@phishing-example.ml")
        ]

        for label, email in test_emails:
            def fill_email(e=email):
                self.email_entry.delete(0, 'end')
                self.email_entry.insert(0, e)

            btn = ctk.CTkButton(
                quick_fill_frame,
                text=label,
                command=fill_email,
                width=80,
                height=28,
                fg_color=COLORS['bg_tertiary'],
                hover_color=COLORS['bg_quaternary'],
                corner_radius=6,
                font=ctk.CTkFont(size=11)
            )
            btn.pack(side="left", padx=2)

        # Analyze + Stop buttons
        btn_frame = ctk.CTkFrame(input_frame, fg_color="transparent")
        btn_frame.pack(fill="x", pady=16)

        self.analyze_btn = GradientButton(
            btn_frame,
            text="🔍 Analyze Security",
            command=self.analyze_email,
            height=44,
            font=ctk.CTkFont(size=15, weight="bold"),
            gradient_colors=[COLORS['accent_primary'], COLORS['accent_secondary']],
            corner_radius=8
        )
        self.analyze_btn.pack(side="left", fill="x", expand=True, padx=(0, 4))

        self.stop_analyze_btn = GradientButton(
            btn_frame,
            text="⏹ Stop",
            command=self.stop_single_analysis,
            height=44,
            font=ctk.CTkFont(size=15, weight="bold"),
            gradient_colors=[COLORS['danger'], '#ff7b72'],
            corner_radius=8,
            state="disabled"
        )
        self.stop_analyze_btn.pack(side="right", fill="x", expand=True, padx=(4, 0))

        # Cancel flag for single analysis
        self._single_cancel = threading.Event()

        # Progress bar for single analysis
        self.analysis_progress_bar = AnimatedProgressBar(input_frame)
        self.analysis_progress_bar.pack(fill="x", pady=(0, 10))
        self.analysis_progress_bar.pack_forget()

        # Progress indicator (circular)
        self.analysis_progress = CircularProgress(input_frame, size=100, fg_color="transparent")
        self.analysis_progress.pack(pady=20)
        self.analysis_progress.pack_forget()

        # Options
        options_frame = ctk.CTkFrame(left_column, fg_color="transparent")
        options_frame.pack(fill="x", padx=24, pady=(0, 24))

        ctk.CTkLabel(
            options_frame,
            text="Analysis Options",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=(0, 8))

        self.deep_scan_var = ctk.BooleanVar(value=getattr(self.config, "deep_scan_default", True))
        ctk.CTkCheckBox(
            options_frame,
            text="Deep Scan (DNS, WHOIS, Breach Check)",
            variable=self.deep_scan_var,
            font=ctk.CTkFont(size=13)
        ).pack(anchor="w", pady=4)

        self.ml_analysis_var = ctk.BooleanVar(value=self.config.enable_ml)
        ml_checkbox = ctk.CTkCheckBox(
            options_frame,
            text="Machine Learning Analysis",
            variable=self.ml_analysis_var,
            font=ctk.CTkFont(size=13)
        )
        ml_checkbox.pack(anchor="w", pady=4)

        if not self.config.enable_ml:
            ml_checkbox.configure(state="disabled")
            ctk.CTkLabel(
                options_frame,
                text="(ML disabled in settings)",
                font=ctk.CTkFont(size=11),
                text_color=COLORS['warning']
            ).pack(anchor="w")

        self.threat_intel_var = ctk.BooleanVar(value=bool(getattr(self.config, "enable_threat_feeds", True)))
        ctk.CTkCheckBox(
            options_frame,
            text="Threat Intelligence Lookup",
            variable=self.threat_intel_var,
            font=ctk.CTkFont(size=13)
        ).pack(anchor="w", pady=4)

        # Right column - Results
        self.results_column = ctk.CTkScrollableFrame(
            content_frame,
            fg_color=COLORS['bg_secondary'],
            corner_radius=12,
            border_width=1,
            border_color=COLORS['border']
        )
        self.results_column.pack(side="right", fill="both", expand=True, padx=(8, 0))

        initial_label = ctk.CTkLabel(
            self.results_column,
            text="Analysis results will appear here",
            font=ctk.CTkFont(size=14),
            text_color=COLORS['text_tertiary']
        )
        initial_label.pack(expand=True, pady=80)

        return page

    def create_bulk_page(self, parent):
        """Create bulk analysis page"""
        page = ctk.CTkFrame(parent, fg_color=COLORS['bg_primary'])

        # Header
        header_frame = ctk.CTkFrame(page, fg_color="transparent")
        header_frame.pack(fill="x", padx=24, pady=(16, 12))

        ctk.CTkLabel(
            header_frame,
            text="Bulk Email Scanner",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color=COLORS['text_primary']
        ).pack(anchor="w")

        # Main content
        content_frame = ctk.CTkFrame(page, fg_color="transparent")
        content_frame.pack(fill="both", expand=True, padx=24)

        # Upload section
        upload_frame = ctk.CTkFrame(content_frame, fg_color=COLORS['bg_secondary'], corner_radius=12,
                                     border_width=1, border_color=COLORS['border'])
        upload_frame.pack(fill="x", pady=(0, 12))

        upload_content = ctk.CTkFrame(upload_frame, fg_color="transparent")
        upload_content.pack(padx=24, pady=24)

        ctk.CTkLabel(
            upload_content,
            text="Upload Email List",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(0, 16))

        file_frame = ctk.CTkFrame(upload_content, fg_color="transparent")
        file_frame.pack(fill="x", pady=8)

        self.file_label = ctk.CTkLabel(
            file_frame,
            text="No file selected",
            font=ctk.CTkFont(size=13),
            text_color=COLORS['text_secondary']
        )
        self.file_label.pack(side="left", padx=8)

        select_btn = ctk.CTkButton(
            file_frame,
            text="📁  Select File",
            command=self.select_bulk_file,
            width=140,
            height=36,
            fg_color=COLORS['accent_primary'],
            hover_color=COLORS['info'],
            corner_radius=8
        )
        select_btn.pack(side="right")

        bulk_btn_frame = ctk.CTkFrame(upload_content, fg_color="transparent")
        bulk_btn_frame.pack(fill="x", pady=16)

        self.process_btn = GradientButton(
            bulk_btn_frame,
            text="⚡ Start Bulk Analysis",
            command=self.process_bulk,
            height=44,
            font=ctk.CTkFont(size=15, weight="bold"),
            gradient_colors=[COLORS['success'], COLORS['accent_secondary']],
            corner_radius=8
        )
        self.process_btn.pack(side="left", fill="x", expand=True, padx=(0, 4))

        self.stop_bulk_btn = GradientButton(
            bulk_btn_frame,
            text="⏹ Stop",
            command=self.stop_bulk_analysis,
            height=44,
            font=ctk.CTkFont(size=15, weight="bold"),
            gradient_colors=[COLORS['danger'], '#ff7b72'],
            corner_radius=8,
            state="disabled"
        )
        self.stop_bulk_btn.pack(side="right", fill="x", expand=True, padx=(4, 0))

        # Progress section
        progress_frame = ctk.CTkFrame(content_frame, fg_color=COLORS['bg_secondary'], corner_radius=12,
                                       border_width=1, border_color=COLORS['border'])
        progress_frame.pack(fill="x", pady=(0, 12))

        progress_content = ctk.CTkFrame(progress_frame, fg_color="transparent")
        progress_content.pack(padx=24, pady=16)

        self.bulk_progress_label = ctk.CTkLabel(
            progress_content,
            text="Ready to process",
            font=ctk.CTkFont(size=14)
        )
        self.bulk_progress_label.pack(pady=8)

        self.bulk_progress_bar = AnimatedProgressBar(progress_content)
        self.bulk_progress_bar.pack(fill="x", pady=8)

        # Results section (hidden until analysis completes)
        self.bulk_results_frame = ctk.CTkFrame(content_frame, fg_color=COLORS['bg_secondary'], corner_radius=12,
                                                border_width=1, border_color=COLORS['border'])
        results_frame = self.bulk_results_frame

        results_header = ctk.CTkFrame(results_frame, fg_color="transparent")
        results_header.pack(fill="x", padx=24, pady=(16, 8))

        ctk.CTkLabel(
            results_header,
            text="Analysis Results",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(side="left")

        export_frame = ctk.CTkFrame(results_header, fg_color="transparent")
        export_frame.pack(side="right")

        for text, icon, cmd in [
            ("CSV", "📄", self.export_bulk_csv),
            ("Excel", "📊", self.export_bulk_excel),
            ("PDF", "📑", self.export_bulk_pdf)
        ]:
            btn = ctk.CTkButton(
                export_frame,
                text=f"{icon} {text}",
                command=cmd,
                width=80,
                height=32,
                fg_color=COLORS['bg_tertiary'],
                hover_color=COLORS['bg_quaternary'],
                corner_radius=6,
                font=ctk.CTkFont(size=11)
            )
            btn.pack(side="left", padx=4)

        self.bulk_results = ctk.CTkTextbox(
            results_frame,
            fg_color=COLORS['bg_tertiary'],
            text_color=COLORS['text_primary'],
            font=ctk.CTkFont(family="Consolas", size=11),
            corner_radius=8
        )
        self.bulk_results.pack(fill="both", expand=True, padx=24, pady=(0, 24))

        return page

    def create_monitor_page(self, parent):
        """Create real-time monitoring page"""
        page = ctk.CTkFrame(parent, fg_color=COLORS['bg_primary'])

        # Header
        header_frame = ctk.CTkFrame(page, fg_color="transparent")
        header_frame.pack(fill="x", padx=24, pady=(16, 12))

        ctk.CTkLabel(
            header_frame,
            text="Real-time Security Monitor",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color=COLORS['text_primary']
        ).pack(anchor="w")

        ctk.CTkLabel(
            header_frame,
            text="Live monitoring of email security events",
            font=ctk.CTkFont(size=12),
            text_color=COLORS['text_secondary']
        ).pack(anchor="w")

        # Control buttons
        control_frame = ctk.CTkFrame(page, fg_color="transparent")
        control_frame.pack(fill="x", padx=24, pady=(0, 8))

        self.monitor_status_label = ctk.CTkLabel(
            control_frame,
            text="● Monitor Active",
            font=ctk.CTkFont(size=13),
            text_color=COLORS['success']
        )
        self.monitor_status_label.pack(side="left")

        clear_btn = ctk.CTkButton(
            control_frame,
            text="Clear Log",
            command=self.clear_monitor_feed,
            width=100,
            height=36,
            fg_color=COLORS['bg_tertiary'],
            hover_color=COLORS['bg_quaternary'],
            corner_radius=8
        )
        clear_btn.pack(side="right", padx=8)

        pause_btn = ctk.CTkButton(
            control_frame,
            text="Pause",
            command=self.toggle_monitor,
            width=100,
            height=36,
            fg_color=COLORS['accent_primary'],
            corner_radius=8
        )
        pause_btn.pack(side="right")
        self.monitor_pause_btn = pause_btn

        # Monitor display frame
        monitor_frame = ctk.CTkFrame(page, fg_color=COLORS['bg_secondary'], corner_radius=12,
                                      border_width=1, border_color=COLORS['border'])
        monitor_frame.pack(fill="both", expand=True, padx=24, pady=(0, 16))

        self.monitor_feed = ctk.CTkTextbox(
            monitor_frame,
            fg_color=COLORS['bg_tertiary'],
            text_color=COLORS['text_primary'],
            font=ctk.CTkFont(family="Consolas", size=11),
            wrap="word",
            corner_radius=8
        )
        self.monitor_feed.pack(fill="both", expand=True, padx=16, pady=16)

        # Initialize with welcome message
        welcome_msg = """========================================
EMAIL SECURITY MONITOR - INITIALIZED
========================================
System ready. Monitoring for security events...

"""
        self.monitor_feed.insert("1.0", welcome_msg)

        # Monitor starts on first navigation to the page (see show_monitor)

        return page

    def start_monitor_simulation(self):
        """Start real-time monitoring simulation with proper data"""

        def generate_monitor_data():
            """Generate realistic monitoring data"""
            # Sample email addresses for simulation
            domains = [
                'gmail.com', 'yahoo.com', 'outlook.com', 'hotmail.com',
                'company.com', 'example.org', 'business.net',
                'suspicious-site.tk', 'phishing.ml', 'malware.ga',
                'fake-bank.cf', 'scam-alert.tk'
            ]

            users = [
                'john.doe', 'jane.smith', 'admin', 'user123', 'contact',
                'support', 'security', 'test.user', 'employee001', 'manager',
                'suspicious.sender', 'unknown.user', 'fake.admin'
            ]

            # Event templates (ASCII-safe for Windows console)
            events = [
                {
                    'type': 'scan',
                    'icon': '[SCAN]',
                    'messages': [
                        "Email scanned: {email} - Status: Clean",
                        "Quick scan completed: {email} - No threats detected",
                        "Deep analysis: {email} - SPF/DMARC passed"
                    ],
                    'color': COLORS['text_primary']
                },
                {
                    'type': 'threat',
                    'icon': '[ALERT]',
                    'messages': [
                        "THREAT DETECTED: {email} - Phishing attempt blocked",
                        "CRITICAL: Malware detected in attachment from {email}",
                        "WARNING: Suspicious link detected in email from {email}"
                    ],
                    'color': COLORS['danger']
                },
                {
                    'type': 'warning',
                    'icon': '[WARN]',
                    'messages': [
                        "Suspicious activity: {email} - Unknown sender",
                        "Alert: {email} - Failed SPF check",
                        "Caution: {email} - New domain detected (registered 5 days ago)"
                    ],
                    'color': COLORS['warning']
                },
                {
                    'type': 'success',
                    'icon': '[OK]',
                    'messages': [
                        "Verified: {email} - Authenticated successfully",
                        "Safe email confirmed: {email} - All security checks passed",
                        "Trusted sender verified: {email}"
                    ],
                    'color': COLORS['success']
                },
                {
                    'type': 'info',
                    'icon': '[INFO]',
                    'messages': [
                        "ML Analysis: {email} - Risk score: {score}/100",
                        "DNS Check: {email} - DNSSEC enabled",
                        "Reputation check: {email} - Score: {score}/100"
                    ],
                    'color': COLORS['info']
                },
                {
                    'type': 'breach',
                    'icon': '[BREACH]',
                    'messages': [
                        "DATA BREACH: {email} found in {count} breach database(s)",
                        "Alert: {email} - Credentials may be compromised",
                        "Security Notice: {email} - Password change recommended"
                    ],
                    'color': COLORS['accent_tertiary']
                }
            ]

            # Wait briefly for mainloop to start before generating events
            time.sleep(1.0)

            while self.monitor_running:
                try:
                    # Only use the boolean flag from the thread (no Tkinter calls)
                    if not hasattr(self, 'monitor_feed'):
                        time.sleep(0.5)
                        continue

                    # Generate random event
                    event = random.choice(events)
                    user = random.choice(users)
                    domain = random.choice(domains)
                    email = f"{user}@{domain}"

                    # Format message
                    message = random.choice(event['messages'])
                    score = random.randint(10, 95)
                    count = random.randint(1, 5)

                    # Replace all placeholders at once
                    message = message.format(
                        email=email,
                        score=score,
                        count=count
                    ) if '{' in message else message

                    # Create timestamp
                    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]

                    # Format log entry with color coding
                    log_entry = f"[{timestamp}] {event['icon']} {message}\n"

                    # Safely update the monitor feed (capture log_entry in default arg)
                    def update_feed(entry=log_entry):
                        try:
                            if hasattr(self, 'monitor_feed') and self.monitor_feed.winfo_exists():
                                # Insert at the beginning for latest-first view
                                self.monitor_feed.insert("1.0", entry)

                                # Keep only last 500 lines to prevent memory issues
                                content = self.monitor_feed.get("1.0", tk.END)
                                lines = content.split('\n')
                                if len(lines) > 500:
                                    self.monitor_feed.delete("501.0", tk.END)

                                # Auto-scroll to top to show latest
                                self.monitor_feed.see("1.0")
                        except Exception:
                            pass

                    # Schedule update in main thread (safe winfo_exists check)
                    self.ui(update_feed)

                    # Random delay between events (more realistic)
                    # Use interruptible sleep so pause/stop takes effect quickly
                    delay = random.uniform(0.5, 3.0)
                    end_time = time.time() + delay
                    while time.time() < end_time and self.monitor_running:
                        time.sleep(0.1)

                except Exception:
                    time.sleep(1)

        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=generate_monitor_data, daemon=True)
        self.monitor_thread.start()
        logger.debug("Monitor thread started")

    def clear_monitor_feed(self):
        """Clear the monitor feed"""
        if hasattr(self, 'monitor_feed') and self.monitor_feed.winfo_exists():
            self.monitor_feed.delete("1.0", tk.END)
            header = """========================================
EMAIL SECURITY MONITOR - CLEARED
========================================
Monitoring continues...

"""
            self.monitor_feed.insert("1.0", header)

    def toggle_monitor(self):
        """Toggle monitor pause/resume"""
        if hasattr(self, 'monitor_running'):
            self.monitor_running = not self.monitor_running

            if self.monitor_running:
                self.monitor_pause_btn.configure(text="Pause")
                self.monitor_status_label.configure(
                    text="● Monitor Active",
                    text_color=COLORS['success']
                )
                # Only restart if no thread is already running
                if not hasattr(self, 'monitor_thread') or not self.monitor_thread.is_alive():
                    self.start_monitor_simulation()

                # Add resume message
                timestamp = datetime.now().strftime("%H:%M:%S")
                msg = f"[{timestamp}] ▶️ Monitor RESUMED\n"
                if hasattr(self, 'monitor_feed') and self.monitor_feed.winfo_exists():
                    self.monitor_feed.insert("1.0", msg)
            else:
                self.monitor_pause_btn.configure(text="Resume")
                self.monitor_status_label.configure(
                    text="● Monitor Paused",
                    text_color=COLORS['warning']
                )

                # Add pause message
                timestamp = datetime.now().strftime("%H:%M:%S")
                msg = f"[{timestamp}] ⏸️ Monitor PAUSED\n"
                if hasattr(self, 'monitor_feed') and self.monitor_feed.winfo_exists():
                    self.monitor_feed.insert("1.0", msg)

    def create_mitre_page(self, parent):
        """Create MITRE ATT&CK page"""
        page = ctk.CTkScrollableFrame(parent, fg_color=COLORS['bg_primary'])

        header_frame = ctk.CTkFrame(page, fg_color="transparent")
        header_frame.pack(fill="x", padx=24, pady=(16, 12))

        ctk.CTkLabel(
            header_frame,
            text="MITRE ATT&CK Framework",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color=COLORS['text_primary']
        ).pack(anchor="w")

        techniques_frame = ctk.CTkFrame(page, fg_color="transparent")
        techniques_frame.pack(fill="both", expand=True, padx=24, pady=(0, 16))

        self.mitre_cards = []
        self.mitre_techniques_frame = techniques_frame

        loading_label = ctk.CTkLabel(
            techniques_frame,
            text="Loading MITRE ATT&CK techniques...",
            font=ctk.CTkFont(size=14),
            text_color=COLORS['text_tertiary']
        )
        loading_label.pack(expand=True, pady=40)

        return page

    def create_threat_intel_page(self, parent):
        """Create threat intelligence page"""
        page = ctk.CTkScrollableFrame(parent, fg_color=COLORS['bg_primary'])

        header_frame = ctk.CTkFrame(page, fg_color="transparent")
        header_frame.pack(fill="x", padx=24, pady=(16, 12))

        ctk.CTkLabel(
            header_frame,
            text="Threat Intelligence",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color=COLORS['text_primary']
        ).pack(anchor="w")

        feeds_frame = ctk.CTkFrame(page, fg_color=COLORS['bg_secondary'], corner_radius=12,
                                    border_width=1, border_color=COLORS['border'])
        feeds_frame.pack(fill="x", padx=24, pady=(0, 16))

        ctk.CTkLabel(
            feeds_frame,
            text="Active Threat Feeds",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(20, 12), padx=24, anchor="w")

        feeds_grid = ctk.CTkFrame(feeds_frame, fg_color="transparent")
        feeds_grid.pack(fill="x", padx=24, pady=(0, 24))

        self.threat_feeds_grid = feeds_grid

        if self.analyzer and hasattr(self.analyzer, 'threat_intel'):
            self.populate_threat_feeds(feeds_grid)
        else:
            loading_label = ctk.CTkLabel(
                feeds_grid,
                text="Loading threat intelligence feeds...",
                font=ctk.CTkFont(size=14),
                text_color=COLORS['text_tertiary']
            )
            loading_label.pack(expand=True, pady=40)

        return page

    def populate_threat_feeds(self, parent_frame):
        """Populate threat feeds in the given frame"""
        for widget in parent_frame.winfo_children():
            widget.destroy()

        if self.analyzer and hasattr(self.analyzer, 'threat_intel'):
            for i, (feed_name, feed_data) in enumerate(self.analyzer.threat_intel.threat_feeds.items()):
                feed_card = ctk.CTkFrame(parent_frame, fg_color=COLORS['bg_tertiary'], corner_radius=8,
                                          border_width=1, border_color=COLORS['border'])
                feed_card.grid(row=i // 2, column=i % 2, padx=6, pady=6, sticky="ew")

                feed_content = ctk.CTkFrame(feed_card, fg_color="transparent")
                feed_content.pack(padx=16, pady=12)

                ctk.CTkLabel(
                    feed_content,
                    text=feed_name.replace('_', ' ').title(),
                    font=ctk.CTkFont(size=14, weight="bold")
                ).pack()

                count = len(feed_data) if isinstance(feed_data, (set, list)) else 0
                ctk.CTkLabel(
                    feed_content,
                    text=f"{count} indicators",
                    font=ctk.CTkFont(size=20, weight="bold"),
                    text_color=COLORS['accent_primary']
                ).pack(pady=8)

                parent_frame.columnconfigure(i % 2, weight=1)

    def create_ml_page(self, parent):
        """Create ML models page"""
        page = ctk.CTkScrollableFrame(parent, fg_color=COLORS['bg_primary'])

        header_frame = ctk.CTkFrame(page, fg_color="transparent")
        header_frame.pack(fill="x", padx=24, pady=(16, 12))

        ctk.CTkLabel(
            header_frame,
            text="Machine Learning Models",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color=COLORS['text_primary']
        ).pack(anchor="w")

        # Training metrics notice — softer treatment
        self.ml_warning_frame = ctk.CTkFrame(page, fg_color=COLORS['bg_quaternary'], corner_radius=12,
                                              border_width=1, border_color=COLORS['warning'])
        self.ml_warning_frame.pack(fill="x", padx=24, pady=(0, 16))

        warning_content = ctk.CTkFrame(self.ml_warning_frame, fg_color="transparent")
        warning_content.pack(padx=20, pady=12)

        ctk.CTkLabel(
            warning_content,
            text="⚠️  Training Performance Metrics",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=COLORS['warning']
        ).pack(anchor="w")

        ctk.CTkLabel(
            warning_content,
            text="These metrics show model validation on test data, not real email predictions.",
            font=ctk.CTkFont(size=12),
            text_color=COLORS['text_secondary']
        ).pack(anchor="w")

        if not self.config.enable_ml:
            disabled_frame = ctk.CTkFrame(page, fg_color=COLORS['bg_secondary'], corner_radius=12,
                                           border_width=1, border_color=COLORS['border'])
            disabled_frame.pack(fill="x", padx=24, pady=16)

            disabled_content = ctk.CTkFrame(disabled_frame, fg_color="transparent")
            disabled_content.pack(padx=24, pady=24)

            ctk.CTkLabel(
                disabled_content,
                text="Machine Learning Disabled",
                font=ctk.CTkFont(size=20, weight="bold"),
                text_color=COLORS['warning']
            ).pack(pady=8)

            ctk.CTkLabel(
                disabled_content,
                text="ML features are currently disabled in settings.\n"
                     "Enable ML in config.ini [ml] section: auto_train = true",
                font=ctk.CTkFont(size=13),
                text_color=COLORS['text_secondary']
            ).pack(pady=8)

            return page

        models_frame = ctk.CTkFrame(page, fg_color="transparent")
        models_frame.pack(fill="both", expand=True, padx=24)

        self.ml_models_frame = models_frame

        if not self.analyzer or not hasattr(self.analyzer, 'ml_engine'):
            loading_label = ctk.CTkLabel(
                models_frame,
                text="Loading ML models...",
                font=ctk.CTkFont(size=14),
                text_color=COLORS['text_tertiary']
            )
            loading_label.pack(expand=True, pady=40)
        else:
            self.populate_ml_models(models_frame)

        # ML Engine Info
        info_frame = ctk.CTkFrame(page, fg_color=COLORS['bg_secondary'], corner_radius=12,
                                   border_width=1, border_color=COLORS['border'])
        info_frame.pack(fill="x", padx=24, pady=16)

        info_content = ctk.CTkFrame(info_frame, fg_color="transparent")
        info_content.pack(padx=24, pady=20)

        ctk.CTkLabel(
            info_content,
            text="ML Engine v2",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(0, 12), anchor="w")

        engine_info = ctk.CTkFrame(info_content, fg_color=COLORS['bg_tertiary'], corner_radius=8)
        engine_info.pack(fill="x", pady=8)

        engine_content = ctk.CTkFrame(engine_info, fg_color="transparent")
        engine_content.pack(padx=16, pady=12)

        ctk.CTkLabel(
            engine_content,
            text="XGBoost + Random Forest (Calibrated)",
            font=ctk.CTkFont(size=14),
            text_color=COLORS['accent_primary']
        ).pack(anchor="w")

        ctk.CTkLabel(
            engine_content,
            text="44 features | Precision-optimized threshold | Isolation Forest anomaly detection",
            font=ctk.CTkFont(size=12),
            text_color=COLORS['text_secondary']
        ).pack(anchor="w", pady=4)

        if self.analyzer and hasattr(self.analyzer, 'ml_engine') and self.analyzer.ml_engine:
            ml_engine = self.analyzer.ml_engine
            models_file = self.config.models_dir / "ml_v2_models.pkl"
            if models_file.exists():
                import os
                size_kb = os.path.getsize(models_file) / 1024
                ctk.CTkLabel(
                    engine_content,
                    text=f"Model saved: {models_file.name} ({size_kb:.0f} KB)",
                    font=ctk.CTkFont(size=11),
                    text_color=COLORS['success']
                ).pack(anchor="w", pady=2)
            feedback_count = ml_engine.get_feedback_count() if hasattr(ml_engine, 'get_feedback_count') else 0
            ctk.CTkLabel(
                engine_content,
                text=f"Feedback samples: {feedback_count} | Threshold: {ml_engine.precision_threshold:.4f}",
                font=ctk.CTkFont(size=11),
                text_color=COLORS['text_secondary']
            ).pack(anchor="w", pady=2)

        return page

    def populate_ml_models(self, parent_frame):
        """Populate ML models in the given frame"""
        # Clear existing content
        for widget in parent_frame.winfo_children():
            widget.destroy()

        if self.analyzer and hasattr(self.analyzer, 'ml_engine') and self.analyzer.ml_engine:
            ml_engine = self.analyzer.ml_engine

            # Check if any predictions have been made
            has_predictions = hasattr(ml_engine, 'prediction_history') and len(ml_engine.prediction_history) > 0

            # Hide or show warning banner based on predictions
            if hasattr(self, 'ml_warning_frame'):
                try:
                    if has_predictions:
                        self.ml_warning_frame.pack_forget()  # Hide the banner
                except Exception:
                    pass  # Ignore if banner can't be hidden

            if not has_predictions:
                # Show "No predictions yet" message
                message_frame = ctk.CTkFrame(parent_frame, fg_color=COLORS['bg_secondary'], corner_radius=12, border_width=1, border_color=COLORS['border'])
                message_frame.grid(row=0, column=0, columnspan=3, padx=20, pady=50, sticky="nsew")

                message_content = ctk.CTkFrame(message_frame, fg_color="transparent")
                message_content.pack(padx=40, pady=40)

                ctk.CTkLabel(
                    message_content,
                    text="📧 No Email Analysis Yet",
                    font=ctk.CTkFont(size=24, weight="bold"),
                    text_color=COLORS['accent_primary']
                ).pack(pady=10)

                ctk.CTkLabel(
                    message_content,
                    text="ML prediction statistics will appear here after you analyze emails",
                    font=ctk.CTkFont(size=14),
                    text_color=COLORS['text_secondary']
                ).pack(pady=5)

                ctk.CTkLabel(
                    message_content,
                    text="Go to 'Email Analysis' tab to start analyzing emails",
                    font=ctk.CTkFont(size=12),
                    text_color=COLORS['info']
                ).pack(pady=10)

                return

            # Show REAL prediction statistics from prediction_history
            for i, (model_name, model) in enumerate(ml_engine.models.items()):
                if model is None:
                    continue

                model_card = ctk.CTkFrame(parent_frame, fg_color=COLORS['bg_secondary'], corner_radius=12, border_width=1, border_color=COLORS['border'])
                model_card.grid(row=i // 3, column=i % 3, padx=6, pady=6, sticky="nsew")

                # Model info
                model_content = ctk.CTkFrame(model_card, fg_color="transparent")
                model_content.pack(padx=20, pady=20)

                # Model name
                ctk.CTkLabel(
                    model_content,
                    text=model_name.replace('_', ' ').title(),
                    font=ctk.CTkFont(size=16, weight="bold"),
                    text_color=COLORS['accent_primary']
                ).pack()

                # Model type
                ctk.CTkLabel(
                    model_content,
                    text=type(model).__name__,
                    font=ctk.CTkFont(size=12),
                    text_color=COLORS['text_secondary']
                ).pack(pady=5)

                # Calculate prediction statistics from prediction_history
                total_predictions = len(ml_engine.prediction_history)

                # Calculate this model's statistics from prediction history
                model_scores = []
                for pred_record in ml_engine.prediction_history:
                    if model_name in pred_record.get('predictions', {}):
                        score = pred_record['predictions'][model_name]
                        model_scores.append(score)

                # Calculate average confidence and risk distribution
                if model_scores:
                    avg_confidence = sum(model_scores) / len(model_scores)
                    high_risk = sum(1 for s in model_scores if s > 0.7)
                    medium_risk = sum(1 for s in model_scores if 0.3 <= s <= 0.7)
                    low_risk = sum(1 for s in model_scores if s < 0.3)
                else:
                    avg_confidence = 0.0
                    high_risk = medium_risk = low_risk = 0

                # Show prediction statistics
                pred_info_frame = ctk.CTkFrame(model_content, fg_color=COLORS['bg_tertiary'], corner_radius=8)
                pred_info_frame.pack(fill="x", pady=10)

                ctk.CTkLabel(
                    pred_info_frame,
                    text=f"Total Predictions: {total_predictions}",
                    font=ctk.CTkFont(size=12, weight="bold"),
                    text_color=COLORS['accent_primary']
                ).pack(pady=(10, 5))

                ctk.CTkLabel(
                    pred_info_frame,
                    text=f"Avg Confidence: {avg_confidence:.1%}",
                    font=ctk.CTkFont(size=11),
                    text_color=COLORS['info']
                ).pack(pady=2)

                # Risk distribution
                risk_text = f"High: {high_risk} | Med: {medium_risk} | Low: {low_risk}"
                ctk.CTkLabel(
                    pred_info_frame,
                    text=risk_text,
                    font=ctk.CTkFont(size=10),
                    text_color=COLORS['text_secondary']
                ).pack(pady=2)

                ctk.CTkLabel(
                    pred_info_frame,
                    text="(Real email analysis data)",
                    font=ctk.CTkFont(size=9),
                    text_color=COLORS['text_secondary']
                ).pack(pady=(2, 10))

                # Status indicator
                ctk.CTkLabel(
                    model_content,
                    text="● Active - Analyzing Emails",
                    font=ctk.CTkFont(size=12),
                    text_color=COLORS['success']
                ).pack(pady=5)

                parent_frame.columnconfigure(i % 3, weight=1)

    def create_dns_page(self, parent):
        """Create DNS security page"""
        page = ctk.CTkFrame(parent, fg_color=COLORS['bg_primary'])

        header_frame = ctk.CTkFrame(page, fg_color="transparent")
        header_frame.pack(fill="x", padx=24, pady=(16, 12))

        ctk.CTkLabel(
            header_frame,
            text="DNS Security Checker",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color=COLORS['text_primary']
        ).pack(anchor="w")

        check_frame = ctk.CTkFrame(page, fg_color=COLORS['bg_secondary'], corner_radius=12,
                                    border_width=1, border_color=COLORS['border'])
        check_frame.pack(fill="x", padx=24, pady=(0, 12))

        check_content = ctk.CTkFrame(check_frame, fg_color="transparent")
        check_content.pack(padx=24, pady=24)

        ctk.CTkLabel(
            check_content,
            text="Enter Domain to Check",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(0, 12))

        self.dns_entry = ctk.CTkEntry(
            check_content,
            placeholder_text="example.com",
            height=44,
            font=ctk.CTkFont(size=14),
            fg_color=COLORS['bg_tertiary'],
            border_color=COLORS['border'],
            border_width=1,
            corner_radius=8
        )
        self.dns_entry.pack(fill="x", pady=8)

        dns_check_btn = GradientButton(
            check_content,
            text="🔍 Check DNS Security",
            command=self.check_dns_security,
            height=44,
            font=ctk.CTkFont(size=15, weight="bold"),
            gradient_colors=[COLORS['accent_primary'], COLORS['accent_secondary']],
            corner_radius=8
        )
        dns_check_btn.pack(fill="x", pady=8)

        self.dns_results = ctk.CTkFrame(page, fg_color=COLORS['bg_secondary'], corner_radius=12,
                                         border_width=1, border_color=COLORS['border'])
        self.dns_results.pack(fill="both", expand=True, padx=24, pady=(0, 16))
        self.dns_results.pack_forget()

        return page

    def create_analytics_page(self, parent):
        """Create analytics page"""
        page = ctk.CTkScrollableFrame(parent, fg_color=COLORS['bg_primary'])

        header_frame = ctk.CTkFrame(page, fg_color="transparent")
        header_frame.pack(fill="x", padx=24, pady=(16, 12))

        ctk.CTkLabel(
            header_frame,
            text="Security Analytics",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color=COLORS['text_primary']
        ).pack(anchor="w")

        charts_container = ctk.CTkFrame(page, fg_color="transparent")
        charts_container.pack(fill="both", expand=True, padx=24)

        # Row 1: Risk Distribution & Heatmap
        # 1. Risk Level Distribution (Pie Chart)
        risk_dist_frame = ctk.CTkFrame(charts_container, fg_color=COLORS['bg_secondary'], corner_radius=12, border_width=1, border_color=COLORS['border'])
        risk_dist_frame.grid(row=0, column=0, padx=6, pady=6, sticky="nsew")

        ctk.CTkLabel(
            risk_dist_frame,
            text="Risk Level Distribution",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(16, 8))

        self.risk_dist_fig = Figure(figsize=(6, 5), facecolor='#161b22')
        self.risk_dist_ax = self.risk_dist_fig.add_subplot(111)
        self.risk_dist_ax.set_facecolor('#161b22')

        self.risk_dist_canvas = FigureCanvasTkAgg(self.risk_dist_fig, risk_dist_frame)
        self.risk_dist_canvas.get_tk_widget().pack(fill="both", expand=True, padx=16, pady=(0, 16))

        # 2. Risk Heatmap by Domain
        heatmap_frame = ctk.CTkFrame(charts_container, fg_color=COLORS['bg_secondary'], corner_radius=12, border_width=1, border_color=COLORS['border'])
        heatmap_frame.grid(row=0, column=1, padx=6, pady=6, sticky="nsew")

        ctk.CTkLabel(
            heatmap_frame,
            text="Risk Heatmap by Domain",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(16, 8))

        self.heatmap_fig = Figure(figsize=(8, 5), facecolor='#161b22')
        self.heatmap_ax = self.heatmap_fig.add_subplot(111)
        self.heatmap_ax.set_facecolor('#161b22')

        self.heatmap_canvas = FigureCanvasTkAgg(self.heatmap_fig, heatmap_frame)
        self.heatmap_canvas.get_tk_widget().pack(fill="both", expand=True, padx=16, pady=(0, 16))

        # Row 2: Top Threats & Breach Statistics
        # 3. Top Threat Types (Bar Chart)
        threat_types_frame = ctk.CTkFrame(charts_container, fg_color=COLORS['bg_secondary'], corner_radius=12, border_width=1, border_color=COLORS['border'])
        threat_types_frame.grid(row=1, column=0, padx=6, pady=6, sticky="nsew")

        ctk.CTkLabel(
            threat_types_frame,
            text="Top Threat Types",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(16, 8))

        self.threat_types_fig = Figure(figsize=(6, 5), facecolor='#161b22')
        self.threat_types_ax = self.threat_types_fig.add_subplot(111)
        self.threat_types_ax.set_facecolor('#161b22')

        self.threat_types_canvas = FigureCanvasTkAgg(self.threat_types_fig, threat_types_frame)
        self.threat_types_canvas.get_tk_widget().pack(fill="both", expand=True, padx=16, pady=(0, 16))

        # 4. Breach Statistics (Bar Chart)
        breach_stats_frame = ctk.CTkFrame(charts_container, fg_color=COLORS['bg_secondary'], corner_radius=12, border_width=1, border_color=COLORS['border'])
        breach_stats_frame.grid(row=1, column=1, padx=6, pady=6, sticky="nsew")

        ctk.CTkLabel(
            breach_stats_frame,
            text="Breach Statistics by Domain",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(16, 8))

        self.breach_stats_fig = Figure(figsize=(8, 5), facecolor='#161b22')
        self.breach_stats_ax = self.breach_stats_fig.add_subplot(111)
        self.breach_stats_ax.set_facecolor('#161b22')

        self.breach_stats_canvas = FigureCanvasTkAgg(self.breach_stats_fig, breach_stats_frame)
        self.breach_stats_canvas.get_tk_widget().pack(fill="both", expand=True, padx=16, pady=(0, 16))

        # Row 3: Risk Score Histogram & MITRE Techniques
        # 5. Risk Score Distribution (Histogram)
        risk_hist_frame = ctk.CTkFrame(charts_container, fg_color=COLORS['bg_secondary'], corner_radius=12, border_width=1, border_color=COLORS['border'])
        risk_hist_frame.grid(row=2, column=0, padx=6, pady=6, sticky="nsew")

        ctk.CTkLabel(
            risk_hist_frame,
            text="Risk Score Distribution",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(16, 8))

        self.risk_hist_fig = Figure(figsize=(6, 5), facecolor='#161b22')
        self.risk_hist_ax = self.risk_hist_fig.add_subplot(111)
        self.risk_hist_ax.set_facecolor('#161b22')

        self.risk_hist_canvas = FigureCanvasTkAgg(self.risk_hist_fig, risk_hist_frame)
        self.risk_hist_canvas.get_tk_widget().pack(fill="both", expand=True, padx=16, pady=(0, 16))

        # 6. Top MITRE Techniques (Horizontal Bar)
        mitre_top_frame = ctk.CTkFrame(charts_container, fg_color=COLORS['bg_secondary'], corner_radius=12, border_width=1, border_color=COLORS['border'])
        mitre_top_frame.grid(row=2, column=1, padx=6, pady=6, sticky="nsew")

        ctk.CTkLabel(
            mitre_top_frame,
            text="Top 10 MITRE ATT&CK Techniques",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(16, 8))

        self.mitre_top_fig = Figure(figsize=(8, 5), facecolor='#161b22')
        self.mitre_top_ax = self.mitre_top_fig.add_subplot(111)
        self.mitre_top_ax.set_facecolor('#161b22')

        self.mitre_top_canvas = FigureCanvasTkAgg(self.mitre_top_fig, mitre_top_frame)
        self.mitre_top_canvas.get_tk_widget().pack(fill="both", expand=True, padx=16, pady=(0, 16))

        # Row 3: ML Prediction Charts
        # 7. ML Model Performance Comparison (Bar Chart)
        ml_models_frame = ctk.CTkFrame(charts_container, fg_color=COLORS['bg_secondary'], corner_radius=12, border_width=1, border_color=COLORS['border'])
        ml_models_frame.grid(row=3, column=0, padx=6, pady=6, sticky="nsew")

        ctk.CTkLabel(
            ml_models_frame,
            text="ML Model Performance",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(16, 8))

        self.ml_models_fig = Figure(figsize=(6, 5), facecolor='#161b22')
        self.ml_models_ax = self.ml_models_fig.add_subplot(111)
        self.ml_models_ax.set_facecolor('#161b22')

        self.ml_models_canvas = FigureCanvasTkAgg(self.ml_models_fig, ml_models_frame)
        self.ml_models_canvas.get_tk_widget().pack(fill="both", expand=True, padx=16, pady=(0, 16))

        # 8. Prediction Confidence Distribution (Doughnut Chart)
        ml_confidence_frame = ctk.CTkFrame(charts_container, fg_color=COLORS['bg_secondary'], corner_radius=12, border_width=1, border_color=COLORS['border'])
        ml_confidence_frame.grid(row=3, column=1, padx=6, pady=6, sticky="nsew")

        ctk.CTkLabel(
            ml_confidence_frame,
            text="ML Prediction Confidence",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(16, 8))

        self.ml_confidence_fig = Figure(figsize=(6, 5), facecolor='#161b22')
        self.ml_confidence_ax = self.ml_confidence_fig.add_subplot(111)
        self.ml_confidence_ax.set_facecolor('#161b22')

        self.ml_confidence_canvas = FigureCanvasTkAgg(self.ml_confidence_fig, ml_confidence_frame)
        self.ml_confidence_canvas.get_tk_widget().pack(fill="both", expand=True, padx=16, pady=(0, 16))

        # Row 4: ML Risk Level Predictions
        # 9. ML Predictions by Risk Level (Horizontal Bar Chart)
        ml_risk_frame = ctk.CTkFrame(charts_container, fg_color=COLORS['bg_secondary'], corner_radius=12, border_width=1, border_color=COLORS['border'])
        ml_risk_frame.grid(row=4, column=0, columnspan=2, padx=6, pady=6, sticky="nsew")

        ctk.CTkLabel(
            ml_risk_frame,
            text="ML Predictions by Risk Level",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(16, 8))

        self.ml_risk_fig = Figure(figsize=(12, 5), facecolor='#161b22')
        self.ml_risk_ax = self.ml_risk_fig.add_subplot(111)
        self.ml_risk_ax.set_facecolor('#161b22')

        self.ml_risk_canvas = FigureCanvasTkAgg(self.ml_risk_fig, ml_risk_frame)
        self.ml_risk_canvas.get_tk_widget().pack(fill="both", expand=True, padx=16, pady=(0, 16))

        # Row 5: ML Classification Metrics Table
        metrics_table_frame = ctk.CTkFrame(charts_container, fg_color=COLORS['bg_secondary'], corner_radius=12, border_width=1, border_color=COLORS['border'])
        metrics_table_frame.grid(row=5, column=0, padx=6, pady=6, sticky="nsew")

        ctk.CTkLabel(
            metrics_table_frame,
            text="ML Classification Metrics",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(16, 8))

        # Create scrollable frame for table
        self.metrics_table_container = ctk.CTkScrollableFrame(
            metrics_table_frame,
            fg_color=COLORS['bg_tertiary'],
            height=300
        )
        self.metrics_table_container.pack(fill="both", expand=True, padx=16, pady=(0, 16))

        # Row 5: Confusion Matrix Heatmap
        confusion_matrix_frame = ctk.CTkFrame(charts_container, fg_color=COLORS['bg_secondary'], corner_radius=12, border_width=1, border_color=COLORS['border'])
        confusion_matrix_frame.grid(row=5, column=1, padx=6, pady=6, sticky="nsew")

        ctk.CTkLabel(
            confusion_matrix_frame,
            text="Confusion Matrix (Ensemble)",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(16, 8))

        self.confusion_fig = Figure(figsize=(6, 5), facecolor='#161b22')
        self.confusion_ax = self.confusion_fig.add_subplot(111)
        self.confusion_ax.set_facecolor('#161b22')

        self.confusion_canvas = FigureCanvasTkAgg(self.confusion_fig, confusion_matrix_frame)
        self.confusion_canvas.get_tk_widget().pack(fill="both", expand=True, padx=16, pady=(0, 16))

        # Configure grid
        for i in range(2):
            charts_container.columnconfigure(i, weight=1)
        for i in range(6):  # Changed from 5 to 6 rows
            charts_container.rowconfigure(i, weight=1)

        # Update analytics
        self.update_analytics_charts()

        return page

    def create_reports_page(self, parent):
        """Create reports page"""
        page = ctk.CTkScrollableFrame(parent, fg_color=COLORS['bg_primary'])

        header_frame = ctk.CTkFrame(page, fg_color="transparent")
        header_frame.pack(fill="x", padx=24, pady=(16, 12))

        ctk.CTkLabel(
            header_frame,
            text="Security Reports",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color=COLORS['text_primary']
        ).pack(anchor="w")

        options_frame = ctk.CTkFrame(page, fg_color=COLORS['bg_secondary'], corner_radius=12, border_width=1, border_color=COLORS['border'])
        options_frame.pack(fill="x", padx=24, pady=(0, 12))

        options_content = ctk.CTkFrame(options_frame, fg_color="transparent")
        options_content.pack(padx=24, pady=24)

        ctk.CTkLabel(
            options_content,
            text="Generate Security Report",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(0, 12), anchor="w")

        # Report type selection
        report_types = [
            ("Executive Summary", "executive", "📊"),
            ("Technical Report", "technical", "🔧"),
            ("Threat Analysis", "threats", "🚨"),
            ("Compliance Report", "compliance", "📋")
        ]

        self.report_type_var = ctk.StringVar(value="executive")

        for text, value, icon in report_types:
            btn = ctk.CTkRadioButton(
                options_content,
                text=f"{icon} {text}",
                variable=self.report_type_var,
                value=value,
                font=ctk.CTkFont(size=13)
            )
            btn.pack(anchor="w", pady=4)

        # Date range
        date_frame = ctk.CTkFrame(options_content, fg_color="transparent")
        date_frame.pack(fill="x", pady=20)

        ctk.CTkLabel(
            date_frame,
            text="Date Range:",
            font=ctk.CTkFont(size=14)
        ).pack(side="left", padx=(0, 10))

        self.date_range_var = ctk.StringVar(value="Last 30 Days")
        date_menu = ctk.CTkOptionMenu(
            date_frame,
            variable=self.date_range_var,
            values=["Last 7 Days", "Last 30 Days", "Last 90 Days", "All Time"],
            width=200
        )
        date_menu.pack(side="left")

        generate_btn = GradientButton(
            options_content,
            text="📄 Generate Report",
            command=self.generate_report,
            height=44,
            font=ctk.CTkFont(size=15, weight="bold"),
            gradient_colors=[COLORS['accent_primary'], COLORS['accent_secondary']],
            corner_radius=8
        )
        generate_btn.pack(fill="x", pady=16)

        export_frame = ctk.CTkFrame(page, fg_color=COLORS['bg_secondary'], corner_radius=12, border_width=1, border_color=COLORS['border'])
        export_frame.pack(fill="x", padx=24, pady=(0, 12))

        export_content = ctk.CTkFrame(export_frame, fg_color="transparent")
        export_content.pack(padx=24, pady=24)

        ctk.CTkLabel(
            export_content,
            text="Export Options",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(0, 12), anchor="w")

        # Enterprise Reports Section
        enterprise_frame = ctk.CTkFrame(export_content, fg_color=COLORS['bg_tertiary'], corner_radius=12, border_width=1, border_color=COLORS['border'])
        enterprise_frame.pack(fill="x", pady=10, padx=10)

        ctk.CTkLabel(
            enterprise_frame,
            text="Enterprise Reports",
            font=ctk.CTkFont(size=15, weight="bold"),
            text_color=COLORS['accent_primary']
        ).pack(pady=(16, 12), padx=16, anchor="w")

        enterprise_buttons = ctk.CTkFrame(enterprise_frame, fg_color="transparent")
        enterprise_buttons.pack(padx=16, pady=(0, 16))

        enterprise_options = [
            ("Enterprise HTML", "🎨", self.export_enterprise_html),
            ("Enterprise Excel", "📊", self.export_enterprise_excel)
        ]

        for text, icon, command in enterprise_options:
            btn = GradientButton(
                enterprise_buttons,
                text=f"{icon} {text}",
                command=command,
                width=180,
                height=44,
                font=ctk.CTkFont(size=14, weight="bold"),
                gradient_colors=[COLORS['accent_primary'], COLORS['accent_secondary']],
                corner_radius=8
            )
            btn.pack(side="left", padx=8)

        # Standard Reports
        standard_frame = ctk.CTkFrame(export_content, fg_color="transparent")
        standard_frame.pack(fill="x", pady=(16, 0))

        ctk.CTkLabel(
            standard_frame,
            text="Standard Reports",
            font=ctk.CTkFont(size=15, weight="bold"),
            text_color=COLORS['text_primary']
        ).pack(pady=(0, 12), anchor="w")

        export_buttons_grid = ctk.CTkFrame(standard_frame, fg_color="transparent")
        export_buttons_grid.pack()

        export_options = [
            ("PDF", "📑", self.export_pdf),
            ("HTML", "🌐", self.export_html),
            ("Excel", "📊", self.export_excel),
            ("CSV", "📄", self.export_csv),
            ("JSON", "📋", self.export_json)
        ]

        for idx, (text, icon, command) in enumerate(export_options):
            btn = ctk.CTkButton(
                export_buttons_grid,
                text=f"{icon}\n{text}",
                command=command,
                width=110,
                height=70,
                font=ctk.CTkFont(size=13, weight="bold"),
                fg_color=COLORS['bg_tertiary'],
                hover_color=COLORS['bg_quaternary'],
                corner_radius=8
            )
            btn.grid(row=0, column=idx, padx=6, pady=4)

        return page

    def create_audit_page(self, parent):
        """Create audit log page"""
        page = ctk.CTkFrame(parent, fg_color=COLORS['bg_primary'])

        header_frame = ctk.CTkFrame(page, fg_color="transparent")
        header_frame.pack(fill="x", padx=24, pady=(16, 12))

        ctk.CTkLabel(
            header_frame,
            text="Security Audit Log",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color=COLORS['text_primary']
        ).pack(side="left")

        clear_btn = ctk.CTkButton(
            header_frame,
            text="🗑️  Clear Log",
            command=self.clear_audit_log,
            fg_color=COLORS['danger'],
            hover_color='#ff7b72',
            height=36,
            corner_radius=8
        )
        clear_btn.pack(side="right")

        log_frame = ctk.CTkFrame(page, fg_color=COLORS['bg_secondary'], corner_radius=12, border_width=1, border_color=COLORS['border'])
        log_frame.pack(fill="both", expand=True, padx=24, pady=(0, 16))

        self.audit_log = ctk.CTkTextbox(
            log_frame,
            fg_color=COLORS['bg_tertiary'],
            text_color=COLORS['text_primary'],
            font=ctk.CTkFont(family="Consolas", size=11),
            corner_radius=8
        )
        self.audit_log.pack(fill="both", expand=True, padx=16, pady=16)

        return page

    def create_settings_page(self, parent):
        """Create settings page"""
        page = ctk.CTkScrollableFrame(parent, fg_color=COLORS['bg_primary'])

        header_frame = ctk.CTkFrame(page, fg_color="transparent")
        header_frame.pack(fill="x", padx=24, pady=(16, 12))

        ctk.CTkLabel(
            header_frame,
            text="Settings",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color=COLORS['text_primary']
        ).pack(anchor="w")

        # Settings sections - load from config
        sections = [
            {
                'title': 'Appearance',
                'settings': [
                    ('Dark Mode', 'switch', getattr(self.config, 'theme_dark', True), 'theme_dark'),
                    ('Animations', 'switch', getattr(self.config, 'enable_animations', True), 'enable_animations'),
                    ('Particle Effects', 'switch', getattr(self.config, 'enable_particles', False), 'enable_particles'),
                    ('Sound Effects', 'switch', getattr(self.config, 'enable_sounds', False), 'enable_sounds')
                ]
            },
            {
                'title': 'Security Analysis',
                'settings': [
                    ('Deep Scan by Default', 'switch', getattr(self.config, 'deep_scan_default', True), 'deep_scan_default'),
                    ('Auto-update Threat Feeds', 'switch', getattr(self.config, 'auto_update_feeds', True), 'auto_update_feeds'),
                    ('Enable ML Analysis', 'switch', self.config.enable_ml, 'enable_ml'),
                    ('Enable DNS Checks', 'switch', self.config.enable_dns, 'enable_dns'),
                    ('Enable WHOIS Lookup', 'switch', self.config.enable_whois, 'enable_whois'),
                    ('Enable Breach Detection', 'switch', self.config.enable_breach_check, 'enable_breach_check')
                ]
            },
            {
                'title': 'Network',
                'settings': [
                    ('Verify SSL Certificates', 'switch', getattr(self.config, 'ssl_verify', True), 'ssl_verify')
                ]
            },
            {
                'title': 'Performance',
                'settings': [
                    ('Max Workers', 'slider', self.config.max_workers, 'max_workers', (1, 20)),
                    ('Batch Size', 'slider', self.config.batch_size, 'batch_size', (10, 500)),
                    ('Timeout (seconds)', 'slider', self.config.timeout, 'timeout', (5, 120)),
                    ('Cache Size (MB)', 'slider', self.config.cache_size, 'cache_size', (10, 1000))
                ]
            },
            {
                'title': 'Machine Learning',
                'settings': [
                    ('ML Threshold', 'slider', self.config.ml_threshold, 'ml_threshold', (0.1, 1.0)),
                    ('Anomaly Contamination', 'slider', self.config.anomaly_contamination, 'anomaly_contamination', (0.01, 0.5)),
                    ('Model Update Frequency', 'dropdown', getattr(self.config, 'model_update_freq', 'Weekly'), 'model_update_freq',
                     ['Daily', 'Weekly', 'Monthly', 'Manual'])
                ]
            }
        ]

        self.settings_vars = {}

        for section in sections:
            section_frame = ctk.CTkFrame(page, fg_color=COLORS['bg_secondary'], corner_radius=12, border_width=1, border_color=COLORS['border'])
            section_frame.pack(fill="x", padx=24, pady=8)

            ctk.CTkLabel(
                section_frame,
                text=section['title'],
                font=ctk.CTkFont(size=16, weight="bold"),
                text_color=COLORS['accent_primary']
            ).pack(pady=(16, 10), padx=24, anchor="w")

            # Settings items
            for setting in section['settings']:
                setting_name, setting_type, default, key = setting[:4]

                setting_row = ctk.CTkFrame(section_frame, fg_color="transparent")
                setting_row.pack(fill="x", padx=24, pady=6)

                ctk.CTkLabel(
                    setting_row,
                    text=setting_name,
                    font=ctk.CTkFont(size=13),
                    anchor="w"
                ).pack(side="left", fill="x", expand=True)

                if setting_type == 'switch':
                    var = ctk.BooleanVar(value=default)

                    # Create visible ON/OFF text indicator
                    status_text = ctk.CTkLabel(
                        setting_row,
                        text="ON" if default else "OFF",
                        font=ctk.CTkFont(size=11, weight="bold"),
                        text_color=COLORS['success'] if default else COLORS['text_secondary'],
                        width=40
                    )
                    status_text.pack(side="right", padx=(0, 10))

                    # Add command callback with text indicator update
                    def make_toggle_callback(setting_key, setting_var, setting_label, text_widget):
                        def callback():
                            new_value = setting_var.get()
                            logger.info(f"Setting '{setting_key}' toggled to: {new_value}")
                            logger.debug(f"[SETTINGS] {setting_key} = {new_value}")

                            # Update text indicator
                            text_widget.configure(
                                text="ON" if new_value else "OFF",
                                text_color=COLORS['success'] if new_value else COLORS['text_secondary']
                            )

                            # Update status label
                            if hasattr(self, 'settings_status_label'):
                                self.settings_status_label.configure(
                                    text=f"✓ {setting_label} = {'ON' if new_value else 'OFF'} (click Save to apply)"
                                )
                        return callback

                    switch = ctk.CTkSwitch(
                        setting_row,
                        text="",
                        variable=var,
                        command=make_toggle_callback(key, var, setting_name, status_text),
                        fg_color=COLORS['accent_primary'],
                        progress_color=COLORS['accent_secondary'],
                        onvalue=True,
                        offvalue=False
                    )
                    switch.pack(side="right", padx=10)

                    self.settings_vars[key] = var

                elif setting_type == 'slider':
                    min_val, max_val = setting[4] if len(setting) > 4 else (0, 100)

                    value_label = ctk.CTkLabel(
                        setting_row,
                        text=str(default),
                        font=ctk.CTkFont(size=14),
                        width=50
                    )
                    value_label.pack(side="right", padx=(10, 0))

                    var = ctk.DoubleVar(value=default)
                    slider = ctk.CTkSlider(
                        setting_row,
                        from_=min_val,
                        to=max_val,
                        variable=var,
                        fg_color=COLORS['bg_tertiary'],
                        progress_color=COLORS['accent_primary'],
                        width=200
                    )
                    slider.pack(side="right", padx=10)

                    # Update label when slider changes
                    def update_label(val, label=value_label, is_int=(min_val == int(min_val))):
                        if is_int:
                            label.configure(text=str(int(val)))
                        else:
                            label.configure(text=f"{val:.2f}")

                    slider.configure(command=update_label)
                    self.settings_vars[key] = var

                elif setting_type == 'dropdown':
                    options = setting[4]
                    var = ctk.StringVar(value=default)
                    dropdown = ctk.CTkOptionMenu(
                        setting_row,
                        variable=var,
                        values=options,
                        width=150,
                        fg_color=COLORS['bg_tertiary']
                    )
                    dropdown.pack(side="right", padx=10)
                    self.settings_vars[key] = var

            # Add some padding at the bottom
            ctk.CTkFrame(section_frame, height=10, fg_color="transparent").pack()

        # Status label (shows when settings are changed)
        self.settings_status_label = ctk.CTkLabel(
            page,
            text="",
            font=ctk.CTkFont(size=12),
            text_color=COLORS['info']
        )
        self.settings_status_label.pack(pady=5)

        save_frame = ctk.CTkFrame(page, fg_color="transparent")
        save_frame.pack(fill="x", padx=24, pady=16)

        save_btn = GradientButton(
            save_frame,
            text="💾 Save Settings",
            command=self.save_settings,
            width=180,
            height=44,
            font=ctk.CTkFont(size=15, weight="bold"),
            gradient_colors=[COLORS['success'], COLORS['accent_secondary']],
            corner_radius=8
        )
        save_btn.pack()

        return page

    def create_about_page(self, parent):
        """Create about page"""
        page = ctk.CTkFrame(parent, fg_color=COLORS['bg_primary'])

        content_frame = ctk.CTkFrame(page, fg_color="transparent")
        content_frame.pack(expand=True)

        logo_label = ctk.CTkLabel(
            content_frame,
            text="🛡️",
            font=ctk.CTkFont(size=80)
        )
        logo_label.pack(pady=(24, 12))

        ctk.CTkLabel(
            content_frame,
            text="AI-MailArmor Ultimate",
            font=ctk.CTkFont(size=28, weight="bold"),
            text_color=COLORS['text_primary']
        ).pack(pady=8)

        version_text = "Version 6.0.1 Premium"
        if PYTHON_313_COMPAT:
            version_text += "  •  Python 3.13 Compatible"

        ctk.CTkLabel(
            content_frame,
            text=version_text,
            font=ctk.CTkFont(size=14),
            text_color=COLORS['accent_primary']
        ).pack()

        desc_frame = ctk.CTkFrame(content_frame, fg_color=COLORS['bg_secondary'], corner_radius=12,
                                   border_width=1, border_color=COLORS['border'])
        desc_frame.pack(pady=24, padx=40)

        desc_text = """The most advanced email security analysis platform featuring:

  XGBoost + Random Forest ML Ensemble
  Anomaly Detection & Threat Scoring
  Complete MITRE ATT&CK Framework
  Real-time Threat Detection
  Global Threat Intelligence
  Advanced Analytics & Reporting
  Comprehensive DNS Security Checks"""

        ctk.CTkLabel(
            desc_frame,
            text=desc_text,
            font=ctk.CTkFont(size=13),
            justify="left",
            text_color=COLORS['text_secondary']
        ).pack(padx=32, pady=24)

        info_frame = ctk.CTkFrame(content_frame, fg_color=COLORS['bg_secondary'], corner_radius=12,
                                   border_width=1, border_color=COLORS['border'])
        info_frame.pack(fill="x", padx=40)

        ml_models = 0
        if self.analyzer and hasattr(self.analyzer, 'ml_engine') and hasattr(self.analyzer.ml_engine, 'models'):
            ml_models = len(self.analyzer.ml_engine.models)

        ml_status = 'Enabled' if self.config.enable_ml else 'Disabled'

        system_info = f"""System Information:
  Platform: {platform.system()} {platform.release()}
  Python: {sys.version.split()[0]}
  ML Models: {ml_models}
  Device: {str(DEVICE) if TORCH_AVAILABLE else 'CPU'}
  ML Status: {ml_status}"""

        ctk.CTkLabel(
            info_frame,
            text=system_info,
            font=ctk.CTkFont(size=12),
            text_color=COLORS['text_secondary'],
            justify="left"
        ).pack(padx=24, pady=16)

        return page

    def update_mitre_page(self):
        """Update MITRE page after analyzer initialization"""
        if not hasattr(self, 'mitre_techniques_frame') or not self.analyzer:
            return

        # Clear loading message
        for widget in self.mitre_techniques_frame.winfo_children():
            widget.destroy()

        if hasattr(self.analyzer, 'mitre'):
            # Group techniques by tactic
            tactics_dict = {}
            for tech_id, details in self.analyzer.mitre.techniques.items():
                tactic = details.get('tactic', 'Unknown')
                if tactic not in tactics_dict:
                    tactics_dict[tactic] = []
                tactics_dict[tactic].append((tech_id, details))

            for i, (tactic, techniques) in enumerate(tactics_dict.items()):
                card = ctk.CTkFrame(self.mitre_techniques_frame, fg_color=COLORS['bg_secondary'],
                                     corner_radius=12, border_width=1, border_color=COLORS['border'])
                card.grid(row=i // 3, column=i % 3, padx=6, pady=6, sticky="nsew")

                header = ctk.CTkFrame(card, fg_color=COLORS['bg_tertiary'], corner_radius=8)
                header.pack(fill="x", padx=12, pady=(12, 8))

                ctk.CTkLabel(
                    header,
                    text=tactic,
                    font=ctk.CTkFont(size=15, weight="bold"),
                    text_color=COLORS['accent_primary']
                ).pack(pady=8)

                tech_frame = ctk.CTkFrame(card, fg_color="transparent")
                tech_frame.pack(fill="both", expand=True, padx=12, pady=(0, 12))

                for tech_id, details in techniques[:5]:
                    tech_item = ctk.CTkFrame(tech_frame, fg_color=COLORS['bg_quaternary'], corner_radius=6)
                    tech_item.pack(fill="x", pady=2)

                    tech_content = ctk.CTkFrame(tech_item, fg_color="transparent")
                    tech_content.pack(fill="x", padx=10, pady=6)

                    ctk.CTkLabel(
                        tech_content,
                        text=f"{tech_id}: {details.get('name', 'Unknown')}",
                        font=ctk.CTkFont(size=11, weight="bold"),
                        anchor="w"
                    ).pack(fill="x")

                    severity_color = {
                        'critical': COLORS['danger'],
                        'high': COLORS['warning'],
                        'medium': COLORS['info'],
                        'low': COLORS['success']
                    }.get(details.get('severity', 'low'), COLORS['text_secondary'])

                    ctk.CTkLabel(
                        tech_content,
                        text=f"Severity: {details.get('severity', 'unknown').upper()}",
                        font=ctk.CTkFont(size=10),
                        text_color=severity_color,
                        anchor="w"
                    ).pack(fill="x")

                self.mitre_techniques_frame.columnconfigure(i % 3, weight=1)
                self.mitre_techniques_frame.rowconfigure(i // 3, weight=1)

    def update_ml_page(self):
        """Update ML page after analyzer initialization"""
        if not self.analyzer or not hasattr(self.analyzer, 'ml_engine'):
            return

        if hasattr(self, 'ml_models_frame'):
            self.populate_ml_models(self.ml_models_frame)

    def update_threat_intel_page(self):
        """Update threat intel page after analyzer initialization"""
        if not self.analyzer or not hasattr(self.analyzer, 'threat_intel'):
            return

        if hasattr(self, 'threat_feeds_grid'):
            self.populate_threat_feeds(self.threat_feeds_grid)

    def show_page(self, page_name):
        """Show specific page with animation"""
        # Hide all pages
        for page in self.pages.values():
            page.pack_forget()

        # Show selected page
        if page_name in self.pages:
            self.pages[page_name].pack(fill="both", expand=True)

        # Update navigation highlighting
        nav_to_page = {
            'Dashboard': 'dashboard', 'Email Analysis': 'analysis',
            'Bulk Scanner': 'bulk', 'Real-time Monitor': 'monitor',
            'MITRE ATT&CK': 'mitre', 'Threat Intelligence': 'threat_intel',
            'ML Models': 'ml', 'DNS Security': 'dns',
            'Analytics': 'analytics', 'Export Reports': 'reports',
            'Audit Log': 'audit', 'Settings': 'settings', 'About': 'about',
        }
        for btn_name, btn in self.nav_buttons.items():
            if nav_to_page.get(btn_name) == page_name:
                btn.configure(fg_color=COLORS['sidebar_active'], text_color=COLORS['text_primary'])
            else:
                btn.configure(fg_color="transparent", text_color=COLORS['text_secondary'])

    # Navigation methods
    def show_dashboard(self):
        self.show_page('dashboard')
        self.refresh_dashboard()

    def show_analysis(self):
        self.show_page('analysis')

    def show_bulk(self):
        self.show_page('bulk')

    def show_monitor(self):
        self.show_page('monitor')
        # Start monitor on first visit (not at app startup)
        if not self.monitor_running:
            self.monitor_running = True
            # Only start a new thread if the old one is gone (prevents duplicate threads)
            if not hasattr(self, 'monitor_thread') or not self.monitor_thread.is_alive():
                self.start_monitor_simulation()
            if hasattr(self, 'monitor_status_label') and self.monitor_status_label.winfo_exists():
                self.monitor_status_label.configure(
                    text="● Monitor Active", text_color=COLORS['success']
                )
            # Reset pause button text when re-activating
            if hasattr(self, 'monitor_pause_btn') and self.monitor_pause_btn.winfo_exists():
                self.monitor_pause_btn.configure(text="Pause")

    def show_mitre(self):
        self.show_page('mitre')

    def show_threat_intel(self):
        self.show_page('threat_intel')
        # Update threat intel page if analyzer is ready
        if self.analyzer:
            self.update_threat_intel_page()

    def show_ml(self):
        self.show_page('ml')
        # Update ML page if analyzer is ready
        if self.analyzer:
            self.update_ml_page()

    def show_dns(self):
        self.show_page('dns')

    def show_analytics(self):
        self.show_page('analytics')
        try:
            self.update_analytics_charts()
        except Exception as e:
            logger.debug(f"Could not refresh analytics on navigation: {e}")

    def show_reports(self):
        self.show_page('reports')

    def show_settings(self):
        self.show_page('settings')

    def show_audit(self):
        self.show_page('audit')

    def show_about(self):
        self.show_page('about')

    def initialize_analyzer(self):
        """Initialize analyzer components in background"""

        def init():
            self.ui(self.update_status, "Initializing security analyzer...")
            self.ui(self.status_progress.set_progress, 0.2)

            analyzer = EmailSecurityAnalyzer(self.config)
            self.ui(setattr, self, 'analyzer', analyzer)
            self.ui(self.status_progress.set_progress, 0.6)

            bulk = BulkProcessingEngine(analyzer, self.config)
            self.ui(setattr, self, 'bulk_processor', bulk)
            self.ui(self.status_progress.set_progress, 0.8)

            # Update ML status
            def post_ml_status():
                if self.config.enable_ml and analyzer.ml_engine and analyzer.ml_engine.is_initialized:
                    self.update_status(f"Ready - {len(analyzer.ml_engine.models)} ML models active")
                else:
                    status_msg = "Ready"
                    if not self.config.enable_ml:
                        status_msg += " - ML disabled"
                    self.update_status(status_msg)
            self.ui(post_ml_status)

            self.ui(self.status_progress.set_progress, 1.0)

            # Update header stats and pages
            self.ui(self.update_header_stats)
            self.ui(self.update_mitre_page)
            self.ui(self.update_ml_page)
            self.ui(self.update_threat_intel_page)

            # Log initialization
            self.ui(self.log_activity, "System initialized successfully")

            # Hide progress after delay
            def hide_progress():
                try:
                    if hasattr(self, 'status_progress') and self.status_progress.winfo_exists():
                        self.status_progress.set_progress(0)
                except Exception:
                    pass
            self.ui(lambda: self.root.after(2000, lambda: self.ui(hide_progress)))

        threading.Thread(target=init, daemon=True).start()

    def stop_single_analysis(self):
        """Stop the current single email analysis"""
        self._single_cancel.set()
        self.update_status("Stopping analysis...")

    def analyze_email(self):
        """Analyze single email with beautiful visualization"""
        email = self.email_entry.get().strip()
        password = self.password_entry.get().strip() if hasattr(self, 'password_entry') else ""
        if not email:
            messagebox.showwarning("Warning", "Please enter an email address")
            return

        if not self.analyzer:
            messagebox.showwarning("Warning", "Analyzer not ready yet")
            return

        # Clear cancel flag
        self._single_cancel.clear()

        # Clear previous results
        for widget in self.results_column.winfo_children():
            widget.destroy()

        # Toggle buttons: disable Analyze, enable Stop
        self.analyze_btn.configure(state="disabled")
        self.stop_analyze_btn.configure(state="normal")

        # Show progress bar
        self.analysis_progress_bar.pack(fill="x", pady=(0, 10))
        self.analysis_progress_bar.set_progress(0)
        self.analysis_progress.pack(pady=10)
        self.analysis_progress.set_progress(0)

        # Update status
        self.update_status(f"Analyzing {email}...")

        # Read Tkinter variables on main thread BEFORE spawning background thread
        threat_intel_enabled = bool(self.threat_intel_var.get()) if hasattr(self, "threat_intel_var") else True
        deep_scan_enabled = (bool(self.deep_scan_var.get()) if hasattr(self, "deep_scan_var") else True) and threat_intel_enabled
        ml_enabled = (bool(self.ml_analysis_var.get()) if hasattr(self, "ml_analysis_var") else False) and bool(self.config.enable_ml)

        def analyze_thread():
            try:
                if self._single_cancel.is_set():
                    return

                # Start a smooth progress animation thread
                analysis_done = threading.Event()
                progress_steps = [
                    (0.05, "Starting analysis..."),
                    (0.15, "Checking domain reputation..."),
                    (0.30, "Running DNS security checks..."),
                    (0.45, "Checking breach databases..."),
                    (0.60, "Analyzing threats..."),
                    (0.75, "Running ML predictions..."),
                    (0.85, "Mapping MITRE ATT&CK..."),
                    (0.92, "Generating recommendations..."),
                ]

                def progress_animator():
                    for prog, msg in progress_steps:
                        if analysis_done.is_set() or self._single_cancel.is_set():
                            return
                        self.ui(self.analysis_progress_bar.set_progress, prog)
                        self.ui(self.analysis_progress.set_progress, prog)
                        self.ui(self.update_status, msg)
                        time.sleep(0.8)

                progress_thread = threading.Thread(target=progress_animator, daemon=True)
                progress_thread.start()

                # Perform actual analysis
                result = self.analyzer.analyze_email(
                    email,
                    full_analysis=deep_scan_enabled,
                    enable_ml=ml_enabled,
                    enable_threat_intel=threat_intel_enabled,
                    password=password if password else None
                )

                # Stop progress animation
                analysis_done.set()

                if self._single_cancel.is_set():
                    self.ui(self.update_status, "Analysis cancelled")
                    return

                # Complete
                self.ui(self.analysis_progress_bar.set_progress, 1.0)
                self.ui(self.analysis_progress.set_progress, 1.0)
                self.ui(self.update_status, "Analysis complete!")

                # Debug: Log breach info
                if 'breach_info' in result:
                    logger.info(f"Breach data found: {result['breach_info'].get('found', False)}, Count: {result['breach_info'].get('count', 0)}")
                else:
                    logger.warning("No breach_info in result!")

                # Update UI
                self.ui(self.display_analysis_results, result)

                # Log activity
                self.ui(self.log_activity, f"Analyzed: {email} - Risk: {result.get('risk_level', 'unknown')}")

                # Refresh ML page to show predictions
                if hasattr(self, 'ml_models_frame'):
                    def refresh_ml():
                        time.sleep(0.1)
                        self.ui(self.populate_ml_models, self.ml_models_frame)
                    threading.Thread(target=refresh_ml, daemon=True).start()

            except Exception as e:
                error_msg = str(e)
                self.ui(lambda msg=error_msg: messagebox.showerror("Error", f"Analysis failed: {msg}"))
                logger.error(f"Analysis error: {e}")
            finally:
                self.ui(self.reset_analysis_ui)

        threading.Thread(target=analyze_thread, daemon=True).start()

    def display_analysis_results(self, result):
        """Display analysis results with beautiful formatting"""
        # Track result in UI thread (thread-safe)
        try:
            self.current_results.append(result)
            # Cap results to prevent unbounded memory growth
            if len(self.current_results) > 5000:
                self.current_results = self.current_results[-5000:]
        except Exception as e:
            logger.debug(f"Could not append result: {e}")

        # Clear loading message
        for widget in self.results_column.winfo_children():
            widget.destroy()

        # Results header
        header_frame = ctk.CTkFrame(self.results_column, fg_color="transparent")
        header_frame.pack(fill="x", padx=30, pady=(30, 20))

        ctk.CTkLabel(
            header_frame,
            text="Analysis Results",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color=COLORS['text_primary']
        ).pack(side="left")

        # Email address
        ctk.CTkLabel(
            header_frame,
            text=result.get('email', 'N/A'),
            font=ctk.CTkFont(size=16),
            text_color=COLORS['text_secondary']
        ).pack(side="left", padx=(20, 0))

        # Risk score card
        risk_card = ctk.CTkFrame(self.results_column, fg_color=COLORS['bg_tertiary'], corner_radius=12, border_width=1, border_color=COLORS['border'])
        risk_card.pack(fill="x", padx=30, pady=10)

        risk_content = ctk.CTkFrame(risk_card, fg_color="transparent")
        risk_content.pack(padx=30, pady=30)

        # Risk level indicator
        risk_level = result.get('risk_level', 'unknown')
        risk_score = result.get('risk_score', 0)
        risk_color = {
            'critical': COLORS['danger'],
            'high': COLORS['warning'],
            'medium': '#d29922',
            'low': COLORS['success'],
            'minimal': COLORS['accent_secondary']
        }.get(risk_level, COLORS['text_primary'])

        # Risk score display
        score_frame = ctk.CTkFrame(risk_content, fg_color="transparent")
        score_frame.pack()

        ctk.CTkLabel(
            score_frame,
            text=str(risk_score),
            font=ctk.CTkFont(size=72, weight="bold"),
            text_color=risk_color
        ).pack(side="left")

        ctk.CTkLabel(
            score_frame,
            text="/100",
            font=ctk.CTkFont(size=36),
            text_color=COLORS['text_secondary']
        ).pack(side="left", padx=(5, 0))

        # Risk level
        ctk.CTkLabel(
            risk_content,
            text=f"Risk Level: {risk_level.upper()}",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color=risk_color
        ).pack(pady=(20, 0))

        # ML Feedback buttons — train the model with real data
        feedback_frame = ctk.CTkFrame(risk_content, fg_color="transparent")
        feedback_frame.pack(pady=(15, 0))

        ctk.CTkLabel(
            feedback_frame,
            text="Is this assessment correct?",
            font=ctk.CTkFont(size=12),
            text_color=COLORS['text_secondary']
        ).pack(side="left", padx=(0, 10))

        email_for_feedback = result.get('email', '')
        features_for_feedback = result.get('_ml_features')  # Per-result features (avoids race)
        ensemble_for_feedback = result.get('_ml_ensemble_score', 0.5)  # Per-result score (avoids stale history)

        def on_feedback_safe():
            if self.analyzer:
                self.analyzer.submit_feedback(email_for_feedback, 0, features=features_for_feedback, ensemble_score=ensemble_for_feedback)
                feedback_label.configure(text="Marked as SAFE — thanks!", text_color=COLORS['success'])

        def on_feedback_risky():
            if self.analyzer:
                self.analyzer.submit_feedback(email_for_feedback, 1, features=features_for_feedback, ensemble_score=ensemble_for_feedback)
                feedback_label.configure(text="Marked as RISKY — thanks!", text_color=COLORS['danger'])

        safe_btn = ctk.CTkButton(
            feedback_frame, text="Safe", width=70, height=28,
            fg_color=COLORS['success'], hover_color="#27ae60",
            command=on_feedback_safe
        )
        safe_btn.pack(side="left", padx=3)

        risky_btn = ctk.CTkButton(
            feedback_frame, text="Risky", width=70, height=28,
            fg_color=COLORS['danger'], hover_color="#c0392b",
            command=on_feedback_risky
        )
        risky_btn.pack(side="left", padx=3)

        feedback_label = ctk.CTkLabel(
            risk_content, text="",
            font=ctk.CTkFont(size=11),
            text_color=COLORS['text_secondary']
        )
        feedback_label.pack(pady=(5, 0))

        # Threats section
        if result.get('threats'):
            threats_frame = ctk.CTkFrame(self.results_column, fg_color=COLORS['bg_tertiary'], corner_radius=12, border_width=1, border_color=COLORS['border'])
            threats_frame.pack(fill="x", padx=30, pady=10)

            threats_content = ctk.CTkFrame(threats_frame, fg_color="transparent")
            threats_content.pack(padx=30, pady=20)

            ctk.CTkLabel(
                threats_content,
                text=f"🚨 {len(result['threats'])} Threat(s) Detected",
                font=ctk.CTkFont(size=16, weight="bold"),
                text_color=COLORS['danger']
            ).pack(pady=(0, 15))

            for threat in result['threats']:
                threat_item = ctk.CTkFrame(threats_content, fg_color=COLORS['bg_quaternary'], corner_radius=8)
                threat_item.pack(fill="x", pady=5)

                threat_text = ctk.CTkFrame(threat_item, fg_color="transparent")
                threat_text.pack(fill="x", padx=15, pady=10)

                severity_color = {
                    'critical': COLORS['danger'],
                    'high': COLORS['warning'],
                    'medium': COLORS['info'],
                    'low': COLORS['text_secondary']
                }.get(threat.get('severity', 'low'), COLORS['text_secondary'])

                ctk.CTkLabel(
                    threat_text,
                    text=f"{threat.get('type', '').replace('_', ' ').title()}",
                    font=ctk.CTkFont(size=14, weight="bold"),
                    text_color=severity_color,
                    anchor="w"
                ).pack(fill="x")

                ctk.CTkLabel(
                    threat_text,
                    text=threat.get('description', 'No description available'),
                    font=ctk.CTkFont(size=12),
                    text_color=COLORS['text_secondary'],
                    anchor="w",
                    wraplength=400
                ).pack(fill="x")

        # ML Predictions
        if result.get('ml_predictions') and self.config.enable_ml:
            ml_frame = ctk.CTkFrame(self.results_column, fg_color=COLORS['bg_tertiary'], corner_radius=12, border_width=1, border_color=COLORS['border'])
            ml_frame.pack(fill="x", padx=30, pady=10)

            ml_content = ctk.CTkFrame(ml_frame, fg_color="transparent")
            ml_content.pack(padx=30, pady=20)

            ctk.CTkLabel(
                ml_content,
                text="🤖 Machine Learning Analysis",
                font=ctk.CTkFont(size=16, weight="bold")
            ).pack(pady=(0, 15))

            # Show top ML predictions (filter out non-numeric keys)
            skip_keys = {'ensemble', 'is_malicious', 'precision_threshold', 'anomaly_score'}
            numeric_preds = {k: v for k, v in result['ml_predictions'].items()
                            if k not in skip_keys and isinstance(v, (int, float))}
            predictions = sorted(numeric_preds.items(),
                                 key=lambda x: x[1], reverse=True)[:5]

            for model_name, score in predictions:

                pred_frame = ctk.CTkFrame(ml_content, fg_color="transparent")
                pred_frame.pack(fill="x", pady=5)

                ctk.CTkLabel(
                    pred_frame,
                    text=model_name.replace('_', ' ').title(),
                    font=ctk.CTkFont(size=12),
                    anchor="w"
                ).pack(side="left", fill="x", expand=True)

                # Progress bar for prediction
                prog = ctk.CTkProgressBar(pred_frame, width=200, height=15)
                prog.pack(side="right", padx=(10, 0))
                prog.set(score)

                # Score label
                score_color = COLORS['danger'] if score > 0.7 else COLORS['warning'] if score > 0.5 else COLORS[
                    'success']
                ctk.CTkLabel(
                    pred_frame,
                    text=f"{score:.1%}",
                    font=ctk.CTkFont(size=12),
                    text_color=score_color,
                    width=50
                ).pack(side="right")

        # DNS Security
        if result.get('dns_security'):
            dns_frame = ctk.CTkFrame(self.results_column, fg_color=COLORS['bg_tertiary'], corner_radius=12, border_width=1, border_color=COLORS['border'])
            dns_frame.pack(fill="x", padx=30, pady=10)

            dns_content = ctk.CTkFrame(dns_frame, fg_color="transparent")
            dns_content.pack(padx=30, pady=20)

            ctk.CTkLabel(
                dns_content,
                text="🔒 DNS Security Status",
                font=ctk.CTkFont(size=16, weight="bold")
            ).pack(pady=(0, 15))

            dns_items_row1 = [
                ('SPF', result['dns_security'].get('spf', False)),
                ('DMARC', result['dns_security'].get('dmarc', False)),
                ('DKIM', result['dns_security'].get('dkim', False)),
                ('DNSSEC', result['dns_security'].get('dnssec', False))
            ]
            dns_items_row2 = [
                ('BIMI', result['dns_security'].get('bimi', False)),
                ('MTA-STS', result['dns_security'].get('mta_sts', False)),
                ('TLS-RPT', result['dns_security'].get('tls_rpt', False))
            ]

            dns_grid = ctk.CTkFrame(dns_content, fg_color="transparent")
            dns_grid.pack()

            for row_idx, items in enumerate([dns_items_row1, dns_items_row2]):
                for i, (label, status) in enumerate(items):
                    item_frame = ctk.CTkFrame(dns_grid, fg_color=COLORS['bg_quaternary'],
                                              corner_radius=8, width=120, height=80)
                    item_frame.grid(row=row_idx, column=i, padx=5, pady=5)
                    item_frame.pack_propagate(False)

                    status_icon = "✅" if status else "❌"
                    status_color = COLORS['success'] if status else COLORS['danger']

                    ctk.CTkLabel(
                        item_frame,
                        text=status_icon,
                        font=ctk.CTkFont(size=24)
                    ).pack(pady=(10, 5))

                    ctk.CTkLabel(
                        item_frame,
                        text=label,
                        font=ctk.CTkFont(size=12),
                        text_color=status_color
                    ).pack()

        # Advanced Security Checks — show section when checks ran (keys exist), not just when findings exist
        has_advanced = any([
            'dnsbl' in result, 'cert_transparency' in result,
            'gravatar' in result, 'threatfox' in result,
            'parked_domain' in result, 'dga_analysis' in result,
        ])
        if has_advanced:
            adv_frame = ctk.CTkFrame(self.results_column, fg_color=COLORS['bg_tertiary'], corner_radius=12, border_width=1, border_color=COLORS['border'])
            adv_frame.pack(fill="x", padx=30, pady=10)

            adv_content = ctk.CTkFrame(adv_frame, fg_color="transparent")
            adv_content.pack(padx=30, pady=20)

            ctk.CTkLabel(
                adv_content,
                text="🛡️ Advanced Security Checks",
                font=ctk.CTkFont(size=16, weight="bold")
            ).pack(pady=(0, 15))

            adv_grid = ctk.CTkFrame(adv_content, fg_color="transparent")
            adv_grid.pack(fill="x")

            # Build items list dynamically
            adv_items = []

            # DNSBL
            dnsbl = result.get('dnsbl') or {}
            if dnsbl:
                is_clean = not dnsbl.get('listed', False)
                listed_count = dnsbl.get('listed_count', 0)
                adv_items.append((
                    'DNSBL',
                    '✅ Clean' if is_clean else f'🚫 Listed ({listed_count})',
                    COLORS['success'] if is_clean else COLORS['danger']
                ))

            # Certificate Transparency
            ct = result.get('cert_transparency') or {}
            if ct:
                cert_count = ct.get('cert_count', 0)
                adv_items.append((
                    'Certificates',
                    f'📜 {cert_count} cert(s)' if ct.get('found') else '❓ None found',
                    COLORS['text_primary'] if ct.get('found') else COLORS['text_secondary']
                ))

            # DGA Detection
            dga = result.get('dga_analysis') or {}
            if dga:
                is_clean = not dga.get('is_dga', False)
                dga_score = float(dga.get('dga_score') or 0.0)
                adv_items.append((
                    'DGA Check',
                    '✅ Normal' if is_clean else f'⚠️ DGA ({dga_score:.0%})',
                    COLORS['success'] if is_clean else COLORS['warning']
                ))

            # Gravatar
            gravatar = result.get('gravatar') or {}
            if gravatar:
                has_profile = gravatar.get('has_profile', False)
                adv_items.append((
                    'Gravatar',
                    '👤 Profile found' if has_profile else '👻 No profile',
                    COLORS['success'] if has_profile else COLORS['text_secondary']
                ))

            # ThreatFox
            threatfox = result.get('threatfox') or {}
            if threatfox:
                is_clean = not threatfox.get('found', False)
                ioc_count = threatfox.get('ioc_count', 0)
                adv_items.append((
                    'ThreatFox',
                    '✅ Clean' if is_clean else f'☠️ {ioc_count} IOC(s)',
                    COLORS['success'] if is_clean else COLORS['danger']
                ))

            # Parked Domain
            parked = result.get('parked_domain') or {}
            if parked:
                is_parked = parked.get('is_parked', False)
                adv_items.append((
                    'Domain Status',
                    '🅿️ Parked' if is_parked else '✅ Active',
                    COLORS['warning'] if is_parked else COLORS['success']
                ))

            # Render items in a grid (3 columns per row)
            for idx, (label, value, color) in enumerate(adv_items):
                row = idx // 3
                col = idx % 3

                item_frame = ctk.CTkFrame(adv_grid, fg_color=COLORS['bg_quaternary'],
                                          corner_radius=8, width=180, height=70)
                item_frame.grid(row=row, column=col, padx=5, pady=5, sticky="ew")
                item_frame.pack_propagate(False)

                ctk.CTkLabel(
                    item_frame,
                    text=label,
                    font=ctk.CTkFont(size=11),
                    text_color=COLORS['text_secondary']
                ).pack(pady=(8, 2))

                ctk.CTkLabel(
                    item_frame,
                    text=value,
                    font=ctk.CTkFont(size=13, weight="bold"),
                    text_color=color
                ).pack()

            # Configure grid columns to be even
            for c in range(3):
                adv_grid.columnconfigure(c, weight=1)

        # Breach Information (Enhanced) - Only show full section when breaches found
        breach_data = result.get('breach_info') or {}
        if breach_data.get('found'):
            breach_frame = ctk.CTkFrame(self.results_column, fg_color=COLORS['bg_tertiary'], corner_radius=12, border_width=1, border_color=COLORS['border'])
            breach_frame.pack(fill="x", padx=30, pady=10)

            breach_content = ctk.CTkFrame(breach_frame, fg_color="transparent")
            breach_content.pack(padx=30, pady=20)

            # Header
            severity_color = {
                'critical': COLORS['danger'],
                'high': COLORS['warning'],
                'medium': '#d29922',
                'low': COLORS['success']
            }.get(breach_data.get('severity', 'low'), COLORS['text_secondary'])

            ctk.CTkLabel(
                breach_content,
                text="[ALERT] Data Breach Analysis",
                font=ctk.CTkFont(size=16, weight="bold"),
                text_color=severity_color
            ).pack(pady=(0, 15))

            # Breach summary
            summary_frame = ctk.CTkFrame(breach_content, fg_color=COLORS['bg_quaternary'], corner_radius=10)
            summary_frame.pack(fill="x", pady=10)

            summary_content = ctk.CTkFrame(summary_frame, fg_color="transparent")
            summary_content.pack(padx=20, pady=15)

            ctk.CTkLabel(
                summary_content,
                text=f"This email appears in {breach_data.get('count', 0)} known data breach(es)",
                font=ctk.CTkFont(size=14, weight="bold"),
                text_color=COLORS['danger']
            ).pack(anchor="w")

            # Breach details
            if breach_data.get('details'):
                details_frame = ctk.CTkFrame(breach_content, fg_color=COLORS['bg_quaternary'], corner_radius=10)
                details_frame.pack(fill="both", expand=True, pady=10)

                # Scrollable frame for breach details
                details_scroll = ctk.CTkScrollableFrame(
                    details_frame,
                    fg_color="transparent",
                    height=200
                )
                details_scroll.pack(fill="both", expand=True, padx=15, pady=15)

                ctk.CTkLabel(
                    details_scroll,
                    text="Breach Details:",
                    font=ctk.CTkFont(size=13, weight="bold"),
                    text_color=COLORS['text_primary']
                ).pack(anchor="w", pady=(0, 10))

                for breach in breach_data.get('details', []):
                    if isinstance(breach, dict) and not breach.get('error'):
                        breach_item = ctk.CTkFrame(details_scroll, fg_color=COLORS['bg_secondary'], corner_radius=8)
                        breach_item.pack(fill="x", pady=5)

                        breach_item_content = ctk.CTkFrame(breach_item, fg_color="transparent")
                        breach_item_content.pack(fill="x", padx=15, pady=10)

                        # Breach name and date
                        ctk.CTkLabel(
                            breach_item_content,
                            text=f"{breach.get('title', breach.get('name', 'Unknown'))}",
                            font=ctk.CTkFont(size=13, weight="bold"),
                            text_color=COLORS['danger']
                        ).pack(anchor="w")

                        if breach.get('domain'):
                            ctk.CTkLabel(
                                breach_item_content,
                                text=f"Domain: {breach['domain']}",
                                font=ctk.CTkFont(size=11),
                                text_color=COLORS['text_secondary']
                            ).pack(anchor="w", pady=(2, 0))

                        if breach.get('breach_date'):
                            ctk.CTkLabel(
                                breach_item_content,
                                text=f"Date: {breach['breach_date']}",
                                font=ctk.CTkFont(size=11),
                                text_color=COLORS['text_secondary']
                            ).pack(anchor="w", pady=(2, 0))

                        if breach.get('description'):
                            ctk.CTkLabel(
                                breach_item_content,
                                text=f"Details: {breach['description']}",
                                font=ctk.CTkFont(size=11),
                                text_color=COLORS['text_secondary'],
                                wraplength=400,
                                justify="left"
                            ).pack(anchor="w", pady=(5, 0))

                        if isinstance(breach.get('pwn_count'), (int, float)) and breach['pwn_count'] > 0:
                            ctk.CTkLabel(
                                breach_item_content,
                                text=f"Affected accounts: {int(breach['pwn_count']):,}",
                                font=ctk.CTkFont(size=11),
                                text_color=COLORS['text_secondary']
                            ).pack(anchor="w", pady=(2, 0))

                        if breach.get('data_classes'):
                            data_text = ", ".join(str(dc) for dc in breach['data_classes'][:5])
                            if len(breach['data_classes']) > 5:
                                data_text += f" (+{len(breach['data_classes'])-5} more)"
                            ctk.CTkLabel(
                                breach_item_content,
                                text=f"Compromised data: {data_text}",
                                font=ctk.CTkFont(size=11),
                                text_color=COLORS['warning'],
                                wraplength=400
                            ).pack(anchor="w", pady=(2, 0))

            # Mitigation steps
            if breach_data.get('mitigation_steps'):
                mitigation_frame = ctk.CTkFrame(breach_content, fg_color=COLORS['bg_quaternary'], corner_radius=10)
                mitigation_frame.pack(fill="both", expand=True, pady=10)

                mitigation_scroll = ctk.CTkScrollableFrame(
                    mitigation_frame,
                    fg_color="transparent",
                    height=250
                )
                mitigation_scroll.pack(fill="both", expand=True, padx=15, pady=15)

                ctk.CTkLabel(
                    mitigation_scroll,
                    text="[ACTION REQUIRED] Security Mitigation Steps:",
                    font=ctk.CTkFont(size=14, weight="bold"),
                    text_color=COLORS['danger']
                ).pack(anchor="w", pady=(0, 10))

                for step in breach_data['mitigation_steps']:
                    step_label = ctk.CTkLabel(
                        mitigation_scroll,
                        text=step,
                        font=ctk.CTkFont(size=11),
                        text_color=COLORS['text_primary'],
                        anchor="w",
                        wraplength=500,
                        justify="left"
                    )
                    step_label.pack(anchor="w", fill="x", pady=2)

        # Password Breach Information
        password_breach = result.get('password_breach') or {}
        if password_breach and password_breach.get('details') and 'No password' not in password_breach.get('details', ''):
            pw_frame = ctk.CTkFrame(self.results_column, fg_color=COLORS['bg_tertiary'], corner_radius=12, border_width=1, border_color=COLORS['border'])
            pw_frame.pack(fill="x", padx=30, pady=10)

            pw_content = ctk.CTkFrame(pw_frame, fg_color="transparent")
            pw_content.pack(padx=30, pady=20)

            if password_breach.get('found'):
                pw_header_color = COLORS['danger']
                pw_header_text = "[ALERT] Password Found in Dark Web Breaches"
            else:
                pw_header_color = COLORS['success']
                pw_header_text = "[SAFE] Password Not Found in Breaches"

            ctk.CTkLabel(
                pw_content,
                text=pw_header_text,
                font=ctk.CTkFont(size=16, weight="bold"),
                text_color=pw_header_color
            ).pack(pady=(0, 15))

            # Privacy note
            ctk.CTkLabel(
                pw_content,
                text=f"Method: {password_breach.get('hash_algorithm', 'Keccak-512')} + k-anonymity | {password_breach.get('privacy_note', '')}",
                font=ctk.CTkFont(size=10),
                text_color=COLORS['text_tertiary']
            ).pack(pady=(0, 10))

            # Details
            details_box = ctk.CTkFrame(pw_content, fg_color=COLORS['bg_quaternary'], corner_radius=10)
            details_box.pack(fill="x", pady=5)

            details_inner = ctk.CTkFrame(details_box, fg_color="transparent")
            details_inner.pack(padx=20, pady=15)

            ctk.CTkLabel(
                details_inner,
                text=password_breach.get('details', ''),
                font=ctk.CTkFont(size=13),
                text_color=pw_header_color,
                wraplength=500,
                justify="left"
            ).pack(anchor="w")

            # Recommendation
            if password_breach.get('recommendation'):
                rec_box = ctk.CTkFrame(pw_content, fg_color=COLORS['bg_quaternary'], corner_radius=10)
                rec_box.pack(fill="x", pady=5)

                rec_inner = ctk.CTkFrame(rec_box, fg_color="transparent")
                rec_inner.pack(padx=20, pady=15)

                ctk.CTkLabel(
                    rec_inner,
                    text="Recommendation:",
                    font=ctk.CTkFont(size=12, weight="bold"),
                    text_color=COLORS['text_primary']
                ).pack(anchor="w", pady=(0, 5))

                ctk.CTkLabel(
                    rec_inner,
                    text=password_breach.get('recommendation', ''),
                    font=ctk.CTkFont(size=12),
                    text_color=COLORS['text_secondary'],
                    wraplength=500,
                    justify="left"
                ).pack(anchor="w")

            # Source attribution
            ctk.CTkLabel(
                pw_content,
                text=f"Source: {password_breach.get('source', 'XposedOrNot')} (free dark web monitoring)",
                font=ctk.CTkFont(size=10),
                text_color=COLORS['text_tertiary']
            ).pack(pady=(10, 0))

        # Recommendations
        if result.get('recommendations'):
            rec_frame = ctk.CTkFrame(self.results_column, fg_color=COLORS['bg_tertiary'], corner_radius=12, border_width=1, border_color=COLORS['border'])
            rec_frame.pack(fill="x", padx=30, pady=10)

            rec_content = ctk.CTkFrame(rec_frame, fg_color="transparent")
            rec_content.pack(padx=30, pady=20)

            ctk.CTkLabel(
                rec_content,
                text="💡 Security Recommendations",
                font=ctk.CTkFont(size=16, weight="bold")
            ).pack(pady=(0, 15))

            for rec in result['recommendations'][:5]:
                rec_item = ctk.CTkFrame(rec_content, fg_color="transparent")
                rec_item.pack(fill="x", pady=3)

                ctk.CTkLabel(
                    rec_item,
                    text=f"• {rec}",
                    font=ctk.CTkFont(size=13),
                    text_color=COLORS['text_primary'],
                    anchor="w",
                    wraplength=450
                ).pack(fill="x")

        # Export button
        export_btn = GradientButton(
            self.results_column,
            text="📄 Export Report",
            command=lambda: self.export_single_report(result),
            width=200,
            height=40,
            gradient_colors=[COLORS['accent_primary'], COLORS['accent_secondary']]
        )
        export_btn.pack(pady=20)

        # Update dashboard
        self.update_header_stats()
        self.refresh_dashboard()

    def reset_analysis_ui(self):
        """Reset analysis UI after completion"""
        try:
            if hasattr(self, 'analysis_progress') and self.analysis_progress.winfo_exists():
                self.analysis_progress.pack_forget()
            if hasattr(self, 'analysis_progress_bar') and self.analysis_progress_bar.winfo_exists():
                self.analysis_progress_bar.pack_forget()
            if hasattr(self, 'analyze_btn') and self.analyze_btn.winfo_exists():
                self.analyze_btn.configure(state="normal")
            if hasattr(self, 'stop_analyze_btn') and self.stop_analyze_btn.winfo_exists():
                self.stop_analyze_btn.configure(state="disabled")
            self.update_status("Ready")
        except Exception:
            pass  # Window may have been destroyed

        # Clear password field after analysis for security
        if hasattr(self, 'password_entry'):
            try:
                if self.password_entry.winfo_exists():
                    self.password_entry.delete(0, 'end')
            except Exception:
                pass

        # Update analytics charts only if analytics page is visible
        if hasattr(self, 'heatmap_ax'):
            try:
                if hasattr(self, 'pages') and self.pages.get('analytics') and \
                        self.pages['analytics'].winfo_ismapped():
                    self.update_analytics_charts()
            except Exception as e:
                logger.debug(f"Could not update analytics charts: {e}")

    def select_bulk_file(self):
        """Select file for bulk analysis"""
        filename = filedialog.askopenfilename(
            title="Select Email List",
            filetypes=[
                ("Text files", "*.txt"),
                ("CSV files", "*.csv"),
                ("All files", "*.*")
            ]
        )

        if filename:
            self.bulk_file = filename
            self.file_label.configure(text=f"Selected: {os.path.basename(filename)}")

            # Count emails in file
            try:
                with open(filename, 'r', encoding='utf-8', errors='replace') as f:
                    emails = [line.strip() for line in f if '@' in line]
                    self.bulk_email_count = len(emails)
                    self.bulk_progress_label.configure(
                        text=f"Ready to process {self.bulk_email_count} emails"
                    )
            except Exception:
                self.bulk_email_count = 0

    def stop_bulk_analysis(self):
        """Stop the current bulk analysis"""
        if hasattr(self, 'bulk_processor') and self.bulk_processor:
            self.bulk_processor.cancel()
        if hasattr(self, 'bulk_progress_label') and self.bulk_progress_label.winfo_exists():
            self.bulk_progress_label.configure(text="Stopping...")
        self.update_status("Stopping bulk analysis...")

    def process_bulk(self):
        """Process bulk email analysis"""
        if not hasattr(self, 'bulk_file') or not self.bulk_file:
            messagebox.showwarning("Warning", "Please select a file first")
            return

        if not self.analyzer:
            messagebox.showwarning("Warning", "Analyzer is still initializing. Please wait a moment and try again.")
            return

        if not self.bulk_processor:
            messagebox.showwarning("Warning", "Bulk processor is still initializing. Please wait a moment and try again.")
            return

        # Load emails
        try:
            with open(self.bulk_file, 'r', encoding='utf-8', errors='replace') as f:
                emails = [line.strip() for line in f if '@' in line.strip()]
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read file: {str(e)}")
            return

        if not emails:
            messagebox.showwarning("Warning", "No valid email addresses found in file")
            return

        logger.info(f"Starting bulk scan: {len(emails)} emails from {self.bulk_file}")

        # Toggle buttons: disable Start, enable Stop
        self.process_btn.configure(state="disabled")
        self.stop_bulk_btn.configure(state="normal")
        self.bulk_progress_label.configure(text=f"Starting scan of {len(emails)} emails...")
        self.bulk_progress_bar.set_progress(0.01)

        def progress_callback(current, total):
            progress = current / total if total else 0
            logger.info(f"Bulk progress update: {current}/{total} = {progress*100:.1f}%")
            self.ui(self.bulk_progress_bar.set_progress, progress)
            self.ui(self.bulk_progress_label.configure, text=f"Processing: {current}/{total} emails ({int(progress*100)}%)")

        def process_thread():
            try:
                logger.info("Bulk process thread started")
                result = self.bulk_processor.process_email_list(emails, progress_callback)
                logger.info(f"Bulk process complete: {result.get('emails_processed', 0)} processed")

                # Check if cancelled
                if self.bulk_processor.cancel_flag.is_set():
                    self.ui(self.bulk_progress_label.configure, text="Analysis stopped by user")
                    self.ui(self.update_status, "Bulk analysis stopped")
                else:
                    # Display results
                    self.ui(self.display_bulk_results, result)
                    self.ui(self.log_activity, f"Bulk analysis completed: {len(emails)} emails processed")

            except Exception as e:
                error_msg = str(e)
                logger.error(f"Bulk processing error: {e}", exc_info=True)
                self.ui(lambda msg=error_msg: messagebox.showerror("Error", f"Processing failed: {msg}"))
            finally:
                # Toggle buttons back: enable Start, disable Stop
                self.ui(lambda: self.process_btn.configure(state="normal"))
                self.ui(lambda: self.stop_bulk_btn.configure(state="disabled"))

        threading.Thread(target=process_thread, daemon=True).start()

    def display_bulk_results(self, result):
        """Display bulk analysis results"""
        # Track results in UI thread (thread-safe)
        try:
            self.current_results.extend(result.get('results', []))
            # Cap results to prevent unbounded memory growth
            if len(self.current_results) > 5000:
                self.current_results = self.current_results[-5000:]
        except Exception as e:
            logger.debug(f"Could not extend results: {e}")

        stats = result.get('statistics', {})
        if not stats:
            self.bulk_progress_label.configure(text="Analysis completed but statistics unavailable")
            self.bulk_progress_bar.set_progress(1.0)
            self.update_header_stats()
            self.refresh_dashboard()
            self.update_status("Bulk analysis complete (no statistics)")
            return

        valid = max(stats.get('valid', 1), 1)
        domain_stats = stats.get('domain_stats', {})
        proc_time = result.get('processing_time', 0)

        # Create summary
        summary = f"""
Bulk Analysis Complete!
{'=' * 50}

Total Emails Processed: {stats.get('total', 0)}
Valid Results: {stats.get('valid', 0)}
Failed: {stats.get('failed', 0)}

Risk Distribution:
• Critical: {stats.get('critical', 0)} ({stats.get('critical', 0) / valid * 100:.1f}%)
• High: {stats.get('high', 0)} ({stats.get('high', 0) / valid * 100:.1f}%)
• Medium: {stats.get('medium', 0)} ({stats.get('medium', 0) / valid * 100:.1f}%)
• Low: {stats.get('low', 0)} ({stats.get('low', 0) / valid * 100:.1f}%)
• Minimal: {stats.get('minimal', 0)} ({stats.get('minimal', 0) / valid * 100:.1f}%)

Average Risk Score: {stats.get('avg_risk', 0):.1f}
Emails with Threats: {stats.get('with_threats', 0)}
Emails in Breaches: {stats.get('with_breaches', 0)}

Processing Time: {proc_time:.2f} seconds
Processing Rate: {stats.get('valid', 0) / max(proc_time, 1):.1f} emails/second

Domain Statistics:
• Unique Domains: {domain_stats.get('unique_domains', 0)}
• Suspicious Domains: {domain_stats.get('suspicious_domains', 0)}

Top Domains:
"""

        for domain, count in domain_stats.get('top_domains', [])[:5]:
            summary += f"  {domain}: {count} emails\n"

        # Add detailed breach information for each email
        if stats.get('with_breaches', 0) > 0:
            summary += f"\n{'=' * 50}\n"
            summary += "DETAILED BREACH INFORMATION\n"
            summary += f"{'=' * 50}\n\n"

            breach_count = 0
            for email_result in result.get('results', []):
                breach_info = email_result.get('breach_info') or {}
                if breach_info.get('found'):
                    breach_count += 1
                    summary += f"\n{breach_count}. Email: {email_result.get('email', 'N/A')}\n"
                    summary += f"   Breach Count: {breach_info.get('count', 0)}\n"
                    summary += f"   Severity: {breach_info.get('severity', 'medium').upper()}\n"

                    if breach_info.get('details'):
                        summary += f"   Breaches:\n"
                        for breach in breach_info['details'][:3]:  # Show first 3 breaches per email
                            if isinstance(breach, dict):
                                summary += f"   - {breach.get('name', 'Unknown')}"
                                if breach.get('breach_date'):
                                    summary += f" ({breach['breach_date']})"
                                summary += "\n"
                                if breach.get('domain'):
                                    summary += f"     Domain: {breach['domain']}\n"
                                if breach.get('data_classes'):
                                    data_types = ', '.join(str(dc) for dc in breach['data_classes'][:5])
                                    summary += f"     Data: {data_types}\n"

                        if breach_info.get('count', 0) > 3:
                            summary += f"   ... and {breach_info.get('count', 0) - 3} more breaches\n"

                    if breach_info.get('mitigation_steps'):
                        summary += f"\n   ACTIONS REQUIRED:\n"
                        for i, step in enumerate(breach_info['mitigation_steps'][:3], 1):
                            summary += f"   {i}. {step}\n"

                    summary += "\n"

        # Show results frame and display summary
        if hasattr(self, 'bulk_results_frame') and self.bulk_results_frame.winfo_exists():
            self.bulk_results_frame.pack(fill="both", expand=True, pady=10)
        if hasattr(self, 'bulk_results') and self.bulk_results.winfo_exists():
            self.bulk_results.delete("1.0", tk.END)
            self.bulk_results.insert("1.0", summary)

        # Update progress
        if hasattr(self, 'bulk_progress_label') and self.bulk_progress_label.winfo_exists():
            self.bulk_progress_label.configure(text="Analysis complete!")
        if hasattr(self, 'bulk_progress_bar') and self.bulk_progress_bar.winfo_exists():
            self.bulk_progress_bar.set_progress(1.0)

        # Update dashboard
        self.update_header_stats()
        self.refresh_dashboard()

        # Update analytics charts
        if hasattr(self, 'heatmap_ax'):
            try:
                self.update_analytics_charts()
            except Exception as e:
                logger.debug(f"Could not update analytics charts: {e}")

        messagebox.showinfo("Success", f"Bulk analysis completed!\n\nProcessed {stats.get('total', 0)} emails")

    def check_dns_security(self):
        """Check DNS security for domain"""
        domain = self.dns_entry.get().strip()
        if not domain:
            messagebox.showwarning("Warning", "Please enter a domain")
            return

        if not self.analyzer:
            messagebox.showwarning("Warning", "Analyzer not ready")
            return

        # Show results frame
        self.dns_results.pack(fill="both", expand=True, padx=20, pady=10)

        # Clear previous results
        for widget in self.dns_results.winfo_children():
            widget.destroy()

        # Loading message
        loading_label = ctk.CTkLabel(
            self.dns_results,
            text="Checking DNS security...",
            font=ctk.CTkFont(size=16),
            text_color=COLORS['text_secondary']
        )
        loading_label.pack(expand=True)

        def check_thread():
            try:
                # Check DNS
                dns_result = self.analyzer.threat_intel.check_dns_security(domain)

                # Check domain reputation
                reputation = self.analyzer.threat_intel.check_domain_reputation(domain)

                # Update UI
                self.ui(self.display_dns_results, domain, dns_result, reputation)

                # Log activity
                self.ui(self.log_activity, f"DNS check: {domain} - Score: {dns_result.get('score', 0)}")

            except Exception as e:
                error_msg = str(e)
                def cleanup_and_error(msg=error_msg):
                    if hasattr(self, 'dns_results') and self.dns_results.winfo_exists():
                        for w in self.dns_results.winfo_children():
                            w.destroy()
                        self.dns_results.pack_forget()
                    messagebox.showerror("Error", f"DNS check failed: {msg}")
                self.ui(cleanup_and_error)
                logger.error(f"DNS check error: {e}")

        threading.Thread(target=check_thread, daemon=True).start()

    def display_dns_results(self, domain, dns_result, reputation):
        """Display DNS security results"""
        # Clear loading message
        for widget in self.dns_results.winfo_children():
            widget.destroy()

        # Results content
        content = ctk.CTkFrame(self.dns_results, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=30, pady=30)

        # Domain header
        ctk.CTkLabel(
            content,
            text=f"DNS Security Report: {domain}",
            font=ctk.CTkFont(size=20, weight="bold")
        ).pack(pady=(0, 20))

        # Security score
        score = dns_result.get('score', 0)
        score_color = COLORS['success'] if score >= 80 else COLORS['warning'] if score >= 50 else COLORS['danger']

        score_frame = ctk.CTkFrame(content, fg_color=COLORS['bg_tertiary'], corner_radius=10)
        score_frame.pack(fill="x", pady=10)

        score_content = ctk.CTkFrame(score_frame, fg_color="transparent")
        score_content.pack(padx=20, pady=20)

        ctk.CTkLabel(
            score_content,
            text=f"Security Score: {score}/100",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color=score_color
        ).pack()

        # DNS records status
        records_frame = ctk.CTkFrame(content, fg_color=COLORS['bg_tertiary'], corner_radius=10)
        records_frame.pack(fill="x", pady=10)

        records_content = ctk.CTkFrame(records_frame, fg_color="transparent")
        records_content.pack(padx=20, pady=20)

        ctk.CTkLabel(
            records_content,
            text="DNS Records Status",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(0, 15))

        # Display each DNS check
        checks = [
            ('SPF Record', dns_result.get('spf'), 'Sender Policy Framework'),
            ('DMARC Policy', dns_result.get('dmarc'), 'Domain-based Message Authentication'),
            ('DKIM Support', dns_result.get('dkim'), 'DomainKeys Identified Mail'),
            ('MX Records', dns_result.get('mx'), 'Mail Exchange Records'),
            ('DNSSEC', dns_result.get('dnssec'), 'DNS Security Extensions'),
            ('BIMI', dns_result.get('bimi'), 'Brand Indicators for Message Identification'),
            ('MTA-STS', dns_result.get('mta_sts'), 'Mail Transfer Agent Strict Transport Security'),
            ('TLS-RPT', dns_result.get('tls_rpt'), 'TLS Reporting Policy'),
        ]

        for check_name, status, description in checks:
            check_frame = ctk.CTkFrame(records_content, fg_color=COLORS['bg_quaternary'], corner_radius=8)
            check_frame.pack(fill="x", pady=5)

            check_content = ctk.CTkFrame(check_frame, fg_color="transparent")
            check_content.pack(fill="x", padx=15, pady=10)

            # Status icon
            status_icon = "✅" if status else "❌"
            status_color = COLORS['success'] if status else COLORS['danger']

            ctk.CTkLabel(
                check_content,
                text=f"{status_icon} {check_name}",
                font=ctk.CTkFont(size=14, weight="bold"),
                text_color=status_color,
                anchor="w"
            ).pack(fill="x")

            ctk.CTkLabel(
                check_content,
                text=description,
                font=ctk.CTkFont(size=12),
                text_color=COLORS['text_secondary'],
                anchor="w"
            ).pack(fill="x")

        # Domain reputation
        if reputation:
            rep_frame = ctk.CTkFrame(content, fg_color=COLORS['bg_tertiary'], corner_radius=10)
            rep_frame.pack(fill="x", pady=10)

            rep_content = ctk.CTkFrame(rep_frame, fg_color="transparent")
            rep_content.pack(padx=20, pady=20)

            ctk.CTkLabel(
                rep_content,
                text="Domain Reputation",
                font=ctk.CTkFont(size=16, weight="bold")
            ).pack(pady=(0, 15))

            rep_score = reputation.get('score', 50)
            rep_color = COLORS['success'] if rep_score >= 70 else COLORS['warning'] if rep_score >= 40 else COLORS[
                'danger']

            ctk.CTkLabel(
                rep_content,
                text=f"Reputation Score: {rep_score}/100",
                font=ctk.CTkFont(size=14),
                text_color=rep_color
            ).pack()

            if reputation.get('age'):
                ctk.CTkLabel(
                    rep_content,
                    text=f"Domain Age: {reputation['age']} days",
                    font=ctk.CTkFont(size=12),
                    text_color=COLORS['text_secondary']
                ).pack(pady=5)

            if reputation.get('flags'):
                flags_text = "Flags: " + ", ".join(str(f) for f in reputation['flags'])
                ctk.CTkLabel(
                    rep_content,
                    text=flags_text,
                    font=ctk.CTkFont(size=12),
                    text_color=COLORS['warning']
                ).pack(pady=5)

    def _get_ml_accuracy_display(self):
        """Get real ML accuracy for dashboard display."""
        if not self.config.enable_ml:
            return "N/A"
        if (self.analyzer and hasattr(self.analyzer, 'ml_engine')
                and self.analyzer.ml_engine
                and hasattr(self.analyzer.ml_engine, 'model_metrics')
                and self.analyzer.ml_engine.model_metrics):
            metrics = self.analyzer.ml_engine.model_metrics
            for name in ['ensemble', 'random_forest', 'xgboost']:
                if name in metrics and 'accuracy' in metrics[name]:
                    acc = metrics[name]['accuracy']
                    if isinstance(acc, (int, float)) and acc > 0:
                        return f"{acc * 100:.1f}%"
            first = next(iter(metrics.values()), {})
            acc = first.get('accuracy', 0)
            if isinstance(acc, (int, float)) and acc > 0:
                return f"{acc * 100:.1f}%"
        return "N/A"

    def refresh_dashboard(self):
        """Refresh dashboard data"""
        if not self.current_results:
            return

        # Calculate statistics
        total = len(self.current_results)
        critical = sum(1 for r in self.current_results if r.get('risk_level') == 'critical')
        threats = sum(1 for r in self.current_results if r.get('threats'))
        safe = sum(1 for r in self.current_results if r.get('risk_level') in ['low', 'minimal'])
        avg_risk = np.mean([(r.get('risk_score') or 0) for r in self.current_results])

        # Update stat cards
        card_updates = {
            'total_scans': str(total),
            'critical_risks': str(critical),
            'active_threats': str(threats),
            'safe_emails': str(safe),
            'avg_risk': f"{avg_risk:.1f}",
            'ml_accuracy': self._get_ml_accuracy_display()
        }

        for card in self.dashboard_cards:
            if hasattr(card, 'key') and card.key in card_updates:
                card.value_label.configure(text=card_updates[card.key])

        # Update charts
        self.update_dashboard_charts()

        # Update radar chart
        threat_categories = {
            'Phishing': 0,
            'Malware': 0,
            'Spam': 0,
            'Breach': 0,
            'DNS': 0,
            'Domain': 0
        }

        for result in self.current_results:
            for threat in result.get('threats', []):
                threat_type = threat.get('type', '').lower()
                if 'phishing' in threat_type:
                    threat_categories['Phishing'] += 1
                elif 'malware' in threat_type:
                    threat_categories['Malware'] += 1
                elif 'spam' in threat_type:
                    threat_categories['Spam'] += 1
                elif 'breach' in threat_type:
                    threat_categories['Breach'] += 1

            dns_checked = result.get('analysis_flags', {}).get('dns', False)
            if dns_checked and not (result.get('dns_security') or {}).get('spf'):
                threat_categories['DNS'] += 1

            if (result.get('domain_reputation') or {}).get('score', 100) < 50:
                threat_categories['Domain'] += 1

        # Normalize radar values safely
        values = list(threat_categories.values())
        max_val = max(values) if values else 1

        if max_val == 0:
            normalized_values = [0 for _ in values]
        else:
            normalized_values = [v / max_val * 100 for v in values]

        # Update radar chart
        if hasattr(self, 'radar_chart') and self.radar_chart:
            self.radar_chart.update_values(normalized_values)

    def update_dashboard_charts(self):
        """Update dashboard visualization charts"""
        # Risk distribution pie chart
        self.risk_chart_ax.clear()

        if self.current_results:
            risk_levels = [r.get('risk_level', 'unknown') for r in self.current_results]
            risk_counts = Counter(risk_levels)

            if risk_counts:
                colors_map = {
                    'critical': '#f85149',
                    'high': '#d29922',
                    'medium': '#d29922',
                    'low': '#3fb950',
                    'minimal': '#4f8ff7'
                }

                labels = []
                sizes = []
                colors = []

                for level in ['critical', 'high', 'medium', 'low', 'minimal']:
                    if level in risk_counts:
                        labels.append(level.title())
                        sizes.append(risk_counts[level])
                        colors.append(colors_map[level])

                if sizes:
                    _wedges, _texts, autotexts = self.risk_chart_ax.pie(
                        sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                        startangle=90, textprops={'color': COLORS['text_primary']}
                    )

                    for autotext in autotexts:
                        autotext.set_color('white')
                        autotext.set_weight('bold')

        self.risk_chart_ax.set_title('Risk Distribution', color='#e6edf3', fontsize=14, pad=20)
        self.risk_chart_canvas.draw()

        # Threat timeline
        self.timeline_ax.clear()

        if len(self.current_results) > 1:
            times = list(range(len(self.current_results)))
            scores = [r.get('risk_score', 0) for r in self.current_results]

            self.timeline_ax.plot(times, scores, color='#4f8ff7', linewidth=2, marker='o')
            self.timeline_ax.fill_between(times, scores, alpha=0.3, color='#4f8ff7')

            self.timeline_ax.set_xlabel('Analysis #', color='#8b949e')
            self.timeline_ax.set_ylabel('Risk Score', color='#8b949e')
            self.timeline_ax.set_title('Risk Score Timeline', color='#e6edf3', fontsize=14, pad=20)
            self.timeline_ax.grid(True, alpha=0.3, color='#30363d')

            # Add risk level zones
            self.timeline_ax.axhspan(80, 100, alpha=0.1, color='#f85149')
            self.timeline_ax.axhspan(60, 80, alpha=0.1, color='#d29922')
            self.timeline_ax.axhspan(40, 60, alpha=0.1, color='#d29922')
            self.timeline_ax.axhspan(0, 40, alpha=0.1, color='#3fb950')

        self.timeline_canvas.draw()

    def update_analytics_charts(self):
        """Update analytics page charts with REAL data"""
        try:
            if not hasattr(self, 'heatmap_ax'):
                return

            import numpy as np
            from collections import Counter

            # Chart 1: Risk Level Distribution (Pie Chart)
            if hasattr(self, 'risk_dist_ax'):
                self.risk_dist_ax.clear()

                if self.current_results and len(self.current_results) > 0:
                    # Count risk levels from real data
                    risk_levels = [r.get('risk_level', 'unknown') for r in self.current_results]
                    risk_counts = Counter(risk_levels)

                    # Define order and colors
                    level_order = ['critical', 'high', 'medium', 'low', 'minimal']
                    level_colors = {
                        'critical': '#f85149',
                        'high': '#d29922',
                        'medium': '#d29922',
                        'low': '#3fb950',
                        'minimal': '#4f8ff7'
                    }

                    labels = []
                    sizes = []
                    colors = []

                    for level in level_order:
                        if level in risk_counts:
                            labels.append(level.upper())
                            sizes.append(risk_counts[level])
                            colors.append(level_colors.get(level, '#6e7681'))

                    if sizes:
                        wedges, texts, autotexts = self.risk_dist_ax.pie(
                            sizes, labels=labels, colors=colors,
                            autopct='%1.1f%%', startangle=90,
                            textprops={'color': '#e6edf3', 'fontsize': 11, 'weight': 'bold'}
                        )

                        for autotext in autotexts:
                            autotext.set_color('#000000')
                            autotext.set_fontsize(10)

                        self.risk_dist_ax.set_title(f'Risk Distribution ({sum(sizes)} Emails)',
                                                   color='#e6edf3', fontsize=13, pad=10)
                else:
                    self.risk_dist_ax.text(0.5, 0.5, 'No data available\n\nAnalyze emails first',
                                          ha='center', va='center', transform=self.risk_dist_ax.transAxes,
                                          color='#8b949e', fontsize=12)
                    self.risk_dist_ax.set_title('Risk Distribution', color='#e6edf3', fontsize=13)

                self.risk_dist_canvas.draw()

            # Chart 3: Top Threat Types (Bar Chart)
            if hasattr(self, 'threat_types_ax'):
                self.threat_types_ax.clear()

                if self.current_results and len(self.current_results) > 0:
                    # Count threat types from real data
                    threat_types = []
                    for r in self.current_results:
                        for threat in r.get('threats', []):
                            threat_type = threat.get('type', 'unknown').replace('_', ' ').title()
                            threat_types.append(threat_type)

                    if threat_types:
                        threat_counts = Counter(threat_types)
                        top_threats = threat_counts.most_common(10)

                        types = [t[0] for t in top_threats]
                        counts = [t[1] for t in top_threats]

                        bars = self.threat_types_ax.barh(types, counts, color='#f85149', alpha=0.8, edgecolor='#ff7b72')

                        # Add value labels
                        for i, (bar, count) in enumerate(zip(bars, counts)):
                            self.threat_types_ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                                                     str(count), ha='left', va='center',
                                                     color='#8b949e', fontsize=9, weight='bold')

                        self.threat_types_ax.set_xlabel('Count', color='#8b949e', fontsize=10)
                        self.threat_types_ax.set_title(f'Top Threat Types ({sum(counts)} Total)',
                                                      color='#e6edf3', fontsize=13, pad=10)
                        self.threat_types_ax.tick_params(colors='#8b949e', labelsize=9)
                        self.threat_types_ax.grid(axis='x', alpha=0.2)
                        self.threat_types_ax.invert_yaxis()
                    else:
                        self.threat_types_ax.text(0.5, 0.5, 'No threats detected\n\nAll emails appear safe',
                                                 ha='center', va='center', transform=self.threat_types_ax.transAxes,
                                                 color='#3fb950', fontsize=12, weight='bold')
                        self.threat_types_ax.set_title('Top Threat Types', color='#e6edf3', fontsize=13)
                else:
                    self.threat_types_ax.text(0.5, 0.5, 'No data available\n\nAnalyze emails first',
                                             ha='center', va='center', transform=self.threat_types_ax.transAxes,
                                             color='#8b949e', fontsize=12)
                    self.threat_types_ax.set_title('Top Threat Types', color='#e6edf3', fontsize=13)

                self.threat_types_canvas.draw()

            # Risk heatmap — recreate axes to avoid colorbar accumulation
            self.heatmap_fig.clear()
            self.heatmap_ax = self.heatmap_fig.add_subplot(111)
            self.heatmap_ax.set_facecolor('#161b22')

            if self.current_results and len(self.current_results) > 0:
                # Group by domain
                domain_risks = {}
                for result in self.current_results:
                    email = result.get('email', '')
                    domain = email.split('@')[1] if '@' in email else 'unknown'
                    if domain not in domain_risks:
                        domain_risks[domain] = []
                    domain_risks[domain].append(result.get('risk_score', 0))

                # Create heatmap data
                if domain_risks:
                    # Sort domains by average risk score (highest first)
                    sorted_domains = sorted(domain_risks.items(), key=lambda x: sum(x[1])/len(x[1]), reverse=True)
                    domains = [d[0] for d in sorted_domains[:10]]  # Top 10 highest risk domains
                    risk_matrix = []

                    max_emails = max(len(domain_risks[d]) for d in domains)
                    display_count = min(max_emails, 10)  # Show up to 10 emails per domain

                    for domain in domains:
                        scores = domain_risks[domain][:display_count]
                        # Pad with None to make uniform length
                        padded_scores = scores + [None] * (display_count - len(scores))
                        risk_matrix.append(padded_scores)

                    # Convert None to NaN for matplotlib
                    import numpy as np
                    risk_matrix_array = np.array([[s if s is not None else np.nan for s in row] for row in risk_matrix])

                    if risk_matrix_array.size > 0:
                        im = self.heatmap_ax.imshow(risk_matrix_array, cmap='RdYlGn_r', aspect='auto',
                                                    vmin=0, vmax=100, interpolation='nearest')

                        self.heatmap_ax.set_xticks(range(display_count))
                        self.heatmap_ax.set_xticklabels([f'Email {i + 1}' for i in range(display_count)],
                                                        rotation=45, ha='right', color='#8b949e')
                        self.heatmap_ax.set_yticks(range(len(domains)))
                        self.heatmap_ax.set_yticklabels(domains, color='#8b949e')

                        # Add colorbar (remove old one first if it exists)
                        if hasattr(self, '_heatmap_colorbar') and self._heatmap_colorbar is not None:
                            try:
                                self._heatmap_colorbar.remove()
                            except Exception:
                                pass  # Ignore if removal fails
                        self._heatmap_colorbar = self.heatmap_fig.colorbar(im, ax=self.heatmap_ax)
                        self._heatmap_colorbar.set_label('Risk Score', color='#8b949e')
                        self._heatmap_colorbar.ax.tick_params(colors='#8b949e')

                        # Add values as text
                        for i, domain_scores in enumerate(risk_matrix_array):
                            for j, score in enumerate(domain_scores):
                                if not np.isnan(score):
                                    text_color = '#000000' if score < 50 else '#e6edf3'
                                    self.heatmap_ax.text(j, i, f'{int(score)}',
                                                        ha='center', va='center',
                                                        color=text_color, fontsize=9, weight='bold')

                        self.heatmap_ax.set_title('Risk Heatmap by Domain (Top 10 Highest Risk)',
                                                 color='#e6edf3', fontsize=14, pad=10)
                    else:
                        # No valid data
                        self.heatmap_ax.text(0.5, 0.5, 'No risk data available\nAnalyze emails to see heatmap',
                                           ha='center', va='center', transform=self.heatmap_ax.transAxes,
                                           color='#8b949e', fontsize=14)
                        self.heatmap_ax.set_title('Risk Heatmap by Domain', color='#e6edf3', fontsize=14)
                else:
                    # No domains found
                    self.heatmap_ax.text(0.5, 0.5, 'No domains found\nAnalyze emails to see heatmap',
                                       ha='center', va='center', transform=self.heatmap_ax.transAxes,
                                       color='#8b949e', fontsize=14)
                    self.heatmap_ax.set_title('Risk Heatmap by Domain', color='#e6edf3', fontsize=14)
            else:
                # No results available - show placeholder
                self.heatmap_ax.text(0.5, 0.5, 'No analysis data available\n\nGo to "Analysis" or "Bulk" page\nto analyze emails first',
                                   ha='center', va='center', transform=self.heatmap_ax.transAxes,
                                   color='#8b949e', fontsize=14, style='italic')
                self.heatmap_ax.set_title('Risk Heatmap by Domain', color='#e6edf3', fontsize=14)
                self.heatmap_ax.set_xticks([])
                self.heatmap_ax.set_yticks([])

            self.heatmap_canvas.draw()

            # Chart 4: Breach Statistics by Domain (Bar Chart)
            if hasattr(self, 'breach_stats_ax'):
                self.breach_stats_ax.clear()

                if self.current_results and len(self.current_results) > 0:
                    # Count breaches by domain from real data
                    domain_breaches = {}
                    for r in self.current_results:
                        email = r.get('email', '')
                        domain = email.split('@')[1] if '@' in email else 'unknown'
                        breach_info = r.get('breach_info') or {}

                        if domain not in domain_breaches:
                            domain_breaches[domain] = {'count': 0, 'total': 0}

                        domain_breaches[domain]['total'] += 1
                        if breach_info.get('found'):
                            domain_breaches[domain]['count'] += 1

                    if domain_breaches:
                        # Calculate breach percentage
                        domain_data = []
                        for domain, data in domain_breaches.items():
                            if data['total'] > 0:
                                percentage = (data['count'] / data['total']) * 100
                                domain_data.append((domain, percentage, data['count'], data['total']))

                        # Sort by percentage (highest first) and take top 10
                        domain_data.sort(key=lambda x: x[1], reverse=True)
                        top_domains = domain_data[:10]

                        if top_domains:
                            domains = [d[0] for d in top_domains]
                            percentages = [d[1] for d in top_domains]

                            bars = self.breach_stats_ax.barh(domains, percentages, color='#f85149', alpha=0.8, edgecolor='#ff7b72')

                            # Add labels showing count/total
                            for i, (bar, data) in enumerate(zip(bars, top_domains)):
                                label = f'{data[2]}/{data[3]} ({data[1]:.0f}%)'
                                self.breach_stats_ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                                                         label, ha='left', va='center',
                                                         color='#8b949e', fontsize=9, weight='bold')

                            self.breach_stats_ax.set_xlabel('Breach Rate (%)', color='#8b949e', fontsize=10)
                            self.breach_stats_ax.set_xlim(0, 100)
                            self.breach_stats_ax.set_title('Breach Rate by Domain (Top 10)',
                                                          color='#e6edf3', fontsize=13, pad=10)
                            self.breach_stats_ax.tick_params(colors='#8b949e', labelsize=9)
                            self.breach_stats_ax.grid(axis='x', alpha=0.2)
                            self.breach_stats_ax.invert_yaxis()
                        else:
                            self.breach_stats_ax.text(0.5, 0.5, 'No breach data',
                                                     ha='center', va='center', transform=self.breach_stats_ax.transAxes,
                                                     color='#8b949e', fontsize=12)
                    else:
                        self.breach_stats_ax.text(0.5, 0.5, 'No breach data available',
                                                 ha='center', va='center', transform=self.breach_stats_ax.transAxes,
                                                 color='#8b949e', fontsize=12)
                else:
                    self.breach_stats_ax.text(0.5, 0.5, 'No data available\n\nAnalyze emails first',
                                             ha='center', va='center', transform=self.breach_stats_ax.transAxes,
                                             color='#8b949e', fontsize=12)
                    self.breach_stats_ax.set_title('Breach Statistics', color='#e6edf3', fontsize=13)

                self.breach_stats_canvas.draw()

            # Chart 5: Risk Score Distribution (Histogram)
            if hasattr(self, 'risk_hist_ax'):
                self.risk_hist_ax.clear()

                if self.current_results and len(self.current_results) > 0:
                    # Get risk scores from real data
                    risk_scores = [r.get('risk_score', 0) for r in self.current_results]

                    if risk_scores:
                        # Create histogram
                        n, bins, patches = self.risk_hist_ax.hist(risk_scores, bins=20, color='#4f8ff7',
                                                                  alpha=0.7, edgecolor='#3fb950')

                        # Color bars based on risk level
                        for i, patch in enumerate(patches):
                            bin_center = (bins[i] + bins[i+1]) / 2
                            if bin_center >= 80:
                                patch.set_facecolor('#f85149')
                            elif bin_center >= 60:
                                patch.set_facecolor('#d29922')
                            elif bin_center >= 40:
                                patch.set_facecolor('#d29922')
                            else:
                                patch.set_facecolor('#3fb950')

                        self.risk_hist_ax.set_xlabel('Risk Score', color='#8b949e', fontsize=10)
                        self.risk_hist_ax.set_ylabel('Frequency', color='#8b949e', fontsize=10)
                        self.risk_hist_ax.set_title(f'Risk Score Distribution (Avg: {np.mean(risk_scores):.1f})',
                                                   color='#e6edf3', fontsize=13, pad=10)
                        self.risk_hist_ax.tick_params(colors='#8b949e', labelsize=9)
                        self.risk_hist_ax.grid(alpha=0.2)

                        # Add risk level zones
                        self.risk_hist_ax.axvspan(80, 100, alpha=0.1, color='#f85149')
                        self.risk_hist_ax.axvspan(60, 80, alpha=0.1, color='#d29922')
                        self.risk_hist_ax.axvspan(40, 60, alpha=0.1, color='#d29922')
                        self.risk_hist_ax.axvspan(0, 40, alpha=0.1, color='#3fb950')
                else:
                    self.risk_hist_ax.text(0.5, 0.5, 'No data available\n\nAnalyze emails first',
                                          ha='center', va='center', transform=self.risk_hist_ax.transAxes,
                                          color='#8b949e', fontsize=12)
                    self.risk_hist_ax.set_title('Risk Score Distribution', color='#e6edf3', fontsize=13)

                self.risk_hist_canvas.draw()

            # Chart 6: Top MITRE ATT&CK Techniques (Horizontal Bar)
            if hasattr(self, 'mitre_top_ax'):
                self.mitre_top_ax.clear()

                if self.current_results and len(self.current_results) > 0:
                    # Count MITRE techniques from real data
                    mitre_techniques = []
                    for r in self.current_results:
                        for tech in r.get('mitre_details', []):
                            tech_id = tech.get('id', 'Unknown')
                            tech_name = tech.get('name', 'Unknown')
                            mitre_techniques.append(f"{tech_id}: {tech_name[:30]}")  # Truncate long names

                    if mitre_techniques:
                        tech_counts = Counter(mitre_techniques)
                        top_techniques = tech_counts.most_common(10)

                        techniques = [t[0] for t in top_techniques]
                        counts = [t[1] for t in top_techniques]

                        bars = self.mitre_top_ax.barh(techniques, counts, color='#58a6ff', alpha=0.8, edgecolor='#79b8ff')

                        # Add value labels
                        for i, (bar, count) in enumerate(zip(bars, counts)):
                            self.mitre_top_ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                                                  str(count), ha='left', va='center',
                                                  color='#8b949e', fontsize=9, weight='bold')

                        self.mitre_top_ax.set_xlabel('Frequency', color='#8b949e', fontsize=10)
                        self.mitre_top_ax.set_title(f'Top MITRE Techniques ({sum(counts)} Total)',
                                                   color='#e6edf3', fontsize=13, pad=10)
                        self.mitre_top_ax.tick_params(colors='#8b949e', labelsize=8)
                        self.mitre_top_ax.grid(axis='x', alpha=0.2)
                        self.mitre_top_ax.invert_yaxis()
                    else:
                        self.mitre_top_ax.text(0.5, 0.5, 'No MITRE techniques detected',
                                              ha='center', va='center', transform=self.mitre_top_ax.transAxes,
                                              color='#8b949e', fontsize=12)
                        self.mitre_top_ax.set_title('Top MITRE Techniques', color='#e6edf3', fontsize=13)
                else:
                    self.mitre_top_ax.text(0.5, 0.5, 'No data available\n\nAnalyze emails first',
                                          ha='center', va='center', transform=self.mitre_top_ax.transAxes,
                                          color='#8b949e', fontsize=12)
                    self.mitre_top_ax.set_title('Top MITRE Techniques', color='#e6edf3', fontsize=13)

                self.mitre_top_canvas.draw()

            # Chart 7: ML Model Performance Comparison (Bar Chart)
            if hasattr(self, 'ml_models_ax'):
                self.ml_models_ax.clear()

                if self.current_results and len(self.current_results) > 0:
                    # Extract ML predictions from results - dynamically discover model names
                    skip_keys = {'ensemble', 'is_malicious', 'precision_threshold', 'anomaly_score'}

                    model_scores = {}
                    for r in self.current_results:
                        ml_preds = r.get('ml_predictions') or {}
                        for key, val in ml_preds.items():
                            if key in skip_keys:
                                continue
                            if not isinstance(val, (int, float)):
                                continue
                            if key not in model_scores:
                                model_scores[key] = []
                            model_scores[key].append(val)

                    # Also include ensemble if present
                    for r in self.current_results:
                        ml_preds = r.get('ml_predictions') or {}
                        if 'ensemble' in ml_preds and isinstance(ml_preds['ensemble'], (int, float)):
                            if 'ensemble' not in model_scores:
                                model_scores['ensemble'] = []
                            model_scores['ensemble'].append(ml_preds['ensemble'])

                    # Average the scores (multiply by 100 for percentage display)
                    model_scores = {k: (sum(v) / len(v)) * 100 for k, v in model_scores.items() if v}

                    if model_scores:
                        # Sort by score
                        sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
                        model_names = [m[0].replace('_', ' ').title() for m in sorted_models]
                        scores = [m[1] for m in sorted_models]

                        # Create gradient colors
                        colors = ['#4CAF50', '#2196F3', '#FF9800', '#E91E63',
                                 '#9C27B0', '#00BCD4', '#FFC107'][:len(model_names)]

                        bars = self.ml_models_ax.bar(range(len(model_names)), scores,
                                                     color=colors, alpha=0.8, edgecolor='#e6edf3')

                        self.ml_models_ax.set_xticks(range(len(model_names)))
                        self.ml_models_ax.set_xticklabels(model_names, rotation=45, ha='right', color='#8b949e')
                        self.ml_models_ax.set_ylabel('Avg Risk Score (%)', color='#8b949e', fontsize=10)
                        self.ml_models_ax.set_ylim(0, 100)
                        self.ml_models_ax.set_title(f'ML Model Performance ({len(self.current_results)} Emails)',
                                                   color='#e6edf3', fontsize=13, pad=10)
                        self.ml_models_ax.tick_params(colors='#8b949e', labelsize=9)
                        self.ml_models_ax.grid(axis='y', alpha=0.2)

                        # Add value labels on bars
                        for bar, score in zip(bars, scores):
                            height = bar.get_height()
                            self.ml_models_ax.text(bar.get_x() + bar.get_width()/2., height,
                                                  f'{score:.1f}%', ha='center', va='bottom',
                                                  color='#e6edf3', fontsize=9, weight='bold')
                    else:
                        self.ml_models_ax.text(0.5, 0.5, 'No ML predictions available\n\nML may be disabled',
                                              ha='center', va='center', transform=self.ml_models_ax.transAxes,
                                              color='#d29922', fontsize=12, weight='bold')
                        self.ml_models_ax.set_title('ML Model Performance', color='#e6edf3', fontsize=13)
                else:
                    self.ml_models_ax.text(0.5, 0.5, 'No data available\n\nAnalyze emails first',
                                          ha='center', va='center', transform=self.ml_models_ax.transAxes,
                                          color='#8b949e', fontsize=12)
                    self.ml_models_ax.set_title('ML Model Performance', color='#e6edf3', fontsize=13)

                self.ml_models_canvas.draw()

            # Chart 8: ML Prediction Confidence Distribution (Doughnut Chart)
            if hasattr(self, 'ml_confidence_ax'):
                self.ml_confidence_ax.clear()

                if self.current_results and len(self.current_results) > 0:
                    # Get ensemble predictions and calculate confidence ranges
                    confidence_ranges = {
                        'Very High (80-100%)': 0,
                        'High (60-80%)': 0,
                        'Medium (40-60%)': 0,
                        'Low (20-40%)': 0,
                        'Very Low (0-20%)': 0
                    }

                    for r in self.current_results:
                        ml_preds = r.get('ml_predictions') or {}
                        if 'ensemble' in ml_preds and isinstance(ml_preds['ensemble'], (int, float)):
                            score = ml_preds['ensemble'] * 100  # Convert 0-1 to 0-100
                            if score >= 80:
                                confidence_ranges['Very High (80-100%)'] += 1
                            elif score >= 60:
                                confidence_ranges['High (60-80%)'] += 1
                            elif score >= 40:
                                confidence_ranges['Medium (40-60%)'] += 1
                            elif score >= 20:
                                confidence_ranges['Low (20-40%)'] += 1
                            else:
                                confidence_ranges['Very Low (0-20%)'] += 1

                    # Filter out zero values
                    labels = []
                    sizes = []
                    colors_list = []
                    color_map = {
                        'Very High (80-100%)': '#f85149',
                        'High (60-80%)': '#d29922',
                        'Medium (40-60%)': '#d29922',
                        'Low (20-40%)': '#4f8ff7',
                        'Very Low (0-20%)': '#3fb950'
                    }

                    for label, count in confidence_ranges.items():
                        if count > 0:
                            labels.append(label)
                            sizes.append(count)
                            colors_list.append(color_map[label])

                    if sizes:
                        # Create doughnut chart
                        wedges, texts, autotexts = self.ml_confidence_ax.pie(
                            sizes, labels=labels, colors=colors_list,
                            autopct='%1.1f%%', startangle=90, pctdistance=0.85,
                            textprops={'color': '#e6edf3', 'fontsize': 10, 'weight': 'bold'}
                        )

                        # Draw circle in center for doughnut effect
                        centre_circle = plt.Circle((0, 0), 0.70, fc='#161b22')
                        self.ml_confidence_ax.add_artist(centre_circle)

                        for autotext in autotexts:
                            autotext.set_color('#000000')
                            autotext.set_fontsize(9)

                        self.ml_confidence_ax.set_title(f'ML Confidence Distribution ({sum(sizes)} Emails)',
                                                       color='#e6edf3', fontsize=13, pad=10)
                    else:
                        self.ml_confidence_ax.text(0.5, 0.5, 'No ML predictions available\n\nML may be disabled',
                                                  ha='center', va='center', transform=self.ml_confidence_ax.transAxes,
                                                  color='#d29922', fontsize=12, weight='bold')
                        self.ml_confidence_ax.set_title('ML Confidence Distribution', color='#e6edf3', fontsize=13)
                else:
                    self.ml_confidence_ax.text(0.5, 0.5, 'No data available\n\nAnalyze emails first',
                                              ha='center', va='center', transform=self.ml_confidence_ax.transAxes,
                                              color='#8b949e', fontsize=12)
                    self.ml_confidence_ax.set_title('ML Confidence Distribution', color='#e6edf3', fontsize=13)

                self.ml_confidence_canvas.draw()

            # Chart 9: ML Predictions by Risk Level (Horizontal Bar Chart)
            if hasattr(self, 'ml_risk_ax'):
                self.ml_risk_ax.clear()

                if self.current_results and len(self.current_results) > 0:
                    # Group ML predictions by risk level
                    risk_ml_scores = {
                        'Critical (80-100)': [],
                        'High (60-80)': [],
                        'Medium (40-60)': [],
                        'Low (0-40)': []
                    }

                    for r in self.current_results:
                        risk_score = r.get('risk_score', 0)
                        ml_preds = r.get('ml_predictions') or {}

                        if 'ensemble' in ml_preds and isinstance(ml_preds['ensemble'], (int, float)):
                            ml_score = ml_preds['ensemble']

                            if risk_score >= 80:
                                risk_ml_scores['Critical (80-100)'].append(ml_score)
                            elif risk_score >= 60:
                                risk_ml_scores['High (60-80)'].append(ml_score)
                            elif risk_score >= 40:
                                risk_ml_scores['Medium (40-60)'].append(ml_score)
                            else:
                                risk_ml_scores['Low (0-40)'].append(ml_score)

                    # Calculate averages
                    risk_levels = []
                    avg_scores = []
                    counts = []
                    colors_list = []
                    color_map = {
                        'Critical (80-100)': '#f85149',
                        'High (60-80)': '#d29922',
                        'Medium (40-60)': '#d29922',
                        'Low (0-40)': '#3fb950'
                    }

                    for level in ['Critical (80-100)', 'High (60-80)', 'Medium (40-60)', 'Low (0-40)']:
                        if risk_ml_scores[level]:
                            risk_levels.append(level)
                            avg_scores.append(sum(risk_ml_scores[level]) / len(risk_ml_scores[level]) * 100)
                            counts.append(len(risk_ml_scores[level]))
                            colors_list.append(color_map[level])

                    if risk_levels:
                        bars = self.ml_risk_ax.barh(risk_levels, avg_scores, color=colors_list,
                                                    alpha=0.8, edgecolor='#e6edf3')

                        # Add labels showing count and average
                        for bar, avg, count in zip(bars, avg_scores, counts):
                            label = f'{avg:.1f}% (n={count})'
                            self.ml_risk_ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                                                label, ha='left', va='center',
                                                color='#8b949e', fontsize=10, weight='bold')

                        self.ml_risk_ax.set_xlabel('Average ML Score (%)', color='#8b949e', fontsize=11)
                        self.ml_risk_ax.set_xlim(0, 100)
                        self.ml_risk_ax.set_title(f'ML Predictions by Risk Level ({sum(counts)} Emails)',
                                                 color='#e6edf3', fontsize=13, pad=10)
                        self.ml_risk_ax.tick_params(colors='#8b949e', labelsize=10)
                        self.ml_risk_ax.grid(axis='x', alpha=0.2)
                        self.ml_risk_ax.invert_yaxis()
                    else:
                        self.ml_risk_ax.text(0.5, 0.5, 'No ML predictions available\n\nML may be disabled',
                                            ha='center', va='center', transform=self.ml_risk_ax.transAxes,
                                            color='#d29922', fontsize=12, weight='bold')
                        self.ml_risk_ax.set_title('ML Predictions by Risk Level', color='#e6edf3', fontsize=13)
                else:
                    self.ml_risk_ax.text(0.5, 0.5, 'No data available\n\nAnalyze emails first',
                                        ha='center', va='center', transform=self.ml_risk_ax.transAxes,
                                        color='#8b949e', fontsize=12)
                    self.ml_risk_ax.set_title('ML Predictions by Risk Level', color='#e6edf3', fontsize=13)

                self.ml_risk_canvas.draw()

            # NEW: Classification Metrics Table
            if hasattr(self, 'metrics_table_container') and self.analyzer and hasattr(self.analyzer, 'ml_engine'):
                # Clear existing table
                for widget in self.metrics_table_container.winfo_children():
                    widget.destroy()

                ml_engine = self.analyzer.ml_engine

                # Check if any predictions have been made
                if not hasattr(ml_engine, 'prediction_history') or len(ml_engine.prediction_history) == 0:
                    # No predictions made yet - show message
                    ctk.CTkLabel(
                        self.metrics_table_container,
                        text="📧 No Email Analysis Yet\n\nClassification metrics will appear after analyzing emails",
                        font=ctk.CTkFont(size=13),
                        text_color=COLORS['text_secondary']
                    ).pack(pady=50)
                elif ml_engine and hasattr(ml_engine, 'model_metrics') and ml_engine.model_metrics:
                    # Create table header
                    header_frame = ctk.CTkFrame(self.metrics_table_container, fg_color=COLORS['accent_primary'])
                    header_frame.pack(fill="x", pady=(0, 2))

                    headers = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
                    widths = [150, 80, 80, 80, 80, 80]

                    for i, (header, width) in enumerate(zip(headers, widths)):
                        ctk.CTkLabel(
                            header_frame,
                            text=header,
                            font=ctk.CTkFont(size=11, weight="bold"),
                            width=width,
                            text_color='#e6edf3'
                        ).grid(row=0, column=i, padx=2, pady=5)

                    # Add rows for each model
                    for idx, (model_name, metrics) in enumerate(ml_engine.model_metrics.items()):
                        row_frame = ctk.CTkFrame(
                            self.metrics_table_container,
                            fg_color=COLORS['bg_secondary'] if idx % 2 == 0 else COLORS['bg_tertiary']
                        )
                        row_frame.pack(fill="x", pady=1)

                        # Model name
                        ctk.CTkLabel(
                            row_frame,
                            text=model_name.replace('_', ' ').title(),
                            font=ctk.CTkFont(size=10),
                            width=widths[0],
                            anchor='w'
                        ).grid(row=0, column=0, padx=5, pady=5)

                        # Metrics
                        metric_values = [
                            f"{metrics.get('accuracy', 0):.1%}",
                            f"{metrics.get('precision', 0):.1%}",
                            f"{metrics.get('recall', 0):.1%}",
                            f"{metrics.get('f1_score', 0):.1%}",
                            f"{metrics.get('roc_auc', 0):.3f}" if metrics.get('roc_auc') else 'N/A'
                        ]

                        for i, (value, width) in enumerate(zip(metric_values, widths[1:]), 1):
                            ctk.CTkLabel(
                                row_frame,
                                text=value,
                                font=ctk.CTkFont(size=10),
                                width=width
                            ).grid(row=0, column=i, padx=2, pady=5)
                else:
                    # No metrics available
                    ctk.CTkLabel(
                        self.metrics_table_container,
                        text="No ML metrics available\nAnalyze emails with ML enabled",
                        font=ctk.CTkFont(size=12),
                        text_color=COLORS['text_secondary']
                    ).pack(pady=50)

            # NEW: Confusion Matrix Heatmap
            if hasattr(self, 'confusion_ax') and self.analyzer and hasattr(self.analyzer, 'ml_engine'):
                self.confusion_ax.clear()

                ml_engine = self.analyzer.ml_engine

                # Check if any predictions have been made
                if not hasattr(ml_engine, 'prediction_history') or len(ml_engine.prediction_history) == 0:
                    # No predictions made yet
                    self.confusion_ax.text(0.5, 0.5, 'No Email Analysis Yet\n\nConfusion matrix will appear after analyzing emails',
                                          ha='center', va='center',
                                          transform=self.confusion_ax.transAxes,
                                          color='#8b949e', fontsize=12)
                    self.confusion_ax.set_title('Confusion Matrix', color='#e6edf3', fontsize=13)
                elif ml_engine and hasattr(ml_engine, 'model_metrics') and ml_engine.model_metrics:
                    # Get ensemble or first model's confusion matrix
                    cm_data = None
                    if 'ensemble' in ml_engine.model_metrics:
                        cm_data = ml_engine.model_metrics['ensemble'].get('confusion_matrix')
                    elif ml_engine.model_metrics:
                        # Use first available model
                        first_model = list(ml_engine.model_metrics.values())[0]
                        cm_data = first_model.get('confusion_matrix')

                    if cm_data:
                        # Create confusion matrix array
                        cm_array = np.array([
                            [cm_data.get('tn', 0), cm_data.get('fp', 0)],
                            [cm_data.get('fn', 0), cm_data.get('tp', 0)]
                        ])

                        # Create heatmap
                        im = self.confusion_ax.imshow(cm_array, cmap='Blues', alpha=0.8)

                        # Add labels
                        self.confusion_ax.set_xticks([0, 1])
                        self.confusion_ax.set_yticks([0, 1])
                        self.confusion_ax.set_xticklabels(['Predicted\nNegative', 'Predicted\nPositive'],
                                                         color='#8b949e', fontsize=10)
                        self.confusion_ax.set_yticklabels(['Actual\nNegative', 'Actual\nPositive'],
                                                         color='#8b949e', fontsize=10)

                        # Add text annotations
                        for i in range(2):
                            for j in range(2):
                                text = self.confusion_ax.text(j, i, cm_array[i, j],
                                                             ha="center", va="center",
                                                             color="#e6edf3", fontsize=16, weight='bold')

                        self.confusion_ax.set_title('Training Confusion Matrix',
                                                   color='#e6edf3', fontsize=13, pad=10)

                        # Add labels to cells
                        labels = [['TN', 'FP'], ['FN', 'TP']]
                        for i in range(2):
                            for j in range(2):
                                self.confusion_ax.text(j, i + 0.3, f'({labels[i][j]})',
                                                      ha="center", va="center",
                                                      color="#888888", fontsize=9)
                    else:
                        self.confusion_ax.text(0.5, 0.5, 'No confusion matrix data',
                                              ha='center', va='center',
                                              transform=self.confusion_ax.transAxes,
                                              color='#8b949e', fontsize=12)
                        self.confusion_ax.set_title('Confusion Matrix', color='#e6edf3', fontsize=13)
                else:
                    self.confusion_ax.text(0.5, 0.5, 'No data available\n\nTrain ML models first',
                                          ha='center', va='center',
                                          transform=self.confusion_ax.transAxes,
                                          color='#8b949e', fontsize=12)
                    self.confusion_ax.set_title('Confusion Matrix', color='#e6edf3', fontsize=13)

                self.confusion_canvas.draw()

        except Exception as e:
            # If there's any error, log it but don't crash
            logger.error(f"Error updating analytics charts: {e}", exc_info=True)

    def update_header_stats(self):
        """Update header statistics"""
        if self.header_stats and self.current_results:
            total = len(self.current_results)
            threats = sum(1 for r in self.current_results if r.get('threats'))

            self.header_stats[0]['value'].configure(text=str(total))
            self.header_stats[1]['value'].configure(text=str(threats))

            if self.analyzer and hasattr(self.analyzer, 'ml_engine'):
                models_count = len(self.analyzer.ml_engine.models)
                self.header_stats[2]['value'].configure(text=str(models_count))

    def log_activity(self, message):
        """Log activity to audit log"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"

        # Add to activity feed
        if hasattr(self, 'activity_feed') and self.activity_feed.winfo_exists():
            try:
                self.activity_feed.insert("1.0", log_entry)

                # Limit size
                content = self.activity_feed.get("1.0", tk.END)
                lines = content.split('\n')
                if len(lines) > 50:
                    self.activity_feed.delete("51.0", tk.END)
            except Exception:
                pass

        # Add to audit log
        if hasattr(self, 'audit_log') and self.audit_log.winfo_exists():
            try:
                self.audit_log.insert("1.0", log_entry)
            except Exception:
                pass

        # Log to file
        logger.info(message)

    def clear_audit_log(self):
        """Clear audit log"""
        if messagebox.askyesno("Confirm", "Are you sure you want to clear the audit log?"):
            if hasattr(self, 'audit_log') and self.audit_log.winfo_exists():
                self.audit_log.delete("1.0", tk.END)
            self.log_activity("Audit log cleared by user")

    def update_status(self, message):
        """Update status bar"""
        if hasattr(self, 'status_label') and self.status_label:
            self.status_label.configure(text=message)

    def generate_report(self):
        """Generate security report"""
        if not self.current_results:
            messagebox.showwarning("No Data",
                                   "No analysis data available.\n\n"
                                   "Please analyze some emails first before generating a report.")
            return

        report_type = self.report_type_var.get()
        date_range = self.date_range_var.get()

        # Update status
        self.update_status(f"Generating {report_type} report for {date_range}...")

        try:
            # Map report type values to friendly names
            report_names = {
                'executive': 'Executive Summary',
                'technical': 'Technical Report',
                'threats': 'Threat Analysis',
                'compliance': 'Compliance Report'
            }
            report_name = report_names.get(report_type, 'Executive Summary')

            # Generate report based on type
            if report_type == "executive":
                self._generate_summary_report(date_range)
            elif report_type == "technical":
                self._generate_detailed_report(date_range)
            elif report_type == "threats":
                self._generate_threat_intel_report(date_range)
            elif report_type == "compliance":
                self._generate_compliance_report(date_range)
            else:
                self._generate_summary_report(date_range)

            self.update_status("Report generated successfully")
            messagebox.showinfo("Success",
                                f"{report_name} for {date_range} has been generated.\n\n"
                                "Use the export buttons below to save the report.")
        except Exception as e:
            logger.error(f"Error generating report: {e}", exc_info=True)
            messagebox.showerror("Error", f"Failed to generate report: {str(e)}")
            self.update_status("Report generation failed")

    def _generate_summary_report(self, date_range):
        """Generate executive summary report with key statistics"""
        stats = {
            'total': len(self.current_results),
            'critical': sum(1 for r in self.current_results if r.get('risk_level') == 'critical'),
            'high': sum(1 for r in self.current_results if r.get('risk_level') == 'high'),
            'medium': sum(1 for r in self.current_results if r.get('risk_level') == 'medium'),
            'low': sum(1 for r in self.current_results if r.get('risk_level') in ['low', 'minimal']),
            'avg_risk': np.mean([(r.get('risk_score') or 0) for r in self.current_results]) if self.current_results else 0,
            'breached': sum(1 for r in self.current_results if (r.get('breach_info') or {}).get('found')),
            'total_threats': sum(len(r.get('threats', [])) for r in self.current_results)
        }

        # Get top 5 highest risk emails for executive summary
        sorted_results = sorted(self.current_results, key=lambda x: x.get('risk_score', 0), reverse=True)
        top_risks = sorted_results[:5]

        # Store report data for export
        self.current_report = {
            'type': 'Executive Summary',
            'report_mode': 'summary',  # Used by export functions
            'date_range': date_range,
            'generated': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'stats': stats,
            'top_risks': top_risks,
            'results': self.current_results,  # Full data for reference
            'include_details': False  # Executive summary = high level only
        }

        logger.info(f"Executive summary report generated with {stats['total']} entries")

    def _generate_detailed_report(self, date_range):
        """Generate technical detailed analysis report - includes EVERYTHING"""
        # Calculate comprehensive statistics
        stats = {
            'total': len(self.current_results),
            'critical': sum(1 for r in self.current_results if r.get('risk_level') == 'critical'),
            'high': sum(1 for r in self.current_results if r.get('risk_level') == 'high'),
            'medium': sum(1 for r in self.current_results if r.get('risk_level') == 'medium'),
            'low': sum(1 for r in self.current_results if r.get('risk_level') in ['low', 'minimal']),
            'avg_risk': np.mean([(r.get('risk_score') or 0) for r in self.current_results]) if self.current_results else 0,
            'breached': sum(1 for r in self.current_results if (r.get('breach_info') or {}).get('found')),
            'total_threats': sum(len(r.get('threats', [])) for r in self.current_results),
            'mitre_techniques': sum(len(r.get('mitre_details', [])) for r in self.current_results)
        }

        self.current_report = {
            'type': 'Technical Report',
            'report_mode': 'detailed',  # Used by export functions
            'date_range': date_range,
            'generated': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'stats': stats,
            'results': self.current_results,  # ALL emails with ALL details
            'include_details': True,  # Include full technical details
            'include_mitre': True,  # Include MITRE ATT&CK details
            'include_dns': True,  # Include DNS security details
            'include_ml': True  # Include ML predictions if available
        }
        logger.info(f"Detailed technical report generated with {stats['total']} entries")

    def _generate_threat_intel_report(self, date_range):
        """Generate threat intelligence focused report - ONLY threats"""
        # Filter for HIGH and CRITICAL risks only
        threats = []
        for result in self.current_results:
            if result.get('risk_level') in ['critical', 'high']:
                threats.append(result)

        # Get unique threat types
        threat_types = set()
        for result in threats:
            for threat in result.get('threats', []):
                threat_types.add(threat.get('type', 'unknown'))

        # Get breach statistics for threats
        breached_threats = sum(1 for r in threats if (r.get('breach_info') or {}).get('found'))

        stats = {
            'total_analyzed': len(self.current_results),
            'threat_count': len(threats),
            'threat_percentage': (len(threats) / len(self.current_results) * 100) if self.current_results else 0,
            'unique_threat_types': len(threat_types),
            'breached_threats': breached_threats
        }

        self.current_report = {
            'type': 'Threat Analysis',
            'report_mode': 'threats',  # Used by export functions
            'date_range': date_range,
            'generated': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'threats': threats,  # ONLY high/critical risks
            'stats': stats,
            'threat_types': list(threat_types),
            'results': threats,  # Export functions will use this filtered list
            'include_details': True,  # Show details for threats
            'focus': 'security_threats'  # Indicate this is threat-focused
        }
        logger.info(f"Threat intelligence report generated with {len(threats)} threats from {len(self.current_results)} total emails")

    def _generate_compliance_report(self, date_range):
        """Generate compliance report - ISO 27001:2022, GDPR, and DNS security compliance"""
        # Calculate comprehensive compliance metrics
        total = len(self.current_results)

        # DNS Security Compliance (Email Authentication Standards)
        dns_compliant = 0
        spf_configured = 0
        dmarc_configured = 0
        dkim_configured = 0
        dnssec_enabled = 0

        # GDPR Article 32 - Security of Processing (Breach Detection)
        breached = 0
        breach_notified = 0  # Emails that need breach notification

        # ISO 27001:2022 - Information Security Controls
        high_risk = 0  # A.5.7 Threat Intelligence
        encrypted = 0  # A.8.24 Cryptographic controls
        access_controlled = 0  # A.5.15 Access control

        # Personal data exposure risk (GDPR Article 5 - Data Protection Principles)
        personal_data_at_risk = 0

        for result in self.current_results:
            dns = result.get('dns_security') or {}

            # DNS/Email Authentication Compliance
            if dns.get('spf'):
                spf_configured += 1
            if dns.get('dmarc'):
                dmarc_configured += 1
            if dns.get('dkim'):
                dkim_configured += 1
            if dns.get('dnssec'):
                dnssec_enabled += 1

            # Full DNS compliance = SPF + DMARC + DKIM
            if dns.get('spf') and dns.get('dmarc') and dns.get('dkim'):
                dns_compliant += 1

            # GDPR Breach Notification Requirements (Article 33 & 34)
            breach_info = result.get('breach_info') or {}
            if breach_info.get('found'):
                breached += 1
                # High severity breaches require notification under GDPR
                if breach_info.get('severity') in ['high', 'critical']:
                    breach_notified += 1
                    personal_data_at_risk += 1

            # ISO 27001:2022 Risk Assessment
            risk_level = result.get('risk_level', 'unknown')
            if risk_level in ['critical', 'high']:
                high_risk += 1
                if not (breach_info.get('found') and breach_info.get('severity') in ['high', 'critical']):
                    personal_data_at_risk += 1  # Only count once per email

            # Assume encrypted if no breach or low risk (simplified)
            if not breach_info.get('found') and risk_level in ['low', 'minimal']:
                encrypted += 1
                access_controlled += 1

        # Calculate compliance scores
        dns_compliance_score = ((dns_compliant / total) * 100) if total > 0 else 0
        spf_coverage = ((spf_configured / total) * 100) if total > 0 else 0
        dmarc_coverage = ((dmarc_configured / total) * 100) if total > 0 else 0
        dkim_coverage = ((dkim_configured / total) * 100) if total > 0 else 0

        # GDPR Compliance Score (inverse of breach rate and data at risk)
        gdpr_score = ((total - personal_data_at_risk) / total * 100) if total > 0 else 0

        # ISO 27001:2022 Security Posture Score
        iso27001_score = ((total - high_risk) / total * 100) if total > 0 else 0

        # Overall Compliance Score (weighted average)
        overall_compliance = (dns_compliance_score * 0.4 + gdpr_score * 0.3 + iso27001_score * 0.3)

        stats = {
            # General Stats
            'total': total,

            # DNS/Email Authentication Compliance
            'dns_compliant': dns_compliant,
            'dns_compliance_rate': dns_compliance_score,
            'spf_configured': spf_configured,
            'spf_coverage': spf_coverage,
            'dmarc_configured': dmarc_configured,
            'dmarc_coverage': dmarc_coverage,
            'dkim_configured': dkim_configured,
            'dkim_coverage': dkim_coverage,
            'dnssec_enabled': dnssec_enabled,

            # GDPR Compliance (Article 32, 33, 34)
            'gdpr_score': gdpr_score,
            'breached': breached,
            'breach_rate': (breached / total * 100) if total > 0 else 0,
            'breach_notified': breach_notified,  # GDPR Article 33/34
            'personal_data_at_risk': personal_data_at_risk,
            'gdpr_data_protection_rate': ((total - personal_data_at_risk) / total * 100) if total > 0 else 0,

            # ISO 27001:2022 Compliance
            'iso27001_score': iso27001_score,
            'high_risk': high_risk,
            'high_risk_rate': (high_risk / total * 100) if total > 0 else 0,
            'encrypted': encrypted,
            'access_controlled': access_controlled,
            'security_control_coverage': ((access_controlled / total * 100)) if total > 0 else 0,

            # Overall Compliance
            'overall_compliance': overall_compliance,
            'passing': dns_compliant,
            'failing': total - dns_compliant,

            # Compliance Status
            'status': 'COMPLIANT' if overall_compliance >= 80 else 'NON-COMPLIANT' if overall_compliance < 50 else 'PARTIAL'
        }

        self.current_report = {
            'type': 'Compliance Report - ISO 27001-2022 & GDPR',
            'report_mode': 'compliance',
            'date_range': date_range,
            'generated': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'stats': stats,
            'results': self.current_results,
            'include_details': True,
            'include_dns': True,
            'include_breaches': True,
            'include_gdpr': True,  # NEW: GDPR compliance details
            'include_iso27001': True,  # NEW: ISO 27001:2022 details
            'compliance_threshold': 80,
            'frameworks': ['ISO 27001:2022', 'GDPR', 'Email Authentication Standards'],
            'focus': 'regulatory_compliance'
        }
        logger.info(f"Compliance report generated: Overall={overall_compliance:.1f}%, DNS={dns_compliance_score:.1f}%, GDPR={gdpr_score:.1f}%, ISO27001={iso27001_score:.1f}%")

    def export_single_report(self, result):
        """Export single analysis report"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[
                ("PDF files", "*.pdf"),
                ("HTML files", "*.html"),
                ("Text files", "*.txt")
            ]
        )

        if filename:
            try:
                if filename.endswith('.html'):
                    html = self.generate_single_html_report(result)
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(html)
                elif filename.endswith('.pdf'):
                    # Generate PDF using available methods
                    self.generate_single_pdf(result, filename)
                else:
                    # For text format, save as text with UTF-8 encoding
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(self.format_report_text(result))

                messagebox.showinfo("Success", f"Report saved to {filename}")
                self.log_activity(f"Exported single report: {filename}")
            except Exception as e:
                logger.error(f"Error exporting single report: {e}")
                messagebox.showerror("Export Error", f"Failed to export report:\n{str(e)}")

    def generate_single_pdf(self, result, filename):
        """Generate PDF for single email analysis"""
        try:
            # Method 1: Try weasyprint (best quality)
            try:
                from weasyprint import HTML
                html_content = self.generate_single_html_report(result)
                HTML(string=html_content).write_pdf(filename)
                logger.info(f"PDF generated using weasyprint: {filename}")
                return
            except ImportError:
                logger.info("weasyprint not available, trying reportlab")
            except Exception as e:
                logger.warning(f"weasyprint failed: {e}, trying reportlab")

            # Method 2: Try reportlab
            try:
                from reportlab.lib.pagesizes import letter
                from reportlab.lib.units import inch
                from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
                from reportlab.lib.enums import TA_LEFT, TA_CENTER
                from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
                from reportlab.lib import colors

                # Create PDF document
                doc = SimpleDocTemplate(filename, pagesize=letter,
                                        leftMargin=0.75*inch, rightMargin=0.75*inch,
                                        topMargin=1*inch, bottomMargin=1*inch)

                story = []
                styles = getSampleStyleSheet()

                # Custom styles
                title_style = ParagraphStyle(
                    'CustomTitle',
                    parent=styles['Heading1'],
                    fontSize=24,
                    textColor=colors.HexColor('#4f8ff7'),
                    alignment=TA_CENTER,
                    spaceAfter=30
                )

                heading_style = ParagraphStyle(
                    'CustomHeading',
                    parent=styles['Heading2'],
                    fontSize=16,
                    textColor=colors.HexColor('#4f8ff7'),
                    spaceAfter=12
                )

                # Title
                story.append(Paragraph("Email Security Analysis Report", title_style))
                story.append(Spacer(1, 20))

                # XML escape helper for reportlab Paragraph (uses XML internally)
                from xml.sax.saxutils import escape as xml_esc

                # Email info
                email = result.get('email', 'N/A')
                risk_level = result.get('risk_level', 'unknown').upper()
                risk_score = result.get('risk_score', 0)

                story.append(Paragraph(f"<b>Email:</b> {xml_esc(str(email))}", styles['Normal']))
                story.append(Paragraph(f"<b>Risk Level:</b> {xml_esc(str(risk_level))}", styles['Normal']))
                story.append(Paragraph(f"<b>Risk Score:</b> {risk_score}/100", styles['Normal']))
                story.append(Spacer(1, 20))

                # Breach Information
                breach_info = result.get('breach_info') or {}
                if breach_info.get('found'):
                    story.append(Paragraph("🚨 Breach Information", heading_style))
                    story.append(Paragraph(f"<b>Breaches Found:</b> {breach_info.get('count', 0)}", styles['Normal']))
                    story.append(Paragraph(f"<b>Severity:</b> {breach_info.get('severity', 'medium').upper()}", styles['Normal']))
                    story.append(Spacer(1, 10))

                    if breach_info.get('details'):
                        for i, breach in enumerate(breach_info['details'][:5], 1):
                            if isinstance(breach, dict):
                                breach_text = f"{i}. <b>{xml_esc(str(breach.get('name', 'Unknown')))}</b>"
                                if breach.get('breach_date'):
                                    breach_text += f" ({xml_esc(str(breach['breach_date']))})"
                                story.append(Paragraph(breach_text, styles['Normal']))

                                if breach.get('domain'):
                                    story.append(Paragraph(f"   Domain: {xml_esc(str(breach['domain']))}", styles['Normal']))
                                if breach.get('data_classes'):
                                    data = ', '.join(xml_esc(str(dc)) for dc in breach['data_classes'][:5])
                                    story.append(Paragraph(f"   Data: {data}", styles['Normal']))
                                story.append(Spacer(1, 5))

                    # Add MITRE Techniques within breach section
                    if result.get('mitre_details'):
                        story.append(Spacer(1, 10))
                        story.append(Paragraph("MITRE ATT&amp;CK Techniques (Related to Breach)", heading_style))
                        for tech in result['mitre_details'][:3]:  # Show top 3
                            tech_text = f"<b>{xml_esc(str(tech.get('id', 'N/A')))}: {xml_esc(str(tech.get('name', 'Unknown')))}</b>"
                            story.append(Paragraph(tech_text, styles['Normal']))
                            story.append(Paragraph(f"   Tactic: {xml_esc(str(tech.get('tactic', 'Unknown')))} | Confidence: {tech.get('similarity', 0):.1f}%", styles['Normal']))
                            story.append(Spacer(1, 8))

                    story.append(Spacer(1, 20))

                # MITRE Techniques (Full List - Separate Section)
                if result.get('mitre_details'):
                    story.append(Paragraph("MITRE ATT&amp;CK Techniques (Complete Analysis)", heading_style))
                    for tech in result['mitre_details'][:5]:
                        tech_text = f"<b>{xml_esc(str(tech.get('id', 'N/A')))}: {xml_esc(str(tech.get('name', 'Unknown')))}</b>"
                        story.append(Paragraph(tech_text, styles['Normal']))
                        story.append(Paragraph(f"Tactic: {xml_esc(str(tech.get('tactic', 'Unknown')))}", styles['Normal']))
                        story.append(Paragraph(f"Confidence: {tech.get('similarity', 0):.1f}%", styles['Normal']))
                        story.append(Spacer(1, 10))

                # Mitigation Steps
                if breach_info.get('mitigation_steps'):
                    story.append(Paragraph("Recommended Actions", heading_style))
                    for i, step in enumerate(breach_info['mitigation_steps'][:5], 1):
                        story.append(Paragraph(f"{i}. {xml_esc(str(step))}", styles['Normal']))
                        story.append(Spacer(1, 5))

                # Build PDF
                doc.build(story)
                logger.info(f"PDF generated using reportlab: {filename}")
                return

            except ImportError:
                logger.warning("reportlab not available")
            except Exception as e:
                logger.error(f"reportlab PDF generation failed: {e}")

            # Method 3: Fallback to HTML (user can print to PDF)
            html_filename = filename.replace('.pdf', '.html')
            html_content = self.generate_single_html_report(result)
            with open(html_filename, 'w', encoding='utf-8') as f:
                f.write(html_content)

            messagebox.showinfo(
                "PDF Export",
                f"PDF libraries not available.\nHTML report saved instead: {html_filename}\n\nYou can open it in a browser and print to PDF."
            )
            logger.info(f"HTML exported as PDF fallback: {html_filename}")

        except Exception as e:
            logger.error(f"Error in generate_single_pdf: {e}")
            raise

    def generate_single_html_report(self, result):
        """Generate HTML report for single analysis"""
        esc = html_mod.escape
        risk_color = {
            'critical': '#f85149',
            'high': '#d29922',
            'medium': '#d29922',
            'low': '#3fb950',
            'minimal': '#4f8ff7'
        }.get(result.get('risk_level', 'unknown'), '#e6edf3')

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Email Security Analysis Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #0f1117 0%, #161b22 100%);
            color: #e6edf3;
            margin: 0;
            padding: 40px;
            min-height: 100vh;
        }}
        .container {{
            max-width: 1000px;
            margin: 0 auto;
            background: rgba(255,255,255,0.05);
            border-radius: 20px;
            padding: 40px;
            backdrop-filter: blur(10px);
            box-shadow: 0 20px 40px rgba(0,0,0,0.5);
        }}
        h1 {{
            background: linear-gradient(90deg, #4f8ff7 0%, #3fb950 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 48px;
            margin-bottom: 20px;
            text-align: center;
        }}
        .risk-score {{
            text-align: center;
            padding: 40px;
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            margin: 30px 0;
        }}
        .score-value {{
            font-size: 72px;
            font-weight: bold;
            color: {risk_color};
        }}
        .risk-level {{
            font-size: 28px;
            color: {risk_color};
            text-transform: uppercase;
            margin-top: 10px;
        }}
        .section {{
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            padding: 30px;
            margin: 20px 0;
        }}
        .section h2 {{
            color: #4f8ff7;
            border-bottom: 2px solid #4f8ff7;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }}
        .threat-item {{
            background: rgba(255,0,0,0.1);
            border-left: 4px solid #f85149;
            padding: 15px;
            margin: 10px 0;
            border-radius: 8px;
        }}
        .recommendation {{
            background: rgba(63,185,80,0.1);
            border-left: 4px solid #3fb950;
            padding: 15px;
            margin: 10px 0;
            border-radius: 8px;
        }}
        .dns-check {{
            display: inline-block;
            padding: 10px 20px;
            margin: 5px;
            border-radius: 8px;
            background: rgba(255,255,255,0.1);
        }}
        .dns-check.active {{
            background: rgba(0,255,136,0.2);
            color: #3fb950;
        }}
        .dns-check.inactive {{
            background: rgba(255,51,102,0.2);
            color: #f85149;
        }}
        .timestamp {{
            text-align: center;
            color: #6e7681;
            margin-top: 40px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Email Security Analysis Report</h1>

        <div class="section">
            <p><strong>Email:</strong> {esc(str(result.get('email', 'N/A')))}</p>
            <p><strong>Analysis Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>Analysis Mode:</strong> {'ML-Enhanced' if self.config.enable_ml else 'Rule-Based'}</p>
        </div>

        <div class="risk-score">
            <div class="score-value">{result.get('risk_score', 0)}/100</div>
            <div class="risk-level">Risk Level: {esc(str(result.get('risk_level', 'unknown')))}</div>
        </div>
"""

        # Threats section
        if result.get('threats'):
            html += """
        <div class="section">
            <h2>🚨 Threats Detected</h2>"""

            for threat in result['threats']:
                html += f"""
            <div class="threat-item">
                <strong>{esc(threat.get('type', '').replace('_', ' ').title())}</strong><br>
                {esc(threat.get('description', 'No description available'))}<br>
                <small>Severity: {esc(threat.get('severity', 'unknown').upper())}</small>
            </div>"""

            html += """
        </div>"""

        # Breach Information Section
        breach_info = result.get('breach_info') or {}
        if breach_info and breach_info.get('found'):
            breach_severity = breach_info.get('severity', 'medium')
            severity_color = {
                'critical': '#f85149',
                'high': '#d29922',
                'medium': '#d29922',
                'low': '#3fb950'
            }.get(breach_severity, '#d29922')

            html += f"""
        <div class="section" style="border: 2px solid {severity_color};">
            <h2 style="color: {severity_color};">🚨 DATA BREACH DETECTED</h2>
            <p style="font-size: 16px; color: {severity_color}; font-weight: bold;">
                This email appears in {breach_info.get('count', 0)} known data breach(es)
            </p>
            <p style="color: #ff6b6b;">
                <strong>Severity:</strong> {esc(breach_severity.upper())}
            </p>"""

            # Breach details
            if breach_info.get('details'):
                html += """
            <h3 style="color: #4f8ff7; margin-top: 30px;">Breach Details:</h3>"""

                for breach in breach_info['details']:
                    if isinstance(breach, dict):
                        html += f"""
            <div class="threat-item" style="background: rgba(255,51,102,0.2); margin: 15px 0;">
                <h4 style="color: #f85149; margin: 0 0 10px 0;">{esc(str(breach.get('title', breach.get('name', 'Unknown Breach'))))}</h4>"""

                        if breach.get('domain'):
                            html += f"""
                <p><strong>Domain/Origin:</strong> {esc(str(breach['domain']))}</p>"""

                        if breach.get('breach_date'):
                            html += f"""
                <p><strong>Breach Date:</strong> {esc(str(breach['breach_date']))}</p>"""

                        if breach.get('description'):
                            html += f"""
                <p><strong>Details:</strong> {esc(str(breach['description']))}</p>"""

                        if isinstance(breach.get('pwn_count'), (int, float)) and breach['pwn_count'] > 0:
                            html += f"""
                <p><strong>Affected Accounts:</strong> {int(breach['pwn_count']):,}</p>"""

                        if breach.get('data_classes'):
                            data_list = ', '.join(esc(str(dc)) for dc in breach['data_classes'][:10])
                            html += f"""
                <p><strong>Compromised Data Types:</strong> {data_list}</p>"""

                        html += """
            </div>"""

            # Mitigation steps
            if breach_info.get('mitigation_steps'):
                html += """
            <h3 style="color: #3fb950; margin-top: 30px;">🛡️ IMMEDIATE ACTIONS REQUIRED:</h3>
            <div style="background: rgba(63,185,80,0.1); padding: 20px; border-radius: 10px; border-left: 4px solid #3fb950;">"""

                for i, step in enumerate(breach_info['mitigation_steps'], 1):
                    html += f"""
                <p style="margin: 10px 0;"><strong>{i}.</strong> {esc(str(step))}</p>"""

                html += """
            </div>"""

            # Add MITRE techniques within breach section
            mitre_in_breach = result.get('mitre_details', [])
            if mitre_in_breach:
                html += """
            <h3 style="color: #58a6ff; margin-top: 30px;">🎯 MITRE ATT&CK TECHNIQUES (RELATED TO BREACH):</h3>
            <div style="background: rgba(88,166,255,0.1); padding: 20px; border-radius: 10px; border-left: 4px solid #58a6ff;">"""

                for technique in mitre_in_breach[:5]:  # Show top 5 techniques
                    similarity = technique.get('similarity', 0)
                    confidence_color = '#3fb950' if similarity > 85 else '#d29922' if similarity > 70 else '#f85149'

                    html += f"""
                <div style="background: rgba(0,0,0,0.3); padding: 15px; margin: 10px 0; border-radius: 8px; border-left: 3px solid {confidence_color};">
                    <div style="color: {confidence_color}; font-weight: bold; font-size: 15px;">
                        {esc(str(technique.get('id', 'N/A')))}: {esc(str(technique.get('name', 'Unknown')))}
                    </div>
                    <div style="color: #8b949e; font-size: 13px; margin-top: 5px;">
                        <strong>Tactic:</strong> {esc(str(technique.get('tactic', 'Unknown')))} |
                        <strong>Confidence:</strong> <span style="color: {confidence_color};">{similarity:.1f}%</span>
                    </div>
                </div>"""

                html += """
            </div>"""

            html += """
        </div>"""

        # MITRE ATT&CK Techniques Section
        mitre_details = result.get('mitre_details', [])
        if mitre_details:
            html += """
        <div class="section">
            <h2 style="color: #4f8ff7;">🎯 MITRE ATT&CK Techniques</h2>
            <p style="color: #8b949e; margin-bottom: 20px;">
                Attack techniques identified using offline MITRE ATT&CK framework with semantic analysis
            </p>"""

            for technique in mitre_details[:10]:  # Show top 10 techniques
                similarity = technique.get('similarity', 0)
                confidence_color = '#3fb950' if similarity > 85 else '#d29922' if similarity > 70 else '#f85149'

                html += f"""
            <div style="background: rgba(255,255,255,0.05); padding: 20px; margin: 15px 0; border-radius: 10px; border-left: 4px solid {confidence_color};">
                <h4 style="color: #4f8ff7; margin: 0 0 10px 0;">
                    {esc(str(technique.get('id', 'N/A')))}: {esc(str(technique.get('name', 'Unknown')))}
                </h4>
                <p style="color: #8b949e; margin: 5px 0;">
                    <strong>Tactic:</strong> {esc(str(technique.get('tactic', 'Unknown')))}
                </p>
                <p style="color: #8b949e; margin: 5px 0;">
                    <strong>Confidence:</strong> <span style="color: {confidence_color};">{similarity:.1f}%</span>
                </p>
                <p style="color: #d0d0d0; margin: 10px 0 0 0; line-height: 1.6;">
                    {esc(str(technique.get('description', 'No description available'))[:300])}{'...' if len(str(technique.get('description', ''))) > 300 else ''}
                </p>
            </div>"""

            html += """
        </div>"""

        # DNS Security
        dns = result.get('dns_security') or {}
        if dns:
            html += """
        <div class="section">
            <h2>🔒 DNS Security Status</h2>
            <div style="text-align: center;">"""

            dns_checks = [
                ('SPF', dns.get('spf', False)),
                ('DMARC', dns.get('dmarc', False)),
                ('DKIM', dns.get('dkim', False)),
                ('DNSSEC', dns.get('dnssec', False)),
                ('BIMI', dns.get('bimi', False)),
                ('MTA-STS', dns.get('mta_sts', False)),
                ('TLS-RPT', dns.get('tls_rpt', False)),
            ]

            for check, status in dns_checks:
                status_class = 'active' if status else 'inactive'
                status_icon = '✅' if status else '❌'
                html += f"""
                <span class="dns-check {status_class}">{status_icon} {check}</span>"""

            html += """
            </div>
        </div>"""

        # Recommendations
        if result.get('recommendations'):
            html += """
        <div class="section">
            <h2>💡 Security Recommendations</h2>"""

            for rec in result['recommendations']:
                html += f"""
            <div class="recommendation">
                {esc(str(rec))}
            </div>"""

            html += """
        </div>"""

        # ML Predictions
        ml_preds = result.get('ml_predictions') or {}
        if ml_preds and self.config.enable_ml:
            html += """
        <div class="section">
            <h2>🤖 Machine Learning Analysis</h2>
            <table style="width: 100%; color: white;">
                <tr style="border-bottom: 1px solid #444;">
                    <th style="text-align: left; padding: 10px;">Model</th>
                    <th style="text-align: right; padding: 10px;">Risk Score</th>
                </tr>"""

            skip_keys = {'ensemble', 'is_malicious', 'precision_threshold', 'anomaly_score'}
            numeric_preds = {k: v for k, v in ml_preds.items()
                            if k not in skip_keys and isinstance(v, (int, float))}
            for model, score in sorted(numeric_preds.items(), key=lambda x: x[1], reverse=True)[:5]:
                color = '#f85149' if score > 0.7 else '#d29922' if score > 0.5 else '#3fb950'
                html += f"""
                <tr>
                    <td style="padding: 10px;">{esc(model.replace('_', ' ').title())}</td>
                    <td style="text-align: right; padding: 10px; color: {color};">{score:.1%}</td>
                </tr>"""

            html += """
            </table>
        </div>"""

        html += f"""
        <div class="timestamp">
            Report generated by Email Security Analyzer Ultimate<br>
            {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
    </div>
</body>
</html>"""

        return html

    def format_report_text(self, result):
        """Format report as text"""
        report = f"""
EMAIL SECURITY ANALYSIS REPORT
==============================

Email: {result.get('email', 'N/A')}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Analysis Mode: {'ML-Enhanced' if self.config.enable_ml else 'Rule-Based'}

RISK ASSESSMENT
--------------
Risk Score: {result.get('risk_score', 0)}/100
Risk Level: {result.get('risk_level', 'unknown').upper()}

"""

        # Breach Information
        breach_info = result.get('breach_info') or {}
        if breach_info and breach_info.get('found'):
            report += "=" * 60 + "\n"
            report += "⚠️  DATA BREACH DETECTED\n"
            report += "=" * 60 + "\n\n"
            report += f"This email appears in {breach_info.get('count', 0)} known data breach(es)\n"
            report += f"Severity: {breach_info.get('severity', 'medium').upper()}\n\n"

            if breach_info.get('details'):
                report += "BREACH DETAILS\n"
                report += "-" * 60 + "\n"
                for breach in breach_info['details']:
                    if isinstance(breach, dict):
                        report += f"\n• Breach: {breach.get('title', breach.get('name', 'Unknown'))}\n"
                        if breach.get('domain'):
                            report += f"  Domain/Origin: {breach['domain']}\n"
                        if breach.get('breach_date'):
                            report += f"  Date: {breach['breach_date']}\n"
                        if breach.get('description'):
                            report += f"  Details: {breach['description']}\n"
                        if isinstance(breach.get('pwn_count'), (int, float)) and breach['pwn_count'] > 0:
                            report += f"  Affected Accounts: {int(breach['pwn_count']):,}\n"
                        if breach.get('data_classes'):
                            report += f"  Compromised Data: {', '.join(str(dc) for dc in breach['data_classes'][:10])}\n"
                        report += "\n"

            if breach_info.get('mitigation_steps'):
                report += "\n🛡️  IMMEDIATE ACTIONS REQUIRED\n"
                report += "-" * 60 + "\n"
                for i, step in enumerate(breach_info['mitigation_steps'], 1):
                    report += f"{i}. {step}\n"
                report += "\n"

        # MITRE ATT&CK Techniques
        mitre_details = result.get('mitre_details', [])
        if mitre_details:
            report += "\n" + "=" * 60 + "\n"
            report += "🎯 MITRE ATT&CK TECHNIQUES\n"
            report += "=" * 60 + "\n\n"
            report += "Attack techniques identified using offline MITRE ATT&CK framework\n"
            report += f"Total Techniques: {len(mitre_details)}\n\n"

            for i, technique in enumerate(mitre_details[:10], 1):
                report += f"{i}. {technique.get('id', 'N/A')}: {technique.get('name', 'Unknown')}\n"
                report += f"   Tactic: {technique.get('tactic', 'Unknown')}\n"
                report += f"   Confidence: {technique.get('similarity', 0):.1f}%\n"
                desc = technique.get('description', 'No description available')
                if len(desc) > 200:
                    desc = desc[:200] + "..."
                report += f"   Description: {desc}\n\n"

        if result.get('threats'):
            report += "THREATS DETECTED\n"
            report += "----------------\n"
            for threat in result['threats']:
                report += f"• {threat.get('type', '').replace('_', ' ').title()}\n"
                report += f"  {threat.get('description', '')}\n"
                report += f"  Severity: {threat.get('severity', 'unknown').upper()}\n\n"

        if result.get('recommendations'):
            report += "RECOMMENDATIONS\n"
            report += "---------------\n"
            for rec in result['recommendations']:
                report += f"• {rec}\n"

        return report

    # Export methods
    def export_csv(self):
        if not self.current_results:
            messagebox.showwarning("Warning", "No data to export")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")]
        )

        if filename:
            try:
                df = pd.DataFrame(self.current_results)
                df.to_csv(filename, index=False)
                messagebox.showinfo("Success", f"Exported to {filename}")
                self.log_activity(f"Exported CSV report: {filename}")
            except Exception as e:
                logger.error(f"Error exporting CSV: {e}")
                messagebox.showerror("Export Error", f"Failed to export CSV:\n{str(e)}")

    def export_excel(self):
        if not EXCEL_AVAILABLE:
            messagebox.showwarning("Warning", "Excel export requires openpyxl")
            return

        if not self.current_results:
            messagebox.showwarning("Warning", "No data to export")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx")]
        )

        if filename:
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # Summary sheet
                breached_count = sum(1 for r in self.current_results if (r.get('breach_info') or {}).get('found'))
                summary_data = {
                    'Metric': ['Total Emails', 'Critical Risk', 'High Risk', 'Medium Risk', 'Low Risk', 'Breached Emails'],
                    'Count': [
                        len(self.current_results),
                        sum(1 for r in self.current_results if r.get('risk_level') == 'critical'),
                        sum(1 for r in self.current_results if r.get('risk_level') == 'high'),
                        sum(1 for r in self.current_results if r.get('risk_level') == 'medium'),
                        sum(1 for r in self.current_results if r.get('risk_level') in ['low', 'minimal']),
                        breached_count
                    ]
                }
                df_summary = pd.DataFrame(summary_data)
                df_summary.to_excel(writer, sheet_name='Summary', index=False)

                # Main results sheet
                main_data = []
                for r in self.current_results:
                    breach_info = r.get('breach_info') or {}
                    main_data.append({
                        'Email': r.get('email', 'N/A'),
                        'Risk Score': r.get('risk_score', 0),
                        'Risk Level': r.get('risk_level', 'unknown').upper(),
                        'Breached': 'YES' if breach_info.get('found') else 'NO',
                        'Breach Count': breach_info.get('count', 0) if breach_info.get('found') else 0,
                        'Breach Severity': breach_info.get('severity', 'N/A').upper() if breach_info.get('found') else 'N/A',
                        'Threats': len(r.get('threats', [])),
                        'DNS Score': (r.get('dns_security') or {}).get('score', 'N/A')
                    })
                df_main = pd.DataFrame(main_data)
                df_main.to_excel(writer, sheet_name='Analysis Results', index=False)

                # Breach Details sheet (NEW!)
                breach_details_data = []
                for r in self.current_results:
                    breach_info = r.get('breach_info') or {}
                    if breach_info.get('found') and breach_info.get('details'):
                        for breach in breach_info['details']:
                            if isinstance(breach, dict):
                                breach_details_data.append({
                                    'Email': r.get('email', 'N/A'),
                                    'Breach Name': breach.get('name', 'Unknown'),
                                    'Breach Date': breach.get('breach_date', 'Unknown'),
                                    'Domain/Origin': breach.get('domain', 'N/A'),
                                    'Description': breach.get('description', 'N/A'),
                                    'Affected Accounts': breach.get('pwn_count', 'N/A'),
                                    'Compromised Data': ', '.join(str(dc) for dc in (breach.get('data_classes') or [])),
                                    'Severity': breach_info.get('severity', 'N/A').upper(),
                                    'Is Verified': breach.get('is_verified', False)
                                })

                if breach_details_data:
                    df_breaches = pd.DataFrame(breach_details_data)
                    df_breaches.to_excel(writer, sheet_name='Breach Details', index=False)

                # Mitigation Steps sheet (NEW!)
                mitigation_data = []
                for r in self.current_results:
                    breach_info = r.get('breach_info') or {}
                    if breach_info.get('found') and breach_info.get('mitigation_steps'):
                        for i, step in enumerate(breach_info['mitigation_steps'], 1):
                            mitigation_data.append({
                                'Email': r.get('email', 'N/A'),
                                'Priority': i,
                                'Action Required': step,
                                'Severity': breach_info.get('severity', 'N/A').upper()
                            })

                if mitigation_data:
                    df_mitigation = pd.DataFrame(mitigation_data)
                    df_mitigation.to_excel(writer, sheet_name='Mitigation Steps', index=False)

                # MITRE Techniques sheet (NEW!)
                mitre_data = []
                for r in self.current_results:
                    if r.get('mitre_details'):
                        for tech in r['mitre_details']:
                            mitre_data.append({
                                'Email': r.get('email', 'N/A'),
                                'Technique ID': tech.get('id', 'N/A'),
                                'Technique Name': tech.get('name', 'N/A'),
                                'Tactic': tech.get('tactic', 'N/A'),
                                'Confidence': f"{tech.get('similarity', 0):.1f}%",
                                'Description': tech.get('description', 'N/A')[:200] + '...' if len(tech.get('description', '')) > 200 else tech.get('description', 'N/A')
                            })

                if mitre_data:
                    df_mitre = pd.DataFrame(mitre_data)
                    df_mitre.to_excel(writer, sheet_name='MITRE Techniques', index=False)

                # Format the workbook
                workbook = writer.book

                # Auto-adjust column widths for all sheets
                for sheet_name in writer.sheets:
                    worksheet = writer.sheets[sheet_name]
                    for column in worksheet.columns:
                        max_length = 0
                        column_letter = column[0].column_letter
                        for cell in column:
                            try:
                                if len(str(cell.value)) > max_length:
                                    max_length = len(str(cell.value))
                            except Exception:
                                pass
                        adjusted_width = min(max_length + 2, 50)  # Cap at 50
                        worksheet.column_dimensions[column_letter].width = adjusted_width

            messagebox.showinfo("Success",
                                f"Excel report exported successfully!\n\n"
                                f"Saved to: {filename}\n\n"
                                f"Sheets included:\n"
                                f"• Summary\n"
                                f"• Analysis Results\n"
                                f"• Breach Details\n"
                                f"• Mitigation Steps\n"
                                f"• MITRE Techniques")
            self.log_activity(f"Exported Excel report: {filename}")

    def export_html(self):
        if not self.current_results:
            messagebox.showwarning("Warning", "No data to export")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".html",
            filetypes=[("HTML files", "*.html")]
        )

        if filename:
            try:
                html = self.generate_full_html_report()
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(html)
                messagebox.showinfo("Success", f"Exported to {filename}")
                self.log_activity(f"Exported HTML report: {filename}")
            except Exception as e:
                logger.error(f"Error exporting HTML: {e}")
                messagebox.showerror("Export Error", f"Failed to export HTML:\n{str(e)}")

    def export_json(self):
        if not self.current_results:
            messagebox.showwarning("Warning", "No data to export")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")]
        )

        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(self.current_results, f, indent=2, default=str, ensure_ascii=False)
                messagebox.showinfo("Success", f"Exported to {filename}")
                self.log_activity(f"Exported JSON report: {filename}")
            except Exception as e:
                logger.error(f"Error exporting JSON: {e}")
                messagebox.showerror("Export Error", f"Failed to export JSON:\n{str(e)}")

    def export_enterprise_html(self):
        """Export beautiful enterprise-level HTML report"""
        # Check if a specific report was generated, otherwise use all results
        if hasattr(self, 'current_report') and self.current_report:
            report_data = self.current_report
            results_to_export = report_data.get('results', self.current_results)
            report_title = report_data.get('type', 'Email Security Report')
            logger.info(f"Exporting {report_title} with {len(results_to_export)} results")
        elif self.current_results:
            results_to_export = self.current_results
            report_title = 'Email Security Analysis'
        else:
            messagebox.showwarning("No Data", "No analysis data available to export.\n\nPlease analyze emails or generate a report first.")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".html",
            filetypes=[("HTML files", "*.html")],
            initialfile=f"Enterprise_{report_title.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        )

        if filename:
            try:
                self.update_status(f"Generating {report_title}...")

                # Import enterprise report generator
                from .EnterpriseReportGenerator import EnterpriseReportGenerator

                generator = EnterpriseReportGenerator(self.config)
                html_content = generator.generate_executive_html_report(
                    results_to_export,  # Use filtered/customized results
                    report_title=report_title
                )

                # Save to file
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(html_content)

                messagebox.showinfo(
                    "Success",
                    f"✅ {report_title} HTML report exported successfully!\n\n"
                    f"📁 Saved to:\n{filename}\n\n"
                    f"📊 Includes ({len(results_to_export)} Emails):\n"
                    f"• Executive summary with KPIs\n"
                    f"• Interactive analytics charts (5 charts)\n"
                    f"• Risk distribution analysis\n"
                    f"• Detailed breach intelligence\n"
                    f"• MITRE ATT&CK techniques\n\n"
                    f"🎨 Features:\n"
                    f"• Professional design & styling\n"
                    f"• Interactive Chart.js visualizations\n"
                    f"• Print-ready layout\n"
                    f"• Responsive design"
                )

                logger.info(f"Enterprise HTML exported: {filename}")
                self.log_activity(f"Exported enterprise HTML: {filename}")
                self.update_status("Enterprise HTML report exported successfully")

            except Exception as e:
                logger.error(f"Error exporting enterprise HTML: {e}")
                messagebox.showerror("Export Error", f"Failed to export enterprise HTML:\n{str(e)}")
                self.update_status("Enterprise HTML export failed")

    def export_enterprise_excel(self):
        """Export beautiful enterprise-level Excel report with charts"""
        # Check if a specific report was generated, otherwise use all results
        if hasattr(self, 'current_report') and self.current_report:
            report_data = self.current_report
            results_to_export = report_data.get('results', self.current_results)
            report_title = report_data.get('type', 'Email Security Report')
            logger.info(f"Exporting Excel {report_title} with {len(results_to_export)} results")
        elif self.current_results:
            results_to_export = self.current_results
            report_title = 'Email Security Analysis'
        else:
            messagebox.showwarning("No Data", "No analysis data available to export.")
            return

        # Check if openpyxl is available
        try:
            import openpyxl
        except ImportError:
            messagebox.showerror(
                "Missing Dependency",
                "Enterprise Excel reports require openpyxl.\n\n"
                "Install with: pip install openpyxl"
            )
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx")],
            initialfile=f"Enterprise_{report_title.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        )

        if filename:
            try:
                self.update_status(f"Generating {report_title} Excel report...")

                # Import enterprise Excel generator
                from .EnterpriseExcelReportGenerator import EnterpriseExcelReportGenerator

                # Pass the REAL analyzer instance so ML metrics use real data
                generator = EnterpriseExcelReportGenerator(self.config, analyzer=self.analyzer)
                generator.generate_enterprise_excel(results_to_export, filename)

                messagebox.showinfo(
                    "Success",
                    f"✅ {report_title} Excel report exported successfully!\n\n"
                    f"📁 Saved to:\n{filename}\n\n"
                    f"📊 Includes 8 Sheets:\n"
                    f"• Executive Dashboard (KPIs & Summary)\n"
                    f"• Risk Analysis (Charts & Distribution)\n"
                    f"• Breach Intelligence (Detailed Data)\n"
                    f"• Detailed Results ({len(results_to_export)} Emails)\n"
                    f"• MITRE ATT&CK Techniques\n"
                    f"• Mitigation Actions (Actionable)\n"
                    f"• DNS Security Analysis\n"
                    f"• ML Prediction Statistics (From Analyzed Emails)\n\n"
                    f"✨ Features:\n"
                    f"• Professional formatting & colors\n"
                    f"• Charts and visualizations\n"
                    f"• Conditional formatting\n"
                    f"• Auto-sized columns\n"
                    f"• Ready for executive presentation"
                )

                logger.info(f"Enterprise Excel exported: {filename}")
                self.log_activity(f"Exported enterprise Excel: {filename}")
                self.update_status("Enterprise Excel report exported successfully")

            except Exception as e:
                logger.error(f"Error exporting enterprise Excel: {e}")
                messagebox.showerror("Export Error", f"Failed to export enterprise Excel:\n{str(e)}")
                self.update_status("Enterprise Excel export failed")

    def export_pdf(self):
        """Export data as PDF"""
        if not self.current_results:
            messagebox.showwarning("No Data", "No analysis data to export.\n\nPlease analyze emails first.")
            return

        # Try multiple PDF generation methods
        try:
            from weasyprint import HTML as WeasyHTML
            WEASYPRINT_AVAILABLE = True
        except ImportError:
            WEASYPRINT_AVAILABLE = False

        # Try reportlab as alternative
        try:
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib import colors
            from reportlab.lib.units import inch
            REPORTLAB_AVAILABLE = True
        except ImportError:
            REPORTLAB_AVAILABLE = False

        filename = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf"), ("HTML files", "*.html"), ("All files", "*.*")]
        )

        if not filename:
            return

        try:
            self.update_status("Generating PDF report...")

            if WEASYPRINT_AVAILABLE:
                # Method 1: Use weasyprint (best quality)
                html_content = self.generate_full_html_report()
                WeasyHTML(string=html_content).write_pdf(filename)
                messagebox.showinfo("Success", f"PDF report exported successfully!\n\nSaved to:\n{filename}")
                logger.info(f"PDF exported using weasyprint: {filename}")

            elif REPORTLAB_AVAILABLE and filename.endswith('.pdf'):
                # Method 2: Use reportlab (good fallback)
                self.generate_reportlab_pdf(filename)
                messagebox.showinfo("Success", f"PDF report exported successfully!\n\nSaved to:\n{filename}")
                logger.info(f"PDF exported using reportlab: {filename}")

            else:
                # Method 3: Save as HTML (user can print to PDF from browser)
                if not filename.endswith('.html'):
                    if filename.endswith('.pdf'):
                        filename = filename[:-4] + '.html'
                    else:
                        filename = filename + '.html'

                html_content = self.generate_full_html_report()
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(html_content)

                messagebox.showinfo("HTML Export",
                                    f"PDF libraries not available. Report saved as HTML.\n\n"
                                    f"Saved to: {filename}\n\n"
                                    f"To generate PDF:\n"
                                    f"1. Open the HTML file in your browser\n"
                                    f"2. Press Ctrl+P to print\n"
                                    f"3. Choose 'Save as PDF'\n\n"
                                    f"Or install PDF library:\n"
                                    f"pip install weasyprint\n"
                                    f"OR\n"
                                    f"pip install reportlab")
                logger.info(f"HTML exported (PDF library unavailable): {filename}")

            self.update_status("Export complete")

        except Exception as e:
            logger.error(f"Error exporting PDF: {e}")
            messagebox.showerror("Export Error", f"Failed to export PDF:\n{str(e)}")
            self.update_status("Export failed")

    def generate_reportlab_pdf(self, filename):
        """Generate PDF using reportlab library"""
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors
        from reportlab.lib.units import inch
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
        from xml.sax.saxutils import escape as xml_esc

        # Create PDF document
        doc = SimpleDocTemplate(filename, pagesize=letter,
                                leftMargin=0.75*inch, rightMargin=0.75*inch,
                                topMargin=1*inch, bottomMargin=1*inch)

        story = []
        styles = getSampleStyleSheet()

        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#4f8ff7'),
            spaceAfter=30,
            alignment=TA_CENTER
        )

        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#4f8ff7'),
            spaceAfter=12,
            spaceBefore=20
        )

        # Title
        story.append(Paragraph("Email Security Analysis Report", title_style))
        story.append(Spacer(1, 0.2*inch))

        # Statistics
        stats = {
            'total': len(self.current_results),
            'critical': sum(1 for r in self.current_results if r.get('risk_level') == 'critical'),
            'high': sum(1 for r in self.current_results if r.get('risk_level') == 'high'),
            'medium': sum(1 for r in self.current_results if r.get('risk_level') == 'medium'),
            'low': sum(1 for r in self.current_results if r.get('risk_level') in ['low', 'minimal']),
            'breached': sum(1 for r in self.current_results if (r.get('breach_info') or {}).get('found'))
        }

        stats_data = [
            ['Total Emails', stats['total']],
            ['Critical Risk', stats['critical']],
            ['High Risk', stats['high']],
            ['Medium Risk', stats['medium']],
            ['Low/Safe', stats['low']],
            ['Breached Emails', stats['breached']]
        ]

        stats_table = Table(stats_data, colWidths=[3*inch, 2*inch])
        stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#161b22')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey)
        ]))

        story.append(stats_table)
        story.append(Spacer(1, 0.3*inch))

        # Results summary table
        story.append(Paragraph("Analysis Results", heading_style))

        table_data = [['Email', 'Risk Score', 'Risk Level', 'Breached']]
        for r in self.current_results[:50]:  # Limit to 50
            breach_marker = 'YES' if (r.get('breach_info') or {}).get('found') else 'NO'
            table_data.append([
                r.get('email', 'N/A')[:40],  # Truncate long emails
                str(r.get('risk_score', 0)),
                r.get('risk_level', 'unknown').upper(),
                breach_marker
            ])

        results_table = Table(table_data, colWidths=[3*inch, 1*inch, 1.2*inch, 1*inch])
        results_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#161b22')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')])
        ]))

        story.append(results_table)

        # Breach details section
        breached_emails = [r for r in self.current_results if (r.get('breach_info') or {}).get('found')]
        if breached_emails:
            story.append(PageBreak())
            story.append(Paragraph("Detailed Breach Information", heading_style))
            story.append(Spacer(1, 0.2*inch))

            for email_result in breached_emails[:20]:  # Limit to 20
                breach_info = email_result.get('breach_info') or {}

                # Email header
                email_para = Paragraph(
                    f"<b>{xml_esc(str(email_result.get('email', 'N/A')))}</b> - {breach_info.get('count', 0)} breach(es) - Severity: {xml_esc(str(breach_info.get('severity', 'medium')).upper())}",
                    styles['Heading3']
                )
                story.append(email_para)
                story.append(Spacer(1, 0.1*inch))

                # Breach details
                if breach_info.get('details'):
                    for i, breach in enumerate(breach_info['details'][:3], 1):
                        if isinstance(breach, dict):
                            breach_text = f"{i}. <b>{xml_esc(str(breach.get('name', 'Unknown')))}</b>"
                            if breach.get('breach_date'):
                                breach_text += f" ({xml_esc(str(breach['breach_date']))})"
                            if breach.get('domain'):
                                breach_text += f"<br/>Domain: {xml_esc(str(breach['domain']))}"
                            if breach.get('data_classes'):
                                data = ', '.join(xml_esc(str(dc)) for dc in breach['data_classes'][:5])
                                breach_text += f"<br/>Data: {data}"

                            story.append(Paragraph(breach_text, styles['Normal']))
                            story.append(Spacer(1, 0.1*inch))

                # Mitigation steps
                if breach_info.get('mitigation_steps'):
                    story.append(Paragraph("<b>Actions Required:</b>", styles['Normal']))
                    for step in breach_info['mitigation_steps'][:3]:
                        story.append(Paragraph(f"&#x2022; {xml_esc(str(step))}", styles['Normal']))
                    story.append(Spacer(1, 0.2*inch))

                story.append(Spacer(1, 0.2*inch))

        # Build PDF
        doc.build(story)

    def export_bulk_csv(self):
        self.export_csv()

    def export_bulk_excel(self):
        self.export_excel()

    def export_bulk_pdf(self):
        self.export_pdf()

    def generate_full_html_report(self):
        """Generate comprehensive HTML report"""
        esc = html_mod.escape
        stats = {
            'total': len(self.current_results),
            'critical': sum(1 for r in self.current_results if r.get('risk_level') == 'critical'),
            'high': sum(1 for r in self.current_results if r.get('risk_level') == 'high'),
            'medium': sum(1 for r in self.current_results if r.get('risk_level') == 'medium'),
            'low': sum(1 for r in self.current_results if r.get('risk_level') in ['low', 'minimal'])
        }

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Email Security Analysis - Full Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #0f1117 0%, #161b22 100%);
            color: #e6edf3;
            margin: 0;
            padding: 40px;
            min-height: 100vh;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        h1 {{
            background: linear-gradient(90deg, #4f8ff7 0%, #3fb950 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 56px;
            text-align: center;
            margin-bottom: 40px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}
        .stat-card {{
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            padding: 30px;
            text-align: center;
            backdrop-filter: blur(10px);
            transition: transform 0.3s;
        }}
        .stat-card:hover {{
            transform: translateY(-5px);
        }}
        .stat-value {{
            font-size: 48px;
            font-weight: bold;
            background: linear-gradient(90deg, #4f8ff7 0%, #3fb950 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .stat-label {{
            font-size: 18px;
            color: #8b949e;
            margin-top: 10px;
        }}
        .results-table {{
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            padding: 30px;
            overflow-x: auto;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th {{
            background: rgba(0,212,255,0.2);
            padding: 15px;
            text-align: left;
            font-weight: bold;
            border-bottom: 2px solid #4f8ff7;
        }}
        td {{
            padding: 15px;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}
        tr:hover {{
            background: rgba(255,255,255,0.05);
        }}
        .critical {{ color: #f85149; font-weight: bold; }}
        .high {{ color: #d29922; font-weight: bold; }}
        .medium {{ color: #d29922; }}
        .low {{ color: #3fb950; }}
        .minimal {{ color: #4f8ff7; }}
        .unknown {{ color: #6e7681; }}
        .footer {{
            text-align: center;
            margin-top: 60px;
            padding: 30px;
            color: #6e7681;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Email Security Analysis Report</h1>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{stats['total']}</div>
                <div class="stat-label">Total Emails Analyzed</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" style="color: #f85149;">{stats['critical']}</div>
                <div class="stat-label">Critical Risk</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" style="color: #d29922;">{stats['high']}</div>
                <div class="stat-label">High Risk</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" style="color: #3fb950;">{stats['low']}</div>
                <div class="stat-label">Safe Emails</div>
            </div>
        </div>

        <div class="results-table">
            <h2 style="color: #4f8ff7; margin-bottom: 20px;">Detailed Analysis Results</h2>
            <table>
                <tr>
                    <th>Email</th>
                    <th>Risk Score</th>
                    <th>Risk Level</th>
                    <th>Threats</th>
                    <th>Breaches</th>
                    <th>DNS Score</th>
                </tr>"""

        valid_levels = {'critical', 'high', 'medium', 'low', 'minimal'}
        for r in self.current_results[:100]:  # Limit to 100 for performance
            level = r.get('risk_level', 'unknown')
            if level not in valid_levels:
                level = 'unknown'
            threats = len(r.get('threats', []))
            breaches = '✓' if (r.get('breach_info') or {}).get('found') else '-'
            dns_score = (r.get('dns_security') or {}).get('score', 'N/A')

            html += f"""
                <tr>
                    <td>{esc(str(r.get('email', 'N/A')))}</td>
                    <td>{r.get('risk_score', 0)}</td>
                    <td class="{level}">{level.upper()}</td>
                    <td>{threats}</td>
                    <td>{breaches}</td>
                    <td>{dns_score}</td>
                </tr>"""

        html += f"""
            </table>
        </div>"""

        # Add detailed breach information section
        breached_emails = [r for r in self.current_results if (r.get('breach_info') or {}).get('found')]
        if breached_emails:
            html += """
        <div class="results-table" style="margin-top: 40px;">
            <h2 style="color: #f85149; margin-bottom: 20px;">🚨 Detailed Breach Information</h2>"""

            for email_result in breached_emails[:50]:  # Limit to 50 for performance
                breach_info = email_result.get('breach_info') or {}
                severity_color = {
                    'critical': '#f85149',
                    'high': '#d29922',
                    'medium': '#d29922',
                    'low': '#3fb950'
                }.get(breach_info.get('severity', 'medium'), '#d29922')

                html += f"""
            <div style="background: rgba(255,255,255,0.05); padding: 25px; margin: 20px 0; border-radius: 10px; border-left: 4px solid {severity_color};">
                <h3 style="color: {severity_color}; margin: 0 0 15px 0;">
                    {esc(str(email_result.get('email', 'N/A')))}
                </h3>
                <p style="color: #8b949e; margin: 5px 0;">
                    <strong>Breach Count:</strong> {breach_info.get('count', 0)} |
                    <strong>Severity:</strong> <span style="color: {severity_color};">{breach_info.get('severity', 'medium').upper()}</span>
                </p>"""

                if breach_info.get('details'):
                    html += """
                <div style="margin-top: 15px;">
                    <strong style="color: #4f8ff7;">Breaches:</strong>"""

                    for breach in breach_info['details'][:5]:  # Show first 5 breaches
                        if isinstance(breach, dict):
                            html += f"""
                    <div style="background: rgba(0,0,0,0.3); padding: 15px; margin: 10px 0; border-radius: 8px;">
                        <div style="color: #f85149; font-weight: bold; margin-bottom: 8px;">
                            {esc(str(breach.get('name', 'Unknown')))}"""
                            if breach.get('breach_date'):
                                html += f" ({esc(str(breach['breach_date']))})"
                            html += """
                        </div>"""

                            if breach.get('domain'):
                                html += f"""
                        <div style="color: #8b949e; font-size: 13px; margin: 3px 0;">
                            <strong>Domain:</strong> {esc(str(breach['domain']))}
                        </div>"""

                            if breach.get('description'):
                                desc_value = breach.get('description', '')
                                desc = desc_value[:200] + "..." if len(desc_value) > 200 else desc_value
                                html += f"""
                        <div style="color: #d0d0d0; font-size: 13px; margin: 3px 0; line-height: 1.4;">
                            {esc(str(desc))}
                        </div>"""

                            if breach.get('data_classes'):
                                data_text = ', '.join(esc(str(dc)) for dc in breach['data_classes'][:8])
                                html += f"""
                        <div style="color: #d29922; font-size: 13px; margin: 3px 0;">
                            <strong>Compromised Data:</strong> {data_text}
                        </div>"""

                            html += """
                    </div>"""

                    html += """
                </div>"""

                if breach_info.get('mitigation_steps'):
                    html += """
                <div style="margin-top: 15px; background: rgba(63,185,80,0.1); padding: 15px; border-radius: 8px; border-left: 3px solid #3fb950;">
                    <strong style="color: #3fb950;">🛡️ Immediate Actions Required:</strong>
                    <ol style="margin: 10px 0; padding-left: 20px; color: #d0d0d0;">"""

                    for step in breach_info['mitigation_steps'][:5]:
                        html += f"""
                        <li style="margin: 5px 0;">{esc(str(step))}</li>"""

                    html += """
                    </ol>
                </div>"""

                # Add MITRE ATT&CK Techniques for this email
                mitre_details = email_result.get('mitre_details', [])
                if mitre_details:
                    html += """
                <div style="margin-top: 15px; background: rgba(88,166,255,0.1); padding: 15px; border-radius: 8px; border-left: 3px solid #58a6ff;">
                    <strong style="color: #58a6ff;">🎯 MITRE ATT&CK Techniques:</strong>"""

                    for technique in mitre_details[:3]:  # Show top 3 techniques
                        similarity = technique.get('similarity', 0)
                        confidence_color = '#3fb950' if similarity > 85 else '#d29922' if similarity > 70 else '#f85149'

                        tech_desc = technique.get('description', '')
                        desc_snippet = ''
                        if tech_desc:
                            truncated = str(tech_desc)[:150] + '...' if len(str(tech_desc)) > 150 else str(tech_desc)
                            desc_snippet = f'<div style="color: #909090; font-size: 11px; margin-top: 3px;">{esc(truncated)}</div>'

                        html += f"""
                    <div style="background: rgba(0,0,0,0.2); padding: 10px; margin: 8px 0; border-radius: 6px; border-left: 3px solid {confidence_color};">
                        <div style="color: {confidence_color}; font-weight: bold; font-size: 13px;">
                            {esc(str(technique.get('id', 'N/A')))}: {esc(str(technique.get('name', 'Unknown')))}
                        </div>
                        <div style="color: #8b949e; font-size: 12px; margin-top: 3px;">
                            Tactic: {esc(str(technique.get('tactic', 'Unknown')))} | Severity: {esc(str(technique.get('severity', 'N/A')).upper())} | Confidence: {similarity:.1f}%
                        </div>
                        {desc_snippet}
                    </div>"""

                    html += """
                </div>"""

                html += """
            </div>"""

            html += """
        </div>"""

        # Advanced Security Checks section
        has_advanced_results = any(
            r.get('dnsbl') or r.get('cert_transparency') or r.get('threatfox') or
            r.get('dga_analysis') or r.get('parked_domain') or r.get('gravatar')
            for r in self.current_results[:50]
        )
        if has_advanced_results:
            html += """
        <div class="results-table" style="margin-top: 40px;">
            <h2 style="color: #4f8ff7; margin-bottom: 20px;">🛡️ Advanced Security Checks</h2>
            <table>
                <tr>
                    <th>Email</th>
                    <th>DNSBL</th>
                    <th>ThreatFox</th>
                    <th>DGA</th>
                    <th>Certs</th>
                    <th>Gravatar</th>
                    <th>Parked</th>
                </tr>"""

            for r in self.current_results[:100]:
                dnsbl = r.get('dnsbl') or {}
                tfox = r.get('threatfox') or {}
                dga = r.get('dga_analysis') or {}
                ct = r.get('cert_transparency') or {}
                grav = r.get('gravatar') or {}
                park = r.get('parked_domain') or {}

                dnsbl_str = f'🚫 {esc(str(dnsbl.get("listed_count", 0)))}' if dnsbl.get('listed') else '✅'
                tfox_str = f'☠️ {esc(str(tfox.get("ioc_count", 0)))}' if tfox.get('found') else '✅'
                dga_score_val = float(dga.get('dga_score') or 0.0)
                dga_str = f'⚠️ {dga_score_val:.0%}' if dga.get('is_dga') else '✅'
                ct_str = esc(str(ct.get('cert_count', 0))) if ct.get('found') else '-'
                grav_str = '👤' if grav.get('has_profile') else '👻'
                park_str = '🅿️' if park.get('is_parked') else '✅'

                html += f"""
                <tr>
                    <td>{esc(str(r.get('email', 'N/A')))}</td>
                    <td>{dnsbl_str}</td>
                    <td>{tfox_str}</td>
                    <td>{dga_str}</td>
                    <td>{ct_str}</td>
                    <td>{grav_str}</td>
                    <td>{park_str}</td>
                </tr>"""

            html += """
            </table>
        </div>"""

        html += f"""
        <div class="footer">
            <p>Generated by Email Security Analyzer Ultimate</p>
            <p>Analysis Mode: {'ML-Enhanced' if self.config.enable_ml else 'Rule-Based'}</p>
            <p>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    </div>
</body>
</html>"""

        return html

    def load_settings(self):
        """Load application settings from file"""
        settings_file = self.config.base_dir / "settings.json"
        if settings_file.exists():
            try:
                with open(settings_file, 'r', encoding='utf-8') as f:
                    settings = json.load(f)

                # Apply settings to config with type conversion
                for key, value in settings.items():
                    if hasattr(self.config, key):
                        # Convert numeric settings to appropriate types
                        if key in ['max_workers', 'batch_size', 'timeout', 'cache_size']:
                            value = int(value)  # Ensure integers
                        setattr(self.config, key, value)

                # Ensure theme fields are consistent and apply immediately
                try:
                    if hasattr(self.config, "theme_dark"):
                        self.config.theme = "dark" if bool(self.config.theme_dark) else "light"
                        ctk.set_appearance_mode(self.config.theme)
                except Exception:
                    pass

                logger.info("Settings loaded successfully")
                return settings
            except Exception as e:
                logger.error(f"Failed to load settings: {e}")
        return {}

    def save_settings(self):
        """Save and apply application settings"""
        settings = {}
        for key, var in self.settings_vars.items():
            value = var.get()
            # Convert numeric settings to appropriate types
            if key in ['max_workers', 'batch_size', 'timeout', 'cache_size']:
                value = int(value)  # These must be integers
            settings[key] = value

        # Apply settings immediately to config
        for key, value in settings.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        # Apply theme change
        if 'theme_dark' in settings:
            ctk.set_appearance_mode("dark" if settings['theme_dark'] else "light")
            try:
                self.config.theme = "dark" if bool(settings["theme_dark"]) else "light"
            except Exception:
                pass

        # Save to file
        settings_file = self.config.base_dir / "settings.json"
        try:
            with open(settings_file, 'w', encoding='utf-8') as f:
                json.dump(settings, f, indent=2, ensure_ascii=False)
            messagebox.showinfo("Success", "Settings saved and applied successfully!\n\nSome settings may require app restart to fully take effect.")
            self.log_activity("Settings updated and applied")
        except Exception as e:
            logger.error(f"Failed to save settings: {e}")
            messagebox.showerror("Save Error", f"Settings applied but could not be saved to disk:\n{str(e)}")
            self.log_activity("Settings applied but not saved (disk error)")

    def start_animations(self):
        """Start UI animations"""

        # Logo animation
        def animate_logo():
            try:
                if not self.root.winfo_exists():
                    return
                if hasattr(self, 'logo_label') and self.logo_label.winfo_exists():
                    icons = ['🛡️', '🔒', '🔐', '🔑']
                    current = self.logo_label.cget("text")
                    if current in icons:
                        current_idx = icons.index(current)
                        next_idx = (current_idx + 1) % len(icons)
                        self.logo_label.configure(text=icons[next_idx])
                    self.root.after(3000, animate_logo)  # only reschedule if widget alive
            except Exception:
                pass

        animate_logo()

        # Connection status animation
        def animate_connection():
            try:
                if not self.root.winfo_exists():
                    return
                if hasattr(self, 'connection_indicator') and self.connection_indicator.winfo_exists():
                    current = self.connection_indicator.cget("text")
                    if "Connected" in current:
                        text = "● Connected"
                        self.connection_indicator.configure(
                            text=text,
                            text_color=COLORS['success']
                        )
                    self.root.after(5000, animate_connection)  # only reschedule if widget alive
            except Exception:
                pass

        animate_connection()

    def run(self):
        """Run the application"""
        self.root.mainloop()
