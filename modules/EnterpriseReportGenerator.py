"""
Enterprise-Level Report Generator
Beautiful, professional reports with charts and comprehensive analysis
"""

import os
import json
import html as html_mod
from datetime import datetime
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class EnterpriseReportGenerator:
    """Generate enterprise-level beautiful reports"""

    def __init__(self, config):
        self.config = config

    @staticmethod
    def _esc(value) -> str:
        """Escape HTML special characters in user-supplied values"""
        return html_mod.escape(str(value)) if value else ''

    @staticmethod
    def _safe_json(value) -> str:
        """Serialize to JSON safe for embedding inside HTML <script> tags.
        json.dumps does not escape </script>, which terminates the script block."""
        return json.dumps(value).replace('</', '<\\/')

    def generate_executive_html_report(self, results: List[Dict], report_title: str = "Email Security Analysis") -> str:
        """Generate stunning enterprise HTML report with executive summary"""

        # Calculate comprehensive statistics
        stats = self._calculate_comprehensive_stats(results)

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self._esc(report_title)} - Executive Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #7c6bf5 0%, #a78bfa 100%);
            color: #1a1a1a;
            line-height: 1.6;
            padding: 0;
            margin: 0;
        }}

        .page {{
            background: white;
            max-width: 1400px;
            margin: 0 auto;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }}

        /* Header Section */
        .header {{
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            padding: 60px 80px;
            position: relative;
            overflow: hidden;
        }}

        .header::before {{
            content: '';
            position: absolute;
            top: 0;
            right: 0;
            width: 500px;
            height: 500px;
            background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
            border-radius: 50%;
            transform: translate(30%, -30%);
        }}

        .header-content {{
            position: relative;
            z-index: 1;
        }}

        .company-logo {{
            font-size: 48px;
            font-weight: 800;
            margin-bottom: 10px;
            letter-spacing: -1px;
        }}

        .report-title {{
            font-size: 56px;
            font-weight: 800;
            margin-bottom: 20px;
            line-height: 1.1;
            background: linear-gradient(90deg, #ffffff 0%, #a8d0e6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}

        .report-subtitle {{
            font-size: 24px;
            font-weight: 300;
            opacity: 0.9;
            margin-bottom: 30px;
        }}

        .report-meta {{
            display: flex;
            gap: 40px;
            font-size: 14px;
            opacity: 0.8;
            margin-top: 30px;
        }}

        .meta-item {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .meta-icon {{
            font-size: 18px;
        }}

        /* Executive Summary */
        .executive-summary {{
            padding: 60px 80px;
            background: linear-gradient(to bottom, #f8f9fa 0%, #ffffff 100%);
            border-bottom: 3px solid #e9ecef;
        }}

        .section-title {{
            font-size: 36px;
            font-weight: 700;
            margin-bottom: 30px;
            color: #1e3c72;
            display: flex;
            align-items: center;
            gap: 15px;
        }}

        .section-icon {{
            font-size: 42px;
        }}

        /* KPI Cards Grid */
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 30px;
            margin: 40px 0;
        }}

        .kpi-card {{
            background: white;
            border-radius: 20px;
            padding: 35px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.08);
            border: 2px solid transparent;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }}

        .kpi-card::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 5px;
            background: linear-gradient(90deg, var(--card-color-start), var(--card-color-end));
        }}

        .kpi-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 15px 40px rgba(0,0,0,0.12);
        }}

        .kpi-card.critical {{
            --card-color-start: #f85149;
            --card-color-end: #ff7b72;
        }}

        .kpi-card.warning {{
            --card-color-start: #d29922;
            --card-color-end: #e3b341;
        }}

        .kpi-card.success {{
            --card-color-start: #3fb950;
            --card-color-end: #00ffb8;
        }}

        .kpi-card.info {{
            --card-color-start: #58a6ff;
            --card-color-end: #7c6bf5;
        }}

        .kpi-label {{
            font-size: 14px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #6c757d;
            margin-bottom: 15px;
        }}

        .kpi-value {{
            font-size: 56px;
            font-weight: 800;
            line-height: 1;
            margin-bottom: 10px;
            background: linear-gradient(135deg, var(--card-color-start), var(--card-color-end));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}

        .kpi-description {{
            font-size: 13px;
            color: #6c757d;
            line-height: 1.4;
        }}

        .kpi-trend {{
            display: inline-flex;
            align-items: center;
            gap: 5px;
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
            margin-top: 10px;
        }}

        .trend-up {{
            background: rgba(255, 51, 102, 0.1);
            color: #f85149;
        }}

        .trend-down {{
            background: rgba(0, 212, 170, 0.1);
            color: #3fb950;
        }}

        /* Content Sections */
        .content-section {{
            padding: 60px 80px;
            border-bottom: 1px solid #e9ecef;
        }}

        .content-section:last-child {{
            border-bottom: none;
        }}

        /* Risk Distribution Chart */
        .risk-distribution {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 40px;
            margin-top: 40px;
        }}

        .risk-breakdown {{
            background: #f8f9fa;
            border-radius: 15px;
            padding: 30px;
        }}

        .risk-item {{
            display: flex;
            align-items: center;
            gap: 20px;
            padding: 20px;
            background: white;
            border-radius: 12px;
            margin-bottom: 15px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        }}

        .risk-color {{
            width: 60px;
            height: 60px;
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 28px;
            font-weight: 700;
            color: white;
        }}

        .risk-color.critical {{ background: linear-gradient(135deg, #f85149, #ff7b72); }}
        .risk-color.high {{ background: linear-gradient(135deg, #d29922, #e3b341); }}
        .risk-color.medium {{ background: linear-gradient(135deg, #d29922, #e3b341); }}
        .risk-color.low {{ background: linear-gradient(135deg, #3fb950, #00ffb8); }}

        .risk-info {{
            flex: 1;
        }}

        .risk-level {{
            font-size: 18px;
            font-weight: 700;
            margin-bottom: 5px;
            text-transform: uppercase;
        }}

        .risk-count {{
            font-size: 14px;
            color: #6c757d;
        }}

        .risk-percentage {{
            font-size: 24px;
            font-weight: 700;
            color: #1e3c72;
        }}

        /* Breach Table */
        .breach-table {{
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            margin-top: 30px;
            background: white;
            border-radius: 15px;
            overflow: hidden;
            box-shadow: 0 5px 20px rgba(0,0,0,0.06);
        }}

        .breach-table thead {{
            background: linear-gradient(135deg, #1e3c72, #2a5298);
            color: white;
        }}

        .breach-table th {{
            padding: 20px 25px;
            text-align: left;
            font-weight: 600;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        .breach-table td {{
            padding: 20px 25px;
            border-bottom: 1px solid #f0f0f0;
        }}

        .breach-table tr:last-child td {{
            border-bottom: none;
        }}

        .breach-table tbody tr:hover {{
            background: #f8f9fa;
        }}

        .email-cell {{
            font-weight: 600;
            color: #1e3c72;
        }}

        .badge {{
            display: inline-block;
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        .badge-critical {{
            background: rgba(255, 51, 102, 0.15);
            color: #f85149;
        }}

        .badge-high {{
            background: rgba(255, 170, 0, 0.15);
            color: #d29922;
        }}

        .badge-medium {{
            background: rgba(255, 136, 0, 0.15);
            color: #d29922;
        }}

        .badge-low {{
            background: rgba(0, 212, 170, 0.15);
            color: #3fb950;
        }}

        .score-badge {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 50px;
            height: 50px;
            border-radius: 50%;
            font-size: 18px;
            font-weight: 700;
            color: white;
        }}

        /* Recommendations */
        .recommendations {{
            background: linear-gradient(135deg, #3fb950 0%, #00ffb8 100%);
            color: white;
            border-radius: 20px;
            padding: 40px;
            margin-top: 40px;
        }}

        .recommendations h3 {{
            font-size: 28px;
            margin-bottom: 25px;
            display: flex;
            align-items: center;
            gap: 15px;
        }}

        .recommendation-list {{
            list-style: none;
        }}

        .recommendation-item {{
            background: rgba(255, 255, 255, 0.15);
            border-left: 5px solid white;
            padding: 20px 25px;
            margin-bottom: 15px;
            border-radius: 10px;
            backdrop-filter: blur(10px);
        }}

        .recommendation-item::before {{
            content: '✓';
            font-weight: 700;
            margin-right: 12px;
            font-size: 18px;
        }}

        /* Footer */
        .footer {{
            background: #1a1a1a;
            color: #ffffff;
            padding: 50px 80px;
            text-align: center;
        }}

        .footer-logo {{
            font-size: 32px;
            font-weight: 800;
            margin-bottom: 15px;
        }}

        .footer-text {{
            font-size: 14px;
            opacity: 0.7;
            margin-bottom: 5px;
        }}

        /* Print Styles */
        @media print {{
            body {{
                background: white;
            }}
            .page {{
                box-shadow: none;
                max-width: 100%;
            }}
            .kpi-card {{
                break-inside: avoid;
            }}
        }}
    </style>
</head>
<body>
    <div class="page">
        <!-- Header -->
        <div class="header">
            <div class="header-content">
                <div class="company-logo">🛡️ AI-MailArmor</div>
                <h1 class="report-title">{self._esc(report_title)}</h1>
                <p class="report-subtitle">Enterprise Security Intelligence Report</p>
                <div class="report-meta">
                    <div class="meta-item">
                        <span class="meta-icon">📅</span>
                        <span>{datetime.now().strftime('%B %d, %Y')}</span>
                    </div>
                    <div class="meta-item">
                        <span class="meta-icon">⏰</span>
                        <span>{datetime.now().strftime('%I:%M %p')}</span>
                    </div>
                    <div class="meta-item">
                        <span class="meta-icon">📊</span>
                        <span>{stats['total']} Emails Analyzed</span>
                    </div>
                    <div class="meta-item">
                        <span class="meta-icon">🔒</span>
                        <span>Confidential</span>
                    </div>
                </div>
            </div>
        </div>

        <!-- Executive Summary -->
        <div class="executive-summary">
            <h2 class="section-title">
                <span class="section-icon">📋</span>
                Executive Summary
            </h2>

            <div class="kpi-grid">
                <div class="kpi-card info">
                    <div class="kpi-label">Total Analyzed</div>
                    <div class="kpi-value">{stats['total']}</div>
                    <div class="kpi-description">Email addresses scanned</div>
                </div>

                <div class="kpi-card critical">
                    <div class="kpi-label">Critical Risks</div>
                    <div class="kpi-value">{stats['critical']}</div>
                    <div class="kpi-description">Require immediate action</div>
                    {f'<div class="kpi-trend trend-up">⚠️ High Priority</div>' if stats['critical'] > 0 else ''}
                </div>

                <div class="kpi-card warning">
                    <div class="kpi-label">Data Breaches</div>
                    <div class="kpi-value">{stats['breached']}</div>
                    <div class="kpi-description">Emails found in known breaches</div>
                    {f'<div class="kpi-trend trend-up">📈 {stats["breach_percentage"]:.1f}%</div>' if stats['breached'] > 0 else ''}
                </div>

                <div class="kpi-card success">
                    <div class="kpi-label">Secure Emails</div>
                    <div class="kpi-value">{stats['safe']}</div>
                    <div class="kpi-description">No threats detected</div>
                    <div class="kpi-trend trend-down">✅ Protected</div>
                </div>

                <div class="kpi-card info">
                    <div class="kpi-label">Avg Risk Score</div>
                    <div class="kpi-value">{stats['avg_risk']}</div>
                    <div class="kpi-description">Out of 100</div>
                </div>

                <div class="kpi-card warning">
                    <div class="kpi-label">High Risk</div>
                    <div class="kpi-value">{stats['high']}</div>
                    <div class="kpi-description">Need attention</div>
                </div>
            </div>
        </div>
"""

        # Add risk distribution section
        html += self._generate_risk_distribution_section(stats)

        # Add enterprise analytics charts section
        html += self._generate_analytics_charts_section(results, stats)

        # Add ML prediction charts section
        html += self._generate_ml_prediction_charts_section(results)

        # Add compliance dashboard if this is a compliance report
        # Check if stats has compliance-specific keys
        if 'gdpr_score' in stats or 'iso27001_score' in stats:
            html += self._generate_compliance_dashboard_section(stats)
            # Add article-level compliance table
            html += self._generate_compliance_details_table(results)

        # Add threat intelligence deep-dive
        html += self._generate_threat_intelligence_section(results)

        # Add disposable email & typosquatting detection section
        html += self._generate_disposable_typosquat_section(results)

        # Add domain intelligence & vulnerability assessment
        html += self._generate_domain_intelligence_section(results)

        # Add advanced security checks section
        html += self._generate_advanced_security_section(results)

        # Add data exposure summary & password breach alerts
        html += self._generate_data_exposure_section(results)

        # Add detailed results table
        html += self._generate_results_table_section(results)

        # Add breach details section
        html += self._generate_breach_details_section(results)

        # Add recommendations
        html += self._generate_recommendations_section(stats)

        # Add footer
        html += f"""
        <!-- Footer -->
        <div class="footer">
            <div class="footer-logo">AI-MailArmor Ultimate</div>
            <div class="footer-text">Enterprise Email Security Intelligence Platform</div>
            <div class="footer-text">Generated {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</div>
            <div class="footer-text" style="margin-top: 20px; font-size: 12px;">
                This report contains confidential information. Distribution is restricted to authorized personnel only.
            </div>
        </div>
    </div>
</body>
</html>"""

        return html

    def _calculate_comprehensive_stats(self, results: List[Dict]) -> Dict:
        """Calculate comprehensive statistics"""
        total = len(results)
        critical = sum(1 for r in results if r.get('risk_level') == 'critical')
        high = sum(1 for r in results if r.get('risk_level') == 'high')
        medium = sum(1 for r in results if r.get('risk_level') == 'medium')
        low = sum(1 for r in results if r.get('risk_level') in ['low', 'minimal'])
        breached = sum(1 for r in results if (r.get('breach_info') or {}).get('found'))

        total_risk = sum(r.get('risk_score', 0) for r in results)
        avg_risk = int(total_risk / total) if total > 0 else 0

        breach_percentage = (breached / total * 100) if total > 0 else 0

        return {
            'total': total,
            'critical': critical,
            'high': high,
            'medium': medium,
            'low': low,
            'safe': low,
            'breached': breached,
            'avg_risk': avg_risk,
            'breach_percentage': breach_percentage,
            'critical_percentage': (critical / total * 100) if total > 0 else 0,
            'high_percentage': (high / total * 100) if total > 0 else 0,
            'medium_percentage': (medium / total * 100) if total > 0 else 0,
            'low_percentage': (low / total * 100) if total > 0 else 0
        }

    def _generate_risk_distribution_section(self, stats: Dict) -> str:
        """Generate risk distribution visualization"""
        return f"""
        <div class="content-section">
            <h2 class="section-title">
                <span class="section-icon">📊</span>
                Risk Distribution Analysis
            </h2>

            <div class="risk-distribution">
                <div class="risk-breakdown">
                    <div class="risk-item">
                        <div class="risk-color critical">{stats['critical']}</div>
                        <div class="risk-info">
                            <div class="risk-level">Critical</div>
                            <div class="risk-count">Immediate action required</div>
                        </div>
                        <div class="risk-percentage">{stats['critical_percentage']:.1f}%</div>
                    </div>

                    <div class="risk-item">
                        <div class="risk-color high">{stats['high']}</div>
                        <div class="risk-info">
                            <div class="risk-level">High</div>
                            <div class="risk-count">Attention needed</div>
                        </div>
                        <div class="risk-percentage">{stats['high_percentage']:.1f}%</div>
                    </div>

                    <div class="risk-item">
                        <div class="risk-color medium">{stats['medium']}</div>
                        <div class="risk-info">
                            <div class="risk-level">Medium</div>
                            <div class="risk-count">Monitor closely</div>
                        </div>
                        <div class="risk-percentage">{stats['medium_percentage']:.1f}%</div>
                    </div>

                    <div class="risk-item">
                        <div class="risk-color low">{stats['low']}</div>
                        <div class="risk-info">
                            <div class="risk-level">Low / Safe</div>
                            <div class="risk-count">No immediate concerns</div>
                        </div>
                        <div class="risk-percentage">{stats['low_percentage']:.1f}%</div>
                    </div>
                </div>

                <div style="background: white; border-radius: 15px; padding: 30px; box-shadow: 0 5px 20px rgba(0,0,0,0.06);">
                    <h3 style="font-size: 22px; margin-bottom: 25px; color: #1e3c72;">Key Findings</h3>
                    <div style="line-height: 2;">
                        <div style="margin-bottom: 15px;">
                            <strong style="color: #1e3c72;">🎯 Overall Risk Level:</strong>
                            <span style="color: {'#f85149' if stats['avg_risk'] > 60 else '#d29922' if stats['avg_risk'] > 40 else '#3fb950'}; font-weight: 700;">{stats['avg_risk']}/100</span>
                        </div>
                        <div style="margin-bottom: 15px;">
                            <strong style="color: #1e3c72;">🚨 Total Breached:</strong>
                            <span style="color: #f85149; font-weight: 700;">{stats['breached']} ({stats['breach_percentage']:.1f}%)</span>
                        </div>
                        <div style="margin-bottom: 15px;">
                            <strong style="color: #1e3c72;">⚡ Action Required:</strong>
                            <span style="color: #f85149; font-weight: 700;">{stats['critical'] + stats['high']} emails</span>
                        </div>
                        <div style="margin-bottom: 15px;">
                            <strong style="color: #1e3c72;">✅ Secure Status:</strong>
                            <span style="color: #3fb950; font-weight: 700;">{stats['low']} emails</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """

    def _generate_analytics_charts_section(self, results: List[Dict], stats: Dict) -> str:
        """Generate comprehensive analytics charts section with Chart.js"""
        from collections import Counter
        import json

        # Prepare data for charts
        # 1. Top Threat Types
        all_threats = []
        for r in results:
            all_threats.extend([t.get('type', 'unknown') for t in (r.get('threats') or [])])
        threat_counts = Counter(all_threats).most_common(10)
        threat_labels = self._safe_json([t[0] for t in threat_counts]) if threat_counts else self._safe_json([])
        threat_values = self._safe_json([t[1] for t in threat_counts]) if threat_counts else self._safe_json([])

        # 2. Breach Statistics by Domain
        domain_breaches = {}
        for r in results:
            email = r.get('email', '')
            if '@' in email:
                domain = email.split('@')[1]
                if domain not in domain_breaches:
                    domain_breaches[domain] = {'total': 0, 'breached': 0}
                domain_breaches[domain]['total'] += 1
                if (r.get('breach_info') or {}).get('found'):
                    domain_breaches[domain]['breached'] += 1

        breach_domains = sorted(domain_breaches.items(), key=lambda x: x[1]['breached'], reverse=True)[:10]
        breach_domain_labels = self._safe_json([d[0] for d in breach_domains]) if breach_domains else self._safe_json([])
        breach_domain_values = self._safe_json([d[1]['breached'] for d in breach_domains]) if breach_domains else self._safe_json([])

        # 3. Risk Score Distribution (histogram)
        score_ranges = self._safe_json(['0-20', '21-40', '41-60', '61-80', '81-100'])
        score_counts = self._safe_json([
            sum(1 for r in results if 0 <= r.get('risk_score', 0) <= 20),
            sum(1 for r in results if 21 <= r.get('risk_score', 0) <= 40),
            sum(1 for r in results if 41 <= r.get('risk_score', 0) <= 60),
            sum(1 for r in results if 61 <= r.get('risk_score', 0) <= 80),
            sum(1 for r in results if 81 <= r.get('risk_score', 0) <= 100)
        ])

        # 4. Top MITRE ATT&CK Techniques (show ID: Name for readability)
        mitre_techniques = []
        for r in results:
            for tech in (r.get('mitre_details') or [])[:5]:
                label = f"{tech.get('id', '?')}: {tech.get('name', 'Unknown')}"
                mitre_techniques.append(label)
        mitre_counts = Counter(mitre_techniques).most_common(10)
        mitre_labels = self._safe_json([t[0] for t in mitre_counts]) if mitre_counts else self._safe_json([])
        mitre_values = self._safe_json([t[1] for t in mitre_counts]) if mitre_counts else self._safe_json([])

        return f"""
        <div class="content-section" style="background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); padding: 50px 80px;">
            <h2 class="section-title" style="margin-bottom: 40px;">
                <span class="section-icon">📊</span>
                Enterprise Analytics Dashboard
            </h2>

            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 30px; margin-bottom: 30px;">
                <!-- Chart 1: Risk Level Distribution (Pie) -->
                <div style="background: white; border-radius: 20px; padding: 30px; box-shadow: 0 10px 30px rgba(0,0,0,0.1);">
                    <h3 style="font-size: 20px; margin-bottom: 20px; color: #1e3c72;">Risk Level Distribution</h3>
                    <canvas id="riskPieChart" style="max-height: 300px;"></canvas>
                </div>

                <!-- Chart 2: Top Threat Types (Bar) -->
                <div style="background: white; border-radius: 20px; padding: 30px; box-shadow: 0 10px 30px rgba(0,0,0,0.1);">
                    <h3 style="font-size: 20px; margin-bottom: 20px; color: #1e3c72;">Top Threat Types</h3>
                    <canvas id="threatBarChart" style="max-height: 300px;"></canvas>
                </div>
            </div>

            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 30px; margin-bottom: 30px;">
                <!-- Chart 3: Breach Statistics by Domain (Bar) -->
                <div style="background: white; border-radius: 20px; padding: 30px; box-shadow: 0 10px 30px rgba(0,0,0,0.1);">
                    <h3 style="font-size: 20px; margin-bottom: 20px; color: #1e3c72;">Breach Statistics by Domain</h3>
                    <canvas id="breachDomainChart" style="max-height: 300px;"></canvas>
                </div>

                <!-- Chart 4: Risk Score Distribution (Histogram) -->
                <div style="background: white; border-radius: 20px; padding: 30px; box-shadow: 0 10px 30px rgba(0,0,0,0.1);">
                    <h3 style="font-size: 20px; margin-bottom: 20px; color: #1e3c72;">Risk Score Distribution</h3>
                    <canvas id="scoreHistChart" style="max-height: 300px;"></canvas>
                </div>
            </div>

            <div style="display: grid; grid-template-columns: 1fr; gap: 30px;">
                <!-- Chart 5: Top MITRE ATT&CK Techniques (Horizontal Bar) -->
                <div style="background: white; border-radius: 20px; padding: 30px; box-shadow: 0 10px 30px rgba(0,0,0,0.1);">
                    <h3 style="font-size: 20px; margin-bottom: 20px; color: #1e3c72;">Top MITRE ATT&CK Techniques Detected</h3>
                    <canvas id="mitreChart" style="max-height: 350px;"></canvas>
                </div>
            </div>
        </div>

        <script>
            // Chart 1: Risk Level Distribution (Pie)
            new Chart(document.getElementById('riskPieChart'), {{
                type: 'doughnut',
                data: {{
                    labels: ['Critical', 'High', 'Medium', 'Low/Safe'],
                    datasets: [{{
                        data: [{stats['critical']}, {stats['high']}, {stats['medium']}, {stats['low']}],
                        backgroundColor: ['#f85149', '#d29922', '#d29922', '#3fb950'],
                        borderWidth: 0
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: true,
                    plugins: {{
                        legend: {{
                            position: 'bottom',
                            labels: {{
                                font: {{ size: 14 }},
                                padding: 15
                            }}
                        }}
                    }}
                }}
            }});

            // Chart 2: Top Threat Types (Bar)
            new Chart(document.getElementById('threatBarChart'), {{
                type: 'bar',
                data: {{
                    labels: {threat_labels},
                    datasets: [{{
                        label: 'Threat Count',
                        data: {threat_values},
                        backgroundColor: '#7c6bf5',
                        borderRadius: 8
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: true,
                    plugins: {{
                        legend: {{ display: false }}
                    }},
                    scales: {{
                        y: {{
                            beginAtZero: true,
                            ticks: {{ font: {{ size: 12 }} }}
                        }},
                        x: {{
                            ticks: {{
                                font: {{ size: 11 }},
                                maxRotation: 45,
                                minRotation: 45
                            }}
                        }}
                    }}
                }}
            }});

            // Chart 3: Breach Statistics by Domain (Bar)
            new Chart(document.getElementById('breachDomainChart'), {{
                type: 'bar',
                data: {{
                    labels: {breach_domain_labels},
                    datasets: [{{
                        label: 'Breached Emails',
                        data: {breach_domain_values},
                        backgroundColor: '#f85149',
                        borderRadius: 8
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: true,
                    plugins: {{
                        legend: {{ display: false }}
                    }},
                    scales: {{
                        y: {{
                            beginAtZero: true,
                            ticks: {{ font: {{ size: 12 }} }}
                        }},
                        x: {{
                            ticks: {{
                                font: {{ size: 11 }},
                                maxRotation: 45,
                                minRotation: 45
                            }}
                        }}
                    }}
                }}
            }});

            // Chart 4: Risk Score Distribution (Histogram)
            new Chart(document.getElementById('scoreHistChart'), {{
                type: 'bar',
                data: {{
                    labels: {score_ranges},
                    datasets: [{{
                        label: 'Email Count',
                        data: {score_counts},
                        backgroundColor: ['#3fb950', '#d29922', '#d29922', '#f85149', '#f85149'],
                        borderRadius: 8
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: true,
                    plugins: {{
                        legend: {{ display: false }}
                    }},
                    scales: {{
                        y: {{
                            beginAtZero: true,
                            ticks: {{ font: {{ size: 12 }} }}
                        }},
                        x: {{
                            ticks: {{ font: {{ size: 12 }} }}
                        }}
                    }}
                }}
            }});

            // Chart 5: Top MITRE ATT&CK Techniques (Horizontal Bar)
            new Chart(document.getElementById('mitreChart'), {{
                type: 'bar',
                data: {{
                    labels: {mitre_labels},
                    datasets: [{{
                        label: 'Detection Count',
                        data: {mitre_values},
                        backgroundColor: '#a78bfa',
                        borderRadius: 6
                    }}]
                }},
                options: {{
                    indexAxis: 'y',
                    responsive: true,
                    maintainAspectRatio: true,
                    plugins: {{
                        legend: {{ display: false }}
                    }},
                    scales: {{
                        x: {{
                            beginAtZero: true,
                            ticks: {{ font: {{ size: 12 }} }}
                        }},
                        y: {{
                            ticks: {{ font: {{ size: 12 }} }}
                        }}
                    }}
                }}
            }});
        </script>
        """

    def _generate_ml_prediction_charts_section(self, results: List[Dict]) -> str:
        """Generate ML prediction analysis charts"""
        import json
        from collections import Counter
        import numpy as np

        # Helper function to convert numpy types to Python types
        def to_python_type(value):
            """Convert numpy types to Python native types for JSON serialization"""
            if isinstance(value, (np.integer, np.int32, np.int64)):
                return int(value)
            elif isinstance(value, (np.floating, np.float32, np.float64)):
                return float(value)
            elif isinstance(value, np.ndarray):
                return value.tolist()
            else:
                return value

        # Extract ML prediction data — dynamically discover model names from results
        skip_keys = {'ensemble', 'is_malicious', 'precision_threshold', 'anomaly_score'}
        ml_data = {}

        for r in results:
            ml_preds = r.get('ml_predictions') or {}
            for model, val in ml_preds.items():
                if model in skip_keys:
                    continue
                val = to_python_type(val)
                if val is not None and isinstance(val, (int, float)):
                    if model not in ml_data:
                        ml_data[model] = {'scores': []}
                    ml_data[model]['scores'].append(float(val))

        for model in ml_data:
            scores = ml_data[model]['scores']
            ml_data[model] = {
                'avg': sum(scores) / len(scores),
                'min': min(scores),
                'max': max(scores),
                'count': len(scores)
            }

        # Prepare data for model comparison chart
        model_names = []
        model_scores = []
        for model, data in ml_data.items():
            model_names.append(model.replace('_', ' ').title())
            # Ensure it's a Python float, not numpy
            model_scores.append(round(float(data['avg'] * 100), 1))

        model_names_json = self._safe_json(model_names)
        model_scores_json = self._safe_json(model_scores)

        # Prepare data for prediction confidence distribution
        confidence_ranges = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
        confidence_counts = [0, 0, 0, 0, 0]

        for r in results:
            ensemble = (r.get('ml_predictions') or {}).get('ensemble') or 0
            # Convert to Python float
            ensemble = float(to_python_type(ensemble) or 0)
            score_pct = ensemble * 100
            if score_pct <= 20:
                confidence_counts[0] += 1
            elif score_pct <= 40:
                confidence_counts[1] += 1
            elif score_pct <= 60:
                confidence_counts[2] += 1
            elif score_pct <= 80:
                confidence_counts[3] += 1
            else:
                confidence_counts[4] += 1

        confidence_ranges_json = self._safe_json(confidence_ranges)
        confidence_counts_json = self._safe_json(confidence_counts)

        # Count ML model accuracy by risk level
        risk_accuracy = {'critical': [], 'high': [], 'medium': [], 'low': []}
        for r in results:
            risk_level = r.get('risk_level', 'low')
            ensemble = (r.get('ml_predictions') or {}).get('ensemble') or 0
            # Convert to Python float
            ensemble = float(to_python_type(ensemble) or 0)
            if risk_level in risk_accuracy:
                risk_accuracy[risk_level].append(ensemble * 100)

        # Average accuracy per risk level
        risk_levels = []
        avg_predictions = []
        for level in ['critical', 'high', 'medium', 'low']:
            if risk_accuracy[level]:
                risk_levels.append(level.title())
                # Ensure Python float
                avg = float(sum(risk_accuracy[level]) / len(risk_accuracy[level]))
                avg_predictions.append(round(avg, 1))

        risk_levels_json = self._safe_json(risk_levels)
        avg_predictions_json = self._safe_json(avg_predictions)

        return f"""
        <div class="content-section" style="background: linear-gradient(135deg, #7c6bf5 0%, #a78bfa 100%); padding: 50px 80px; color: white;">
            <h2 class="section-title" style="color: white; margin-bottom: 40px;">
                <span class="section-icon">🧠</span>
                Machine Learning Prediction Analysis
            </h2>

            <div style="background: rgba(255,255,255,0.1); border-radius: 20px; padding: 30px; margin-bottom: 30px; backdrop-filter: blur(10px);">
                <p style="font-size: 16px; line-height: 1.8; margin-bottom: 20px;">
                    This section shows how our <strong>precision-optimized ML models</strong> (XGBoost + Random Forest) analyzed your emails.
                    Both models are calibrated and combined using <strong>weighted ensemble learning</strong> with
                    an Isolation Forest for unsupervised anomaly detection.
                </p>
                <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin-top: 20px;">
                    <div style="text-align: center; background: rgba(255,255,255,0.15); padding: 15px; border-radius: 10px;">
                        <div style="font-size: 32px; font-weight: 700;">2+1</div>
                        <div style="font-size: 13px; opacity: 0.9;">ML Models + Anomaly</div>
                    </div>
                    <div style="text-align: center; background: rgba(255,255,255,0.15); padding: 15px; border-radius: 10px;">
                        <div style="font-size: 32px; font-weight: 700;">{len(results)}</div>
                        <div style="font-size: 13px; opacity: 0.9;">Emails Analyzed</div>
                    </div>
                    <div style="text-align: center; background: rgba(255,255,255,0.15); padding: 15px; border-radius: 10px;">
                        <div style="font-size: 32px; font-weight: 700;">99.6%</div>
                        <div style="font-size: 13px; opacity: 0.9;">Precision Target</div>
                    </div>
                    <div style="text-align: center; background: rgba(255,255,255,0.15); padding: 15px; border-radius: 10px;">
                        <div style="font-size: 32px; font-weight: 700;">44</div>
                        <div style="font-size: 13px; opacity: 0.9;">Features Analyzed</div>
                    </div>
                </div>
            </div>

            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 30px; margin-bottom: 30px;">
                <!-- Chart 1: ML Model Performance Comparison -->
                <div style="background: white; border-radius: 20px; padding: 30px; box-shadow: 0 10px 30px rgba(0,0,0,0.1);">
                    <h3 style="font-size: 20px; margin-bottom: 20px; color: #1e3c72;">ML Model Performance Comparison</h3>
                    <canvas id="mlModelChart" style="max-height: 300px;"></canvas>
                </div>

                <!-- Chart 2: Prediction Confidence Distribution -->
                <div style="background: white; border-radius: 20px; padding: 30px; box-shadow: 0 10px 30px rgba(0,0,0,0.1);">
                    <h3 style="font-size: 20px; margin-bottom: 20px; color: #1e3c72;">Prediction Confidence Distribution</h3>
                    <canvas id="confidenceChart" style="max-height: 300px;"></canvas>
                </div>
            </div>

            <div style="display: grid; grid-template-columns: 1fr; gap: 30px;">
                <!-- Chart 3: ML Predictions by Risk Level -->
                <div style="background: white; border-radius: 20px; padding: 30px; box-shadow: 0 10px 30px rgba(0,0,0,0.1);">
                    <h3 style="font-size: 20px; margin-bottom: 20px; color: #1e3c72;">Average ML Prediction Score by Risk Level</h3>
                    <canvas id="riskPredictionChart" style="max-height: 300px;"></canvas>
                </div>
            </div>

            <!-- ML Models Legend -->
            <div style="background: rgba(255,255,255,0.1); border-radius: 20px; padding: 30px; margin-top: 30px; backdrop-filter: blur(10px);">
                <h3 style="font-size: 20px; margin-bottom: 20px;">Active Machine Learning Models</h3>
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
                    <div style="display: flex; align-items: center; gap: 10px;">
                        <span style="font-size: 24px;">🌳</span>
                        <div>
                            <div style="font-weight: 600;">Random Forest (Calibrated)</div>
                            <div style="font-size: 13px; opacity: 0.8;">Hyperparameter-tuned ensemble of decision trees</div>
                        </div>
                    </div>
                    <div style="display: flex; align-items: center; gap: 10px;">
                        <span style="font-size: 24px;">🚀</span>
                        <div>
                            <div style="font-weight: 600;">XGBoost (Calibrated)</div>
                            <div style="font-size: 13px; opacity: 0.8;">Extreme gradient boosting with isotonic calibration</div>
                        </div>
                    </div>
                    <div style="display: flex; align-items: center; gap: 10px;">
                        <span style="font-size: 24px;">🔍</span>
                        <div>
                            <div style="font-weight: 600;">Isolation Forest</div>
                            <div style="font-size: 13px; opacity: 0.8;">Unsupervised anomaly detection on legitimate patterns</div>
                        </div>
                    </div>
                    <div style="display: flex; align-items: center; gap: 10px;">
                        <span style="font-size: 24px;">🎯</span>
                        <div>
                            <div style="font-weight: 600;">Weighted Ensemble</div>
                            <div style="font-size: 13px; opacity: 0.8;">F1-weighted combination with precision-optimized threshold</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            // Chart 1: ML Model Performance Comparison
            new Chart(document.getElementById('mlModelChart'), {{
                type: 'bar',
                data: {{
                    labels: {model_names_json},
                    datasets: [{{
                        label: 'Average Risk Score (%)',
                        data: {model_scores_json},
                        backgroundColor: [
                            '#3fb950', '#4f8ff7', '#d29922', '#f85149',
                            '#a78bfa', '#58a6ff', '#e3b341'
                        ],
                        borderRadius: 8
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: true,
                    plugins: {{
                        legend: {{ display: false }},
                        title: {{
                            display: true,
                            text: 'Higher scores indicate better threat detection capability'
                        }}
                    }},
                    scales: {{
                        y: {{
                            beginAtZero: true,
                            max: 100,
                            ticks: {{
                                callback: function(value) {{ return value + '%'; }}
                            }}
                        }}
                    }}
                }}
            }});

            // Chart 2: Prediction Confidence Distribution
            new Chart(document.getElementById('confidenceChart'), {{
                type: 'doughnut',
                data: {{
                    labels: {confidence_ranges_json},
                    datasets: [{{
                        data: {confidence_counts_json},
                        backgroundColor: [
                            '#3fb950', '#56d364', '#e3b341', '#d29922', '#f85149'
                        ],
                        borderWidth: 0
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: true,
                    plugins: {{
                        legend: {{
                            position: 'bottom',
                            labels: {{
                                padding: 15,
                                font: {{ size: 13 }}
                            }}
                        }},
                        title: {{
                            display: true,
                            text: 'Distribution of ML confidence scores across analyzed emails'
                        }}
                    }}
                }}
            }});

            // Chart 3: ML Predictions by Risk Level
            new Chart(document.getElementById('riskPredictionChart'), {{
                type: 'bar',
                data: {{
                    labels: {risk_levels_json},
                    datasets: [{{
                        label: 'Average ML Prediction Score',
                        data: {avg_predictions_json},
                        backgroundColor: ['#f85149', '#d29922', '#e3b341', '#3fb950'],
                        borderRadius: 10
                    }}]
                }},
                options: {{
                    indexAxis: 'y',
                    responsive: true,
                    maintainAspectRatio: true,
                    plugins: {{
                        legend: {{ display: false }},
                        title: {{
                            display: true,
                            text: 'How ML models scored emails in each risk category'
                        }}
                    }},
                    scales: {{
                        x: {{
                            beginAtZero: true,
                            max: 100,
                            ticks: {{
                                callback: function(value) {{ return value + '%'; }}
                            }}
                        }}
                    }}
                }}
            }});
        </script>
        """

    def _generate_compliance_dashboard_section(self, stats: Dict) -> str:
        """Generate ISO 27001:2022 & GDPR compliance dashboard"""
        overall_score = stats.get('overall_compliance', 0)
        gdpr_score = stats.get('gdpr_score', 0)
        iso_score = stats.get('iso27001_score', 0)
        dns_score = stats.get('dns_compliance_rate', 0)

        # Determine compliance status color
        status = stats.get('status', 'UNKNOWN')
        status_color = '#3fb950' if status == 'COMPLIANT' else '#d29922' if status == 'PARTIAL' else '#f85149'

        return f"""
        <div class="content-section" style="background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); padding: 50px 80px; color: white;">
            <h2 class="section-title" style="color: white; margin-bottom: 40px;">
                <span class="section-icon">📋</span>
                Regulatory Compliance Dashboard
            </h2>

            <!-- Overall Compliance Status -->
            <div style="background: rgba(255,255,255,0.1); border-radius: 20px; padding: 40px; margin-bottom: 40px; backdrop-filter: blur(10px);">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <h3 style="font-size: 28px; margin-bottom: 10px;">Overall Compliance Status</h3>
                        <p style="font-size: 16px; opacity: 0.9;">Assessed against ISO 27001:2022, GDPR, and Email Authentication Standards</p>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-size: 72px; font-weight: 800; color: {status_color};">{overall_score:.1f}%</div>
                        <div style="font-size: 24px; font-weight: 600; color: {status_color}; margin-top: 10px;">{status}</div>
                    </div>
                </div>
            </div>

            <!-- Framework Scores Grid -->
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 30px; margin-bottom: 40px;">
                <!-- ISO 27001:2022 -->
                <div style="background: rgba(255,255,255,0.15); border-radius: 20px; padding: 30px; text-align: center; backdrop-filter: blur(10px);">
                    <div style="font-size: 18px; font-weight: 600; margin-bottom: 15px; opacity: 0.9;">ISO 27001:2022</div>
                    <div style="font-size: 48px; font-weight: 800; margin-bottom: 10px;">{iso_score:.1f}%</div>
                    <div style="font-size: 14px; opacity: 0.8;">Information Security Management</div>
                    <div style="margin-top: 20px; padding-top: 20px; border-top: 1px solid rgba(255,255,255,0.2);">
                        <div style="font-size: 13px; text-align: left; line-height: 1.8;">
                            ✓ A.5.7 Threat Intelligence<br>
                            ✓ A.5.15 Access Control<br>
                            ✓ A.8.24 Cryptographic Controls
                        </div>
                    </div>
                </div>

                <!-- GDPR -->
                <div style="background: rgba(255,255,255,0.15); border-radius: 20px; padding: 30px; text-align: center; backdrop-filter: blur(10px);">
                    <div style="font-size: 18px; font-weight: 600; margin-bottom: 15px; opacity: 0.9;">GDPR</div>
                    <div style="font-size: 48px; font-weight: 800; margin-bottom: 10px;">{gdpr_score:.1f}%</div>
                    <div style="font-size: 14px; opacity: 0.8;">Data Protection Regulation</div>
                    <div style="margin-top: 20px; padding-top: 20px; border-top: 1px solid rgba(255,255,255,0.2);">
                        <div style="font-size: 13px; text-align: left; line-height: 1.8;">
                            ✓ Article 5 Data Protection<br>
                            ✓ Article 32 Security Processing<br>
                            ✓ Article 33/34 Breach Notification
                        </div>
                    </div>
                </div>

                <!-- Email Authentication -->
                <div style="background: rgba(255,255,255,0.15); border-radius: 20px; padding: 30px; text-align: center; backdrop-filter: blur(10px);">
                    <div style="font-size: 18px; font-weight: 600; margin-bottom: 15px; opacity: 0.9;">Email Authentication</div>
                    <div style="font-size: 48px; font-weight: 800; margin-bottom: 10px;">{dns_score:.1f}%</div>
                    <div style="font-size: 14px; opacity: 0.8;">DNS Security Standards</div>
                    <div style="margin-top: 20px; padding-top: 20px; border-top: 1px solid rgba(255,255,255,0.2);">
                        <div style="font-size: 13px; text-align: left; line-height: 1.8;">
                            ✓ SPF Configuration<br>
                            ✓ DMARC Policy<br>
                            ✓ DKIM Signatures
                        </div>
                    </div>
                </div>
            </div>

            <!-- Detailed Metrics -->
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 30px;">
                <!-- GDPR Metrics -->
                <div style="background: rgba(255,255,255,0.1); border-radius: 20px; padding: 30px; backdrop-filter: blur(10px);">
                    <h4 style="font-size: 20px; margin-bottom: 25px;">GDPR Compliance Metrics</h4>
                    <div style="line-height: 2.2;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                            <span>📊 Total Emails Assessed:</span>
                            <strong>{stats.get('total', 0)}</strong>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                            <span>🚨 Data Breaches Detected:</span>
                            <strong style="color: #ff6b6b;">{stats.get('breached', 0)} ({stats.get('breach_rate', 0):.1f}%)</strong>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                            <span>📢 Breach Notifications Required:</span>
                            <strong style="color: #d29922;">{stats.get('breach_notified', 0)}</strong>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                            <span>⚠️ Personal Data at Risk:</span>
                            <strong style="color: #ff6b6b;">{stats.get('personal_data_at_risk', 0)}</strong>
                        </div>
                        <div style="display: flex; justify-content: space-between; padding-top: 15px; border-top: 1px solid rgba(255,255,255,0.2);">
                            <span>✅ Data Protection Rate:</span>
                            <strong style="color: #3fb950;">{stats.get('gdpr_data_protection_rate', 0):.1f}%</strong>
                        </div>
                    </div>
                </div>

                <!-- ISO 27001:2022 Metrics -->
                <div style="background: rgba(255,255,255,0.1); border-radius: 20px; padding: 30px; backdrop-filter: blur(10px);">
                    <h4 style="font-size: 20px; margin-bottom: 25px;">ISO 27001:2022 Security Controls</h4>
                    <div style="line-height: 2.2;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                            <span>🎯 High Risk Threats (A.5.7):</span>
                            <strong style="color: #ff6b6b;">{stats.get('high_risk', 0)} ({stats.get('high_risk_rate', 0):.1f}%)</strong>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                            <span>🔐 Access Controlled (A.5.15):</span>
                            <strong style="color: #3fb950;">{stats.get('access_controlled', 0)}</strong>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                            <span>🔒 Encrypted Accounts (A.8.24):</span>
                            <strong style="color: #3fb950;">{stats.get('encrypted', 0)}</strong>
                        </div>
                        <div style="display: flex; justify-content: space-between; padding-top: 15px; border-top: 1px solid rgba(255,255,255,0.2);">
                            <span>✅ Security Control Coverage:</span>
                            <strong style="color: #3fb950;">{stats.get('security_control_coverage', 0):.1f}%</strong>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Email Authentication Details -->
            <div style="background: rgba(255,255,255,0.1); border-radius: 20px; padding: 30px; margin-top: 30px; backdrop-filter: blur(10px);">
                <h4 style="font-size: 20px; margin-bottom: 25px;">Email Authentication Standards Coverage</h4>
                <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 25px;">
                    <div style="text-align: center;">
                        <div style="font-size: 14px; margin-bottom: 10px; opacity: 0.9;">SPF Records</div>
                        <div style="font-size: 36px; font-weight: 700; margin-bottom: 5px;">{stats.get('spf_configured', 0)}</div>
                        <div style="font-size: 13px; opacity: 0.8;">{stats.get('spf_coverage', 0):.1f}% coverage</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-size: 14px; margin-bottom: 10px; opacity: 0.9;">DMARC Policies</div>
                        <div style="font-size: 36px; font-weight: 700; margin-bottom: 5px;">{stats.get('dmarc_configured', 0)}</div>
                        <div style="font-size: 13px; opacity: 0.8;">{stats.get('dmarc_coverage', 0):.1f}% coverage</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-size: 14px; margin-bottom: 10px; opacity: 0.9;">DKIM Signatures</div>
                        <div style="font-size: 36px; font-weight: 700; margin-bottom: 5px;">{stats.get('dkim_configured', 0)}</div>
                        <div style="font-size: 13px; opacity: 0.8;">{stats.get('dkim_coverage', 0):.1f}% coverage</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-size: 14px; margin-bottom: 10px; opacity: 0.9;">DNSSEC Enabled</div>
                        <div style="font-size: 36px; font-weight: 700; margin-bottom: 5px;">{stats.get('dnssec_enabled', 0)}</div>
                        <div style="font-size: 13px; opacity: 0.8;">domains secured</div>
                    </div>
                </div>
            </div>

            <!-- Compliance Recommendations -->
            <div style="background: rgba(255,255,255,0.1); border-radius: 20px; padding: 30px; margin-top: 30px; backdrop-filter: blur(10px);">
                <h4 style="font-size: 20px; margin-bottom: 20px;">🎯 Compliance Recommendations</h4>
                <div style="line-height: 2;">
                    {'<div style="padding: 12px 0; border-bottom: 1px solid rgba(255,255,255,0.1);">🚨 <strong>GDPR Article 33:</strong> ' + str(stats.get('breach_notified', 0)) + ' breach(es) require notification to supervisory authority within 72 hours</div>' if stats.get('breach_notified', 0) > 0 else ''}
                    {'<div style="padding: 12px 0; border-bottom: 1px solid rgba(255,255,255,0.1);">⚠️ <strong>ISO 27001 A.5.7:</strong> ' + str(stats.get('high_risk', 0)) + ' high-risk threat(s) identified - implement threat intelligence controls</div>' if stats.get('high_risk', 0) > 0 else ''}
                    <div style="padding: 12px 0; border-bottom: 1px solid rgba(255,255,255,0.1);">📧 <strong>Email Authentication:</strong> Configure SPF, DMARC, and DKIM for all {stats.get('failing', 0)} non-compliant domains</div>
                    <div style="padding: 12px 0;">✅ <strong>Best Practice:</strong> Maintain 80%+ compliance across all frameworks for optimal security posture</div>
                </div>
            </div>
        </div>
        """

    def _generate_threat_intelligence_section(self, results: List[Dict]) -> str:
        """Generate advanced threat intelligence deep-dive with per-email threat breakdown"""
        from collections import Counter
        import json

        # Collect all threats across all emails
        all_threats = []
        threat_severity_map = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
        threat_type_details = {}

        for r in results:
            for threat in (r.get('threats') or []):
                t_type = threat.get('type', 'unknown')
                severity = threat.get('severity', 'low')
                threat_severity_map[severity] = threat_severity_map.get(severity, 0) + 1
                all_threats.append(t_type)

                if t_type not in threat_type_details:
                    threat_type_details[t_type] = {
                        'description': threat.get('description', ''),
                        'severity': severity,
                        'count': 0,
                        'emails': [],
                        'avg_confidence': []
                    }
                threat_type_details[t_type]['count'] += 1
                threat_type_details[t_type]['emails'].append(r.get('email', 'N/A'))
                if threat.get('confidence'):
                    threat_type_details[t_type]['avg_confidence'].append(threat['confidence'])

        if not all_threats:
            return ""

        # Sort by count descending
        sorted_threats = sorted(threat_type_details.items(), key=lambda x: x[1]['count'], reverse=True)

        # Threat type icons
        threat_icons = {
            'data_breach': '💀', 'typosquatting': '🎭', 'disposable_email': '🗑️',
            'ml_high_risk': '🤖', 'ml_medium_risk': '🤖', 'anomaly_detected': '👁️',
            'suspicious_tld': '🌐', 'new_domain': '🆕', 'spam_domain': '📬',
            'missing_spf': '📧', 'missing_dmarc': '📧', 'missing_dnssec': '🔓',
            'password_breach': '🔑'
        }

        severity_colors = {
            'critical': '#f85149', 'high': '#d29922', 'medium': '#d29922', 'low': '#3fb950'
        }

        # Build chart data
        top_types = sorted_threats[:12]
        chart_labels = self._safe_json([t[0].replace('_', ' ').title() for t in top_types])
        chart_values = self._safe_json([t[1]['count'] for t in top_types])
        chart_colors = self._safe_json([severity_colors.get(t[1]['severity'], '#7c6bf5') for t in top_types])

        severity_json = self._safe_json([threat_severity_map.get('critical', 0), threat_severity_map.get('high', 0),
                                         threat_severity_map.get('medium', 0), threat_severity_map.get('low', 0)])

        html = f"""
        <div class="content-section" style="background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%); padding: 50px 80px; color: white;">
            <h2 class="section-title" style="color: white; margin-bottom: 40px;">
                <span class="section-icon">🔍</span>
                Advanced Threat Intelligence Deep-Dive
            </h2>

            <!-- Threat Severity Overview -->
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin-bottom: 40px;">
                <div style="background: rgba(248,81,73,0.2); border: 2px solid #f85149; border-radius: 15px; padding: 25px; text-align: center;">
                    <div style="font-size: 42px; font-weight: 800; color: #f85149;">{threat_severity_map.get('critical', 0)}</div>
                    <div style="font-size: 14px; opacity: 0.9; margin-top: 5px;">Critical Threats</div>
                </div>
                <div style="background: rgba(210,153,34,0.2); border: 2px solid #d29922; border-radius: 15px; padding: 25px; text-align: center;">
                    <div style="font-size: 42px; font-weight: 800; color: #d29922;">{threat_severity_map.get('high', 0)}</div>
                    <div style="font-size: 14px; opacity: 0.9; margin-top: 5px;">High Threats</div>
                </div>
                <div style="background: rgba(210,153,34,0.2); border: 2px solid #d29922; border-radius: 15px; padding: 25px; text-align: center;">
                    <div style="font-size: 42px; font-weight: 800; color: #d29922;">{threat_severity_map.get('medium', 0)}</div>
                    <div style="font-size: 14px; opacity: 0.9; margin-top: 5px;">Medium Threats</div>
                </div>
                <div style="background: rgba(0,212,170,0.2); border: 2px solid #3fb950; border-radius: 15px; padding: 25px; text-align: center;">
                    <div style="font-size: 42px; font-weight: 800; color: #3fb950;">{threat_severity_map.get('low', 0)}</div>
                    <div style="font-size: 14px; opacity: 0.9; margin-top: 5px;">Low Threats</div>
                </div>
            </div>

            <!-- Charts Row -->
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 30px; margin-bottom: 40px;">
                <div style="background: white; border-radius: 20px; padding: 30px; box-shadow: 0 10px 30px rgba(0,0,0,0.2);">
                    <h3 style="font-size: 20px; margin-bottom: 20px; color: #1e3c72;">Threat Types Detected</h3>
                    <canvas id="threatDeepDiveChart" style="max-height: 300px;"></canvas>
                </div>
                <div style="background: white; border-radius: 20px; padding: 30px; box-shadow: 0 10px 30px rgba(0,0,0,0.2);">
                    <h3 style="font-size: 20px; margin-bottom: 20px; color: #1e3c72;">Threat Severity Breakdown</h3>
                    <canvas id="threatSeverityPie" style="max-height: 300px;"></canvas>
                </div>
            </div>

            <!-- Per-Threat-Type Details -->
            <div style="background: rgba(255,255,255,0.05); border-radius: 20px; padding: 30px; backdrop-filter: blur(10px);">
                <h3 style="font-size: 22px; margin-bottom: 25px;">Threat Catalog</h3>
        """

        for t_type, info in sorted_threats[:15]:
            icon = threat_icons.get(t_type, '⚡')
            sev_color = severity_colors.get(info['severity'], '#7c6bf5')
            avg_conf = sum(info['avg_confidence']) / len(info['avg_confidence']) if info['avg_confidence'] else 0
            affected_emails = list(set(info['emails']))[:5]
            email_list = ', '.join(self._esc(e) for e in affected_emails)
            if len(set(info['emails'])) > 5:
                email_list += f' (+{len(set(info["emails"])) - 5} more)'

            html += f"""
                <div style="background: rgba(255,255,255,0.08); border-left: 5px solid {sev_color}; border-radius: 0 12px 12px 0; padding: 20px 25px; margin-bottom: 15px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                        <div style="display: flex; align-items: center; gap: 12px;">
                            <span style="font-size: 28px;">{icon}</span>
                            <div>
                                <div style="font-weight: 700; font-size: 17px;">{self._esc(t_type.replace('_', ' ').title())}</div>
                                <div style="font-size: 13px; opacity: 0.7;">{self._esc(info['description'][:120])}</div>
                            </div>
                        </div>
                        <div style="display: flex; gap: 15px; align-items: center;">
                            <div style="text-align: center;">
                                <div style="font-size: 24px; font-weight: 700; color: {sev_color};">{info['count']}</div>
                                <div style="font-size: 11px; opacity: 0.7;">Detections</div>
                            </div>
                            <div style="background: {sev_color}; color: white; padding: 5px 15px; border-radius: 20px; font-size: 12px; font-weight: 600; text-transform: uppercase;">{info['severity']}</div>
                        </div>
                    </div>
                    <div style="font-size: 12px; opacity: 0.6; margin-top: 8px;">
                        {'<strong>Confidence:</strong> ' + f'{avg_conf:.0%}' + ' | ' if avg_conf else ''}
                        <strong>Affected:</strong> {email_list}
                    </div>
                </div>
            """

        html += f"""
            </div>
        </div>

        <script>
            new Chart(document.getElementById('threatDeepDiveChart'), {{
                type: 'bar',
                data: {{
                    labels: {chart_labels},
                    datasets: [{{
                        label: 'Detections',
                        data: {chart_values},
                        backgroundColor: {chart_colors},
                        borderRadius: 8
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: true,
                    plugins: {{ legend: {{ display: false }} }},
                    scales: {{
                        y: {{ beginAtZero: true, ticks: {{ font: {{ size: 12 }} }} }},
                        x: {{ ticks: {{ font: {{ size: 10 }}, maxRotation: 45, minRotation: 45 }} }}
                    }}
                }}
            }});

            new Chart(document.getElementById('threatSeverityPie'), {{
                type: 'doughnut',
                data: {{
                    labels: ['Critical', 'High', 'Medium', 'Low'],
                    datasets: [{{
                        data: {severity_json},
                        backgroundColor: ['#f85149', '#d29922', '#d29922', '#3fb950'],
                        borderWidth: 0
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: true,
                    plugins: {{
                        legend: {{ position: 'bottom', labels: {{ padding: 15, font: {{ size: 13 }}, color: '#333' }} }}
                    }}
                }}
            }});
        </script>
        """

        return html

    def _generate_disposable_typosquat_section(self, results: List[Dict]) -> str:
        """Generate disposable email & typosquatting detection section"""
        import json

        # Collect disposable emails
        disposable_emails = []
        typosquat_emails = []

        for r in results:
            email = r.get('email', 'N/A')
            # Check for disposable email threat
            for threat in (r.get('threats') or []):
                if threat.get('type') == 'disposable_email':
                    disposable_emails.append({
                        'email': email,
                        'confidence': threat.get('confidence', 0),
                        'risk_score': r.get('risk_score', 0)
                    })
                    break

            # Check for typosquatting
            typosquat_info = r.get('typosquat_info') or {}
            if typosquat_info.get('is_typosquat'):
                typosquat_emails.append({
                    'email': email,
                    'target_domain': typosquat_info.get('target_domain', 'Unknown'),
                    'attack_type': typosquat_info.get('attack_type', 'Unknown'),
                    'similarity': typosquat_info.get('similarity', 0),
                    'risk_score': r.get('risk_score', 0)
                })

        if not disposable_emails and not typosquat_emails:
            return ""

        html = """
        <div class="content-section" style="background: linear-gradient(135deg, #c31432 0%, #240b36 100%); padding: 50px 80px; color: white;">
            <h2 class="section-title" style="color: white; margin-bottom: 40px;">
                <span class="section-icon">🎭</span>
                Email Fraud & Impersonation Detection
            </h2>
        """

        # Summary cards
        html += f"""
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 30px; margin-bottom: 40px;">
                <div style="background: rgba(255,255,255,0.1); border-radius: 20px; padding: 30px; text-align: center; backdrop-filter: blur(10px);">
                    <div style="font-size: 56px; margin-bottom: 10px;">🗑️</div>
                    <div style="font-size: 48px; font-weight: 800;">{len(disposable_emails)}</div>
                    <div style="font-size: 16px; opacity: 0.9; margin-top: 5px;">Disposable/Temp Emails</div>
                    <div style="font-size: 13px; opacity: 0.6; margin-top: 10px;">Known throwaway email services detected via 718+ domain database, pattern matching, and high-risk TLD analysis</div>
                </div>
                <div style="background: rgba(255,255,255,0.1); border-radius: 20px; padding: 30px; text-align: center; backdrop-filter: blur(10px);">
                    <div style="font-size: 56px; margin-bottom: 10px;">🎭</div>
                    <div style="font-size: 48px; font-weight: 800;">{len(typosquat_emails)}</div>
                    <div style="font-size: 16px; opacity: 0.9; margin-top: 5px;">Typosquatting Domains</div>
                    <div style="font-size: 13px; opacity: 0.6; margin-top: 10px;">Domains impersonating legitimate providers via Levenshtein distance, homoglyph detection, and Shannon entropy analysis</div>
                </div>
            </div>
        """

        # Disposable email details
        if disposable_emails:
            html += """
            <div style="background: rgba(255,255,255,0.08); border-radius: 20px; padding: 30px; margin-bottom: 30px;">
                <h3 style="font-size: 22px; margin-bottom: 20px;">🗑️ Disposable Email Addresses</h3>
                <p style="font-size: 14px; opacity: 0.7; margin-bottom: 20px;">These emails use temporary/disposable services often associated with fraud, spam, or anonymized activity.</p>
            """
            for d in disposable_emails:
                html += f"""
                <div style="background: rgba(255,255,255,0.06); border-left: 4px solid #f85149; border-radius: 0 10px 10px 0; padding: 15px 20px; margin-bottom: 10px; display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <div style="font-weight: 600; font-size: 16px;">{self._esc(d['email'])}</div>
                        <div style="font-size: 12px; opacity: 0.6;">Confidence: {d['confidence']:.0%}</div>
                    </div>
                    <div style="background: #f85149; padding: 6px 16px; border-radius: 20px; font-size: 13px; font-weight: 600;">Risk: {d['risk_score']}/100</div>
                </div>
                """
            html += "</div>"

        # Typosquatting details
        if typosquat_emails:
            html += """
            <div style="background: rgba(255,255,255,0.08); border-radius: 20px; padding: 30px;">
                <h3 style="font-size: 22px; margin-bottom: 20px;">🎭 Typosquatting Impersonation</h3>
                <p style="font-size: 14px; opacity: 0.7; margin-bottom: 20px;">These domains mimic legitimate email providers to deceive recipients. This is a common phishing/social engineering tactic.</p>
            """
            for t in typosquat_emails:
                similarity_color = '#f85149' if t['similarity'] > 0.9 else '#d29922' if t['similarity'] > 0.7 else '#d29922'
                html += f"""
                <div style="background: rgba(255,255,255,0.06); border-left: 4px solid {similarity_color}; border-radius: 0 10px 10px 0; padding: 15px 20px; margin-bottom: 12px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <div style="font-weight: 600; font-size: 16px;">{self._esc(t['email'])}</div>
                            <div style="font-size: 13px; opacity: 0.8; margin-top: 5px;">
                                Impersonating: <strong style="color: #3fb950;">{self._esc(t['target_domain'])}</strong> |
                                Attack: <strong>{self._esc(t['attack_type'].replace('_', ' ').title())}</strong> |
                                Similarity: <strong style="color: {similarity_color};">{t['similarity']:.0%}</strong>
                            </div>
                        </div>
                        <div style="background: {similarity_color}; padding: 6px 16px; border-radius: 20px; font-size: 13px; font-weight: 600;">Risk: {t['risk_score']}/100</div>
                    </div>
                </div>
                """
            html += "</div>"

        html += "</div>"
        return html

    def _generate_domain_intelligence_section(self, results: List[Dict]) -> str:
        """Generate domain intelligence & vulnerability assessment section"""
        import json
        from collections import Counter

        # Collect domain reputation data
        domain_data = {}
        all_vulnerabilities = []

        for r in results:
            email = r.get('email', 'N/A')
            domain = email.split('@')[1] if '@' in email else 'N/A'

            dom_rep = r.get('domain_reputation') or {}
            if domain not in domain_data and domain != 'N/A':
                domain_data[domain] = {
                    'age': dom_rep.get('age', 'Unknown'),
                    'score': dom_rep.get('score', 'N/A'),
                    'flags': dom_rep.get('flags') or [],
                    'registrar': dom_rep.get('registrar', 'Unknown'),
                    'emails_count': 0,
                    'avg_risk': 0,
                    'total_risk': 0
                }
            if domain in domain_data:
                domain_data[domain]['emails_count'] += 1
                domain_data[domain]['total_risk'] += r.get('risk_score', 0)

            # Collect vulnerabilities
            for vuln in (r.get('vulnerabilities') or []):
                all_vulnerabilities.append({
                    'email': email,
                    'domain': domain,
                    'type': vuln.get('type', 'unknown'),
                    'description': vuln.get('description', ''),
                    'severity': vuln.get('severity', 'low'),
                    'remediation': vuln.get('remediation', '')
                })

        # Calculate avg risk per domain
        for domain in domain_data:
            d = domain_data[domain]
            d['avg_risk'] = int(d['total_risk'] / d['emails_count']) if d['emails_count'] > 0 else 0

        # Sort domains by risk
        sorted_domains = sorted(domain_data.items(), key=lambda x: x[1]['avg_risk'], reverse=True)

        # Vulnerability counts
        vuln_counts = Counter([v['type'] for v in all_vulnerabilities])
        vuln_severity = Counter([v['severity'] for v in all_vulnerabilities])

        if not domain_data and not all_vulnerabilities:
            return ""

        # Chart data
        domain_names = self._safe_json([d[0] for d in sorted_domains[:10]])
        domain_risks = self._safe_json([d[1]['avg_risk'] for d in sorted_domains[:10]])
        domain_scores = self._safe_json([d[1]['score'] if isinstance(d[1]['score'], (int, float)) else 50 for d in sorted_domains[:10]])

        vuln_labels = self._safe_json([v[0].replace('_', ' ').title() for v in vuln_counts.most_common(8)])
        vuln_values = self._safe_json([v[1] for v in vuln_counts.most_common(8)])

        html = f"""
        <div class="content-section" style="background: linear-gradient(135deg, #1a2a6c 0%, #b21f1f 50%, #fdbb2d 100%); padding: 50px 80px; color: white;">
            <h2 class="section-title" style="color: white; margin-bottom: 40px;">
                <span class="section-icon">🌐</span>
                Domain Intelligence & Vulnerability Assessment
            </h2>

            <!-- Summary Stats -->
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin-bottom: 40px;">
                <div style="background: rgba(255,255,255,0.15); border-radius: 15px; padding: 25px; text-align: center;">
                    <div style="font-size: 36px; font-weight: 800;">{len(domain_data)}</div>
                    <div style="font-size: 13px; opacity: 0.9;">Unique Domains</div>
                </div>
                <div style="background: rgba(255,255,255,0.15); border-radius: 15px; padding: 25px; text-align: center;">
                    <div style="font-size: 36px; font-weight: 800;">{len(all_vulnerabilities)}</div>
                    <div style="font-size: 13px; opacity: 0.9;">Vulnerabilities Found</div>
                </div>
                <div style="background: rgba(255,255,255,0.15); border-radius: 15px; padding: 25px; text-align: center;">
                    <div style="font-size: 36px; font-weight: 800;">{vuln_severity.get('critical', 0) + vuln_severity.get('high', 0)}</div>
                    <div style="font-size: 13px; opacity: 0.9;">Critical/High Vulns</div>
                </div>
                <div style="background: rgba(255,255,255,0.15); border-radius: 15px; padding: 25px; text-align: center;">
                    <div style="font-size: 36px; font-weight: 800;">{sum(1 for d in domain_data.values() if isinstance(d['age'], (int, float)) and d['age'] < 30)}</div>
                    <div style="font-size: 13px; opacity: 0.9;">New Domains (&lt;30d)</div>
                </div>
            </div>

            <!-- Charts -->
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 30px; margin-bottom: 40px;">
                <div style="background: white; border-radius: 20px; padding: 30px; box-shadow: 0 10px 30px rgba(0,0,0,0.2);">
                    <h3 style="font-size: 18px; margin-bottom: 20px; color: #1e3c72;">Domain Risk vs Reputation Score</h3>
                    <canvas id="domainIntelChart" style="max-height: 300px;"></canvas>
                </div>
                <div style="background: white; border-radius: 20px; padding: 30px; box-shadow: 0 10px 30px rgba(0,0,0,0.2);">
                    <h3 style="font-size: 18px; margin-bottom: 20px; color: #1e3c72;">Vulnerability Types</h3>
                    <canvas id="vulnTypesChart" style="max-height: 300px;"></canvas>
                </div>
            </div>

            <!-- Domain Intelligence Table -->
            <div style="background: rgba(255,255,255,0.08); border-radius: 20px; padding: 30px; margin-bottom: 30px;">
                <h3 style="font-size: 22px; margin-bottom: 20px;">Domain Reputation Intelligence</h3>
                <div style="overflow-x: auto;">
                    <table style="width: 100%; border-collapse: collapse;">
                        <thead>
                            <tr style="border-bottom: 2px solid rgba(255,255,255,0.3);">
                                <th style="text-align: left; padding: 12px 15px; font-size: 13px; text-transform: uppercase; letter-spacing: 0.5px;">Domain</th>
                                <th style="text-align: center; padding: 12px 15px; font-size: 13px;">Reputation</th>
                                <th style="text-align: center; padding: 12px 15px; font-size: 13px;">Age (days)</th>
                                <th style="text-align: center; padding: 12px 15px; font-size: 13px;">Avg Risk</th>
                                <th style="text-align: center; padding: 12px 15px; font-size: 13px;">Emails</th>
                                <th style="text-align: left; padding: 12px 15px; font-size: 13px;">Flags</th>
                            </tr>
                        </thead>
                        <tbody>
        """

        for domain, info in sorted_domains[:20]:
            rep_score = info['score'] if isinstance(info['score'], (int, float)) else 'N/A'
            rep_color = '#3fb950' if isinstance(rep_score, (int, float)) and rep_score > 60 else '#d29922' if isinstance(rep_score, (int, float)) and rep_score > 30 else '#f85149'
            age_display = f"{info['age']}" if isinstance(info['age'], (int, float)) else 'Unknown'
            age_color = '#f85149' if isinstance(info['age'], (int, float)) and info['age'] < 30 else '#d29922' if isinstance(info['age'], (int, float)) and info['age'] < 365 else '#3fb950'
            risk_color = '#f85149' if info['avg_risk'] > 60 else '#d29922' if info['avg_risk'] > 40 else '#3fb950'
            flags_display = ', '.join(self._esc(str(f)) for f in info['flags'][:4]) if info['flags'] else 'None'

            html += f"""
                            <tr style="border-bottom: 1px solid rgba(255,255,255,0.1);">
                                <td style="padding: 12px 15px; font-weight: 600;">{self._esc(domain)}</td>
                                <td style="text-align: center; padding: 12px 15px; color: {rep_color}; font-weight: 700;">{rep_score}</td>
                                <td style="text-align: center; padding: 12px 15px; color: {age_color}; font-weight: 600;">{age_display}</td>
                                <td style="text-align: center; padding: 12px 15px; color: {risk_color}; font-weight: 700;">{info['avg_risk']}</td>
                                <td style="text-align: center; padding: 12px 15px;">{info['emails_count']}</td>
                                <td style="padding: 12px 15px; font-size: 12px; opacity: 0.8;">{flags_display}</td>
                            </tr>
            """

        html += """
                        </tbody>
                    </table>
                </div>
            </div>
        """

        # Vulnerability Assessment
        if all_vulnerabilities:
            # Group by type
            vuln_by_type = {}
            for v in all_vulnerabilities:
                vt = v['type']
                if vt not in vuln_by_type:
                    vuln_by_type[vt] = {'description': v['description'], 'severity': v['severity'],
                                        'remediation': v['remediation'], 'count': 0, 'domains': set()}
                vuln_by_type[vt]['count'] += 1
                vuln_by_type[vt]['domains'].add(v['domain'])

            html += """
            <div style="background: rgba(255,255,255,0.08); border-radius: 20px; padding: 30px;">
                <h3 style="font-size: 22px; margin-bottom: 20px;">Vulnerability Assessment</h3>
            """

            vuln_icons = {
                'missing_spf': '📧', 'missing_dmarc': '🛡️', 'missing_dnssec': '🔓',
                'weak_password_policy': '🔑', 'no_mfa': '📱'
            }

            severity_colors = {'critical': '#f85149', 'high': '#d29922', 'medium': '#d29922', 'low': '#3fb950'}

            for vt, vinfo in sorted(vuln_by_type.items(), key=lambda x: {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}.get(x[1]['severity'], 4)):
                icon = vuln_icons.get(vt, '⚠️')
                sev_color = severity_colors.get(vinfo['severity'], '#7c6bf5')
                domains_list = ', '.join(self._esc(d) for d in list(vinfo['domains'])[:5])
                if len(vinfo['domains']) > 5:
                    domains_list += f' (+{len(vinfo["domains"]) - 5} more)'

                html += f"""
                <div style="background: rgba(255,255,255,0.06); border-left: 5px solid {sev_color}; border-radius: 0 12px 12px 0; padding: 18px 22px; margin-bottom: 12px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <div style="display: flex; align-items: center; gap: 10px;">
                            <span style="font-size: 24px;">{icon}</span>
                            <div style="font-weight: 700; font-size: 16px;">{self._esc(vt.replace('_', ' ').title())}</div>
                        </div>
                        <div style="display: flex; gap: 10px; align-items: center;">
                            <span style="font-size: 20px; font-weight: 700; color: {sev_color};">{vinfo['count']}</span>
                            <span style="background: {sev_color}; color: white; padding: 4px 12px; border-radius: 15px; font-size: 11px; font-weight: 600; text-transform: uppercase;">{vinfo['severity']}</span>
                        </div>
                    </div>
                    <div style="font-size: 13px; opacity: 0.8;">{self._esc(vinfo['description'])}</div>
                    {'<div style="font-size: 12px; color: #3fb950; margin-top: 6px;"><strong>Remediation:</strong> ' + self._esc(vinfo["remediation"]) + '</div>' if vinfo['remediation'] else ''}
                    <div style="font-size: 11px; opacity: 0.5; margin-top: 6px;">Affected domains: {domains_list}</div>
                </div>
                """

            html += "</div>"

        html += f"""
        </div>

        <script>
            new Chart(document.getElementById('domainIntelChart'), {{
                type: 'bar',
                data: {{
                    labels: {domain_names},
                    datasets: [
                        {{
                            label: 'Risk Score',
                            data: {domain_risks},
                            backgroundColor: 'rgba(248,81,73,0.7)',
                            borderRadius: 6
                        }},
                        {{
                            label: 'Reputation',
                            data: {domain_scores},
                            backgroundColor: 'rgba(0,212,170,0.7)',
                            borderRadius: 6
                        }}
                    ]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: true,
                    plugins: {{
                        legend: {{ position: 'top', labels: {{ font: {{ size: 12 }} }} }}
                    }},
                    scales: {{
                        y: {{ beginAtZero: true, max: 100 }},
                        x: {{ ticks: {{ font: {{ size: 10 }}, maxRotation: 45, minRotation: 45 }} }}
                    }}
                }}
            }});

            new Chart(document.getElementById('vulnTypesChart'), {{
                type: 'doughnut',
                data: {{
                    labels: {vuln_labels},
                    datasets: [{{
                        data: {vuln_values},
                        backgroundColor: ['#f85149', '#d29922', '#d29922', '#7c6bf5', '#a78bfa', '#3fb950', '#58a6ff', '#a78bfa'],
                        borderWidth: 0
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: true,
                    plugins: {{
                        legend: {{ position: 'bottom', labels: {{ padding: 12, font: {{ size: 11 }} }} }}
                    }}
                }}
            }});
        </script>
        """

        return html

    def _generate_advanced_security_section(self, results: List[Dict]) -> str:
        """Generate advanced security checks section (DNSBL, CT, DGA, ThreatFox, Gravatar, Parked)"""
        has_any = any(
            r.get('dnsbl') or r.get('cert_transparency') or r.get('threatfox') or
            r.get('dga_analysis') or r.get('parked_domain') or r.get('gravatar')
            for r in results
        )
        if not has_any:
            return ''

        # Aggregate stats
        dnsbl_listed = sum(1 for r in results if (r.get('dnsbl') or {}).get('listed'))
        threatfox_found = sum(1 for r in results if (r.get('threatfox') or {}).get('found'))
        dga_detected = sum(1 for r in results if (r.get('dga_analysis') or {}).get('is_dga'))
        parked_count = sum(1 for r in results if (r.get('parked_domain') or {}).get('is_parked'))
        gravatar_count = sum(1 for r in results if (r.get('gravatar') or {}).get('has_profile'))
        ct_count = sum(1 for r in results if (r.get('cert_transparency') or {}).get('found'))
        total = len(results)

        # DNS protocol adoption
        bimi_count = sum(1 for r in results if (r.get('dns_security') or {}).get('bimi'))
        mta_sts_count = sum(1 for r in results if (r.get('dns_security') or {}).get('mta_sts'))
        tls_rpt_count = sum(1 for r in results if (r.get('dns_security') or {}).get('tls_rpt'))

        html = f"""
        <!-- Advanced Security Checks -->
        <div class="content-section" style="padding: 60px 80px;">
            <h2 class="section-title" style="color: #1e3c72; font-size: 28px; margin-bottom: 30px;">
                <span style="margin-right: 10px;">&#x1f6e1;</span>Advanced Security Intelligence
            </h2>

            <!-- Summary Cards -->
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-bottom: 40px;">
                <div style="background: {'#fff5f5' if dnsbl_listed else '#f0fff4'}; border-radius: 12px; padding: 25px; border-left: 4px solid {'#e53e3e' if dnsbl_listed else '#38a169'};">
                    <div style="font-size: 32px; font-weight: 800; color: {'#e53e3e' if dnsbl_listed else '#38a169'};">{dnsbl_listed}</div>
                    <div style="font-size: 14px; color: #4a5568; margin-top: 5px;">DNSBL Blacklisted</div>
                    <div style="font-size: 12px; color: #718096;">of {total} domains checked</div>
                </div>
                <div style="background: {'#fff5f5' if threatfox_found else '#f0fff4'}; border-radius: 12px; padding: 25px; border-left: 4px solid {'#e53e3e' if threatfox_found else '#38a169'};">
                    <div style="font-size: 32px; font-weight: 800; color: {'#e53e3e' if threatfox_found else '#38a169'};">{threatfox_found}</div>
                    <div style="font-size: 14px; color: #4a5568; margin-top: 5px;">ThreatFox IOC Matches</div>
                    <div style="font-size: 12px; color: #718096;">abuse.ch threat database</div>
                </div>
                <div style="background: {'#fffaf0' if dga_detected else '#f0fff4'}; border-radius: 12px; padding: 25px; border-left: 4px solid {'#dd6b20' if dga_detected else '#38a169'};">
                    <div style="font-size: 32px; font-weight: 800; color: {'#dd6b20' if dga_detected else '#38a169'};">{dga_detected}</div>
                    <div style="font-size: 14px; color: #4a5568; margin-top: 5px;">DGA Domains Detected</div>
                    <div style="font-size: 12px; color: #718096;">algorithmically generated</div>
                </div>
            </div>

            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-bottom: 40px;">
                <div style="background: #f7fafc; border-radius: 12px; padding: 25px; border-left: 4px solid #4299e1;">
                    <div style="font-size: 32px; font-weight: 800; color: #4299e1;">{gravatar_count}</div>
                    <div style="font-size: 14px; color: #4a5568; margin-top: 5px;">Gravatar Profiles</div>
                    <div style="font-size: 12px; color: #718096;">verified user presence</div>
                </div>
                <div style="background: {'#fffaf0' if parked_count else '#f0fff4'}; border-radius: 12px; padding: 25px; border-left: 4px solid {'#dd6b20' if parked_count else '#38a169'};">
                    <div style="font-size: 32px; font-weight: 800; color: {'#dd6b20' if parked_count else '#38a169'};">{parked_count}</div>
                    <div style="font-size: 14px; color: #4a5568; margin-top: 5px;">Parked Domains</div>
                    <div style="font-size: 12px; color: #718096;">inactive or for sale</div>
                </div>
                <div style="background: #f7fafc; border-radius: 12px; padding: 25px; border-left: 4px solid #4299e1;">
                    <div style="font-size: 32px; font-weight: 800; color: #4299e1;">{ct_count}</div>
                    <div style="font-size: 14px; color: #4a5568; margin-top: 5px;">CT Log Entries</div>
                    <div style="font-size: 12px; color: #718096;">certificate transparency</div>
                </div>
            </div>

            <!-- Email Protocol Adoption -->
            <h3 style="font-size: 20px; color: #2d3748; margin-bottom: 20px;">Email Protocol Adoption</h3>
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin-bottom: 30px;">
                <div style="background: #f7fafc; border-radius: 8px; padding: 15px; text-align: center;">
                    <div style="font-size: 24px; font-weight: 700; color: #4299e1;">{bimi_count}/{total}</div>
                    <div style="font-size: 13px; color: #718096;">BIMI</div>
                </div>
                <div style="background: #f7fafc; border-radius: 8px; padding: 15px; text-align: center;">
                    <div style="font-size: 24px; font-weight: 700; color: #4299e1;">{mta_sts_count}/{total}</div>
                    <div style="font-size: 13px; color: #718096;">MTA-STS</div>
                </div>
                <div style="background: #f7fafc; border-radius: 8px; padding: 15px; text-align: center;">
                    <div style="font-size: 24px; font-weight: 700; color: #4299e1;">{tls_rpt_count}/{total}</div>
                    <div style="font-size: 13px; color: #718096;">TLS-RPT</div>
                </div>
            </div>

            <!-- Detailed Table -->
            <h3 style="font-size: 20px; color: #2d3748; margin-bottom: 20px;">Per-Email Advanced Check Results</h3>
            <table style="width: 100%; border-collapse: collapse; font-size: 13px;">
                <thead>
                    <tr style="background: #edf2f7;">
                        <th style="padding: 12px; text-align: left; border-bottom: 2px solid #cbd5e0;">Email</th>
                        <th style="padding: 12px; text-align: center; border-bottom: 2px solid #cbd5e0;">DNSBL</th>
                        <th style="padding: 12px; text-align: center; border-bottom: 2px solid #cbd5e0;">ThreatFox</th>
                        <th style="padding: 12px; text-align: center; border-bottom: 2px solid #cbd5e0;">DGA</th>
                        <th style="padding: 12px; text-align: center; border-bottom: 2px solid #cbd5e0;">Certs</th>
                        <th style="padding: 12px; text-align: center; border-bottom: 2px solid #cbd5e0;">Gravatar</th>
                        <th style="padding: 12px; text-align: center; border-bottom: 2px solid #cbd5e0;">Parked</th>
                    </tr>
                </thead>
                <tbody>"""

        for r in results:
            dnsbl = r.get('dnsbl') or {}
            tfox = r.get('threatfox') or {}
            dga = r.get('dga_analysis') or {}
            ct = r.get('cert_transparency') or {}
            grav = r.get('gravatar') or {}
            park = r.get('parked_domain') or {}

            dnsbl_html = f'<span style="color:#e53e3e;">Listed ({self._esc(str(dnsbl.get("listed_count", 0)))})</span>' if dnsbl.get('listed') else '<span style="color:#38a169;">Clean</span>'
            tfox_html = f'<span style="color:#e53e3e;">{self._esc(str(tfox.get("ioc_count", 0)))} IOC(s)</span>' if tfox.get('found') else '<span style="color:#38a169;">Clean</span>'
            dga_score_val = float(dga.get('dga_score') or 0.0)
            dga_html = f'<span style="color:#dd6b20;">{dga_score_val:.0%}</span>' if dga.get('is_dga') else '<span style="color:#38a169;">Normal</span>'
            ct_html = self._esc(str(ct.get('cert_count', 0))) if ct.get('found') else '-'
            grav_html = '<span style="color:#4299e1;">Yes</span>' if grav.get('has_profile') else 'No'
            park_html = '<span style="color:#dd6b20;">Parked</span>' if park.get('is_parked') else '<span style="color:#38a169;">Active</span>'

            html += f"""
                    <tr style="border-bottom: 1px solid #e2e8f0;">
                        <td style="padding: 10px;">{self._esc(str(r.get('email', 'N/A')))}</td>
                        <td style="padding: 10px; text-align: center;">{dnsbl_html}</td>
                        <td style="padding: 10px; text-align: center;">{tfox_html}</td>
                        <td style="padding: 10px; text-align: center;">{dga_html}</td>
                        <td style="padding: 10px; text-align: center;">{ct_html}</td>
                        <td style="padding: 10px; text-align: center;">{grav_html}</td>
                        <td style="padding: 10px; text-align: center;">{park_html}</td>
                    </tr>"""

        html += """
                </tbody>
            </table>
        </div>
        """

        return html

    def _generate_data_exposure_section(self, results: List[Dict]) -> str:
        """Generate data exposure summary & password breach alerts section"""
        import json
        from collections import Counter

        # Aggregate data classes across all breaches
        data_class_counts = Counter()
        password_breaches = []
        total_breach_count = 0
        total_affected_accounts = 0

        for r in results:
            email = r.get('email', 'N/A')
            breach_info = r.get('breach_info') or {}

            if breach_info.get('found'):
                total_breach_count += 1
                for breach in (breach_info.get('details') or []):
                    if isinstance(breach, dict):
                        data_classes = breach.get('data_classes') or []
                        for dc in data_classes:
                            if isinstance(dc, str):
                                data_class_counts[dc] += 1
                        pwn_count = breach.get('pwn_count', 0)
                        if isinstance(pwn_count, (int, float)):
                            total_affected_accounts += int(pwn_count)
                    elif isinstance(breach, list):
                        pass  # email already counted above; do not re-count per breach name

            # Password breaches
            pwd_breach = r.get('password_breach') or {}
            if pwd_breach.get('found'):
                password_breaches.append({
                    'email': email,
                    'risk_score': r.get('risk_score', 0)
                })

        if not data_class_counts and not password_breaches:
            return ""

        # Top data classes
        top_classes = data_class_counts.most_common(15)

        # Categorize data classes by sensitivity
        high_sensitivity = ['Passwords', 'Credit cards', 'Bank account numbers', 'Social security numbers',
                            'Credit card CVV', 'PINs', 'Security questions and answers', 'Auth tokens']
        medium_sensitivity = ['Email addresses', 'Phone numbers', 'Physical addresses', 'Dates of birth',
                              'IP addresses', 'Government issued IDs', 'Employers']
        low_sensitivity = ['Names', 'Usernames', 'Genders', 'Job titles', 'Education levels']

        high_count = sum(data_class_counts.get(dc, 0) for dc in high_sensitivity)
        medium_count = sum(data_class_counts.get(dc, 0) for dc in medium_sensitivity)
        low_count = sum(v for k, v in data_class_counts.items() if k not in high_sensitivity and k not in medium_sensitivity)

        # Chart data
        dc_labels = self._safe_json([dc[0] for dc in top_classes[:10]])
        dc_values = self._safe_json([dc[1] for dc in top_classes[:10]])
        sensitivity_json = self._safe_json([high_count, medium_count, low_count])

        html = f"""
        <div class="content-section" style="background: linear-gradient(135deg, #4a0e0e 0%, #8b0000 50%, #b91c1c 100%); padding: 50px 80px; color: white;">
            <h2 class="section-title" style="color: white; margin-bottom: 40px;">
                <span class="section-icon">💀</span>
                Data Exposure & Password Breach Intelligence
            </h2>

            <!-- Overview Stats -->
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin-bottom: 40px;">
                <div style="background: rgba(255,255,255,0.1); border-radius: 15px; padding: 25px; text-align: center;">
                    <div style="font-size: 36px; font-weight: 800;">{len(data_class_counts)}</div>
                    <div style="font-size: 13px; opacity: 0.9;">Data Types Exposed</div>
                </div>
                <div style="background: rgba(255,255,255,0.1); border-radius: 15px; padding: 25px; text-align: center;">
                    <div style="font-size: 36px; font-weight: 800; color: #f85149;">{high_count}</div>
                    <div style="font-size: 13px; opacity: 0.9;">High Sensitivity Exposures</div>
                </div>
                <div style="background: rgba(255,255,255,0.1); border-radius: 15px; padding: 25px; text-align: center;">
                    <div style="font-size: 36px; font-weight: 800; color: #d29922;">{len(password_breaches)}</div>
                    <div style="font-size: 13px; opacity: 0.9;">Password Breaches</div>
                </div>
                <div style="background: rgba(255,255,255,0.1); border-radius: 15px; padding: 25px; text-align: center;">
                    <div style="font-size: 36px; font-weight: 800;">{total_affected_accounts:,}</div>
                    <div style="font-size: 13px; opacity: 0.9;">Total Affected Accounts</div>
                </div>
            </div>

            <!-- Charts -->
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 30px; margin-bottom: 40px;">
                <div style="background: white; border-radius: 20px; padding: 30px; box-shadow: 0 10px 30px rgba(0,0,0,0.2);">
                    <h3 style="font-size: 18px; margin-bottom: 20px; color: #1e3c72;">Most Exposed Data Types</h3>
                    <canvas id="dataExposureChart" style="max-height: 300px;"></canvas>
                </div>
                <div style="background: white; border-radius: 20px; padding: 30px; box-shadow: 0 10px 30px rgba(0,0,0,0.2);">
                    <h3 style="font-size: 18px; margin-bottom: 20px; color: #1e3c72;">Exposure by Sensitivity Level</h3>
                    <canvas id="sensitivityChart" style="max-height: 300px;"></canvas>
                </div>
            </div>

            <!-- Data Classes Grid -->
            <div style="background: rgba(255,255,255,0.08); border-radius: 20px; padding: 30px; margin-bottom: 30px;">
                <h3 style="font-size: 22px; margin-bottom: 20px;">Compromised Data Categories</h3>
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px;">
        """

        sensitivity_icons = {
            'Passwords': '🔑', 'Credit cards': '💳', 'Email addresses': '📧',
            'Phone numbers': '📱', 'Names': '👤', 'IP addresses': '🌐',
            'Dates of birth': '🎂', 'Physical addresses': '🏠', 'Usernames': '👤',
            'Social security numbers': '🆔', 'Bank account numbers': '🏦',
            'Security questions and answers': '❓', 'Auth tokens': '🔐'
        }

        for dc, count in top_classes:
            icon = sensitivity_icons.get(dc, '📋')
            is_high = dc in high_sensitivity
            bg_color = 'rgba(248,81,73,0.15)' if is_high else 'rgba(255,255,255,0.06)'
            border_color = '#f85149' if is_high else 'rgba(255,255,255,0.2)'

            html += f"""
                    <div style="background: {bg_color}; border: 1px solid {border_color}; border-radius: 10px; padding: 15px; display: flex; align-items: center; gap: 10px;">
                        <span style="font-size: 22px;">{icon}</span>
                        <div>
                            <div style="font-weight: 600; font-size: 14px;">{self._esc(str(dc))}</div>
                            <div style="font-size: 12px; opacity: 0.6;">{count} exposure(s){'  ⚠️ HIGH' if is_high else ''}</div>
                        </div>
                    </div>
            """

        html += """
                </div>
            </div>
        """

        # Password breach alerts
        if password_breaches:
            html += """
            <div style="background: rgba(255,255,255,0.08); border: 2px solid #f85149; border-radius: 20px; padding: 30px;">
                <h3 style="font-size: 22px; margin-bottom: 10px; color: #f85149;">🔑 PASSWORD BREACH ALERTS</h3>
                <p style="font-size: 14px; opacity: 0.7; margin-bottom: 20px;">
                    The following accounts have passwords found in dark web breach databases (verified via XposedOrNot Keccak-512 k-anonymity API).
                    These passwords MUST be changed immediately.
                </p>
            """
            for pwd in password_breaches:
                html += f"""
                <div style="background: rgba(248,81,73,0.15); border-left: 5px solid #f85149; border-radius: 0 10px 10px 0; padding: 15px 20px; margin-bottom: 10px; display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <div style="font-weight: 700; font-size: 16px;">{self._esc(pwd['email'])}</div>
                        <div style="font-size: 12px; color: #f85149; font-weight: 600;">PASSWORD COMPROMISED - CHANGE IMMEDIATELY</div>
                    </div>
                    <div style="background: #f85149; padding: 8px 20px; border-radius: 20px; font-size: 14px; font-weight: 700;">Risk: {pwd['risk_score']}/100</div>
                </div>
                """
            html += "</div>"

        html += f"""
        </div>

        <script>
            new Chart(document.getElementById('dataExposureChart'), {{
                type: 'bar',
                data: {{
                    labels: {dc_labels},
                    datasets: [{{
                        label: 'Exposure Count',
                        data: {dc_values},
                        backgroundColor: '#f85149',
                        borderRadius: 8
                    }}]
                }},
                options: {{
                    indexAxis: 'y',
                    responsive: true,
                    maintainAspectRatio: true,
                    plugins: {{ legend: {{ display: false }} }},
                    scales: {{
                        x: {{ beginAtZero: true }},
                        y: {{ ticks: {{ font: {{ size: 11 }} }} }}
                    }}
                }}
            }});

            new Chart(document.getElementById('sensitivityChart'), {{
                type: 'doughnut',
                data: {{
                    labels: ['High Sensitivity', 'Medium Sensitivity', 'Low Sensitivity'],
                    datasets: [{{
                        data: {sensitivity_json},
                        backgroundColor: ['#f85149', '#d29922', '#3fb950'],
                        borderWidth: 0
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: true,
                    plugins: {{
                        legend: {{ position: 'bottom', labels: {{ padding: 15, font: {{ size: 13 }}, color: '#333' }} }}
                    }}
                }}
            }});
        </script>
        """

        return html

    def _generate_results_table_section(self, results: List[Dict]) -> str:
        """Generate detailed results table with ALL detection columns"""
        html = f"""
        <div class="content-section">
            <h2 class="section-title">
                <span class="section-icon">📈</span>
                Comprehensive Analysis Results (Top {min(50, len(results))} of {len(results)} by Risk Score)
            </h2>

            <div style="overflow-x: auto;">
            <table class="breach-table">
                <thead>
                    <tr>
                        <th>Email Address</th>
                        <th>Risk Score</th>
                        <th>Risk Level</th>
                        <th>Breached</th>
                        <th>Breach Count</th>
                        <th>Threats</th>
                        <th>Disposable</th>
                        <th>Typosquat</th>
                        <th>DNS Score</th>
                        <th>ML Ensemble</th>
                        <th>Anomaly</th>
                        <th>Threat Types</th>
                    </tr>
                </thead>
                <tbody>
        """

        # Sort by risk score (highest first)
        sorted_results = sorted(results, key=lambda x: x.get('risk_score', 0), reverse=True)

        for result in sorted_results[:50]:  # Show top 50
            email = result.get('email', 'N/A')
            score = result.get('risk_score', 0)
            level = result.get('risk_level', 'unknown')
            breach_info = result.get('breach_info') or {}
            breached = 'YES' if breach_info.get('found') else 'NO'
            breach_count = breach_info.get('count', 0) if breach_info.get('found') else 0
            threats = result.get('threats') or []
            threat_count = len(threats)

            # Detection flags
            is_disposable = any(t.get('type') == 'disposable_email' for t in threats)
            typosquat_info = result.get('typosquat_info') or {}
            is_typosquat = typosquat_info.get('is_typosquat', False)
            typosquat_target = typosquat_info.get('target_domain', '')

            # DNS
            dns = result.get('dns_security') or {}
            dns_score = dns.get('score', 'N/A')

            # ML
            ml_preds = result.get('ml_predictions') or {}
            ensemble = ml_preds.get('ensemble')
            anomaly = ml_preds.get('anomaly_score')

            # Threat types summary
            threat_types = list(set(t.get('type', '') for t in threats))
            threat_summary = ', '.join(self._esc(t.replace('_', ' ').title()) for t in threat_types[:3])
            if len(threat_types) > 3:
                threat_summary += f' +{len(threat_types) - 3}'

            # Colors
            score_color = '#f85149' if score > 60 else '#d29922' if score > 40 else '#3fb950'
            ensemble_val = float(ensemble) if ensemble is not None else 0
            ensemble_color = '#f85149' if ensemble_val > 0.7 else '#d29922' if ensemble_val > 0.4 else '#3fb950'
            anomaly_val = float(anomaly) if anomaly is not None else 0
            anomaly_color = '#f85149' if anomaly_val > 0.7 else '#d29922' if anomaly_val > 0.3 else '#3fb950'

            html += f"""
                    <tr>
                        <td class="email-cell" style="font-size: 13px; max-width: 200px; overflow: hidden; text-overflow: ellipsis;">{self._esc(email)}</td>
                        <td>
                            <span class="score-badge" style="background: {score_color};">{score}</span>
                        </td>
                        <td>
                            <span class="badge badge-{level}">{level.upper()}</span>
                        </td>
                        <td>
                            <span style="color: {'#f85149' if breached == 'YES' else '#3fb950'}; font-weight: 700;">{breached}</span>
                        </td>
                        <td style="text-align: center;">
                            <span style="color: #1e3c72; font-weight: 700;">{breach_count}</span>
                        </td>
                        <td style="text-align: center;">
                            <span style="color: #6c757d;">{threat_count}</span>
                        </td>
                        <td style="text-align: center;">
                            <span style="color: {'#f85149' if is_disposable else '#3fb950'}; font-weight: 700;">{'🗑️ YES' if is_disposable else '✅ NO'}</span>
                        </td>
                        <td style="text-align: center;">
                            {'<span style="color: #f85149; font-weight: 700;" title="Impersonates ' + self._esc(typosquat_target) + '">🎭 ' + self._esc(typosquat_target) + '</span>' if is_typosquat else '<span style="color: #3fb950;">✅ NO</span>'}
                        </td>
                        <td style="text-align: center; font-weight: 600;">
                            {self._esc(str(dns_score))}
                        </td>
                        <td style="text-align: center;">
                            <span style="color: {ensemble_color}; font-weight: 700;">{f'{ensemble_val:.0%}' if ensemble is not None else 'N/A'}</span>
                        </td>
                        <td style="text-align: center;">
                            <span style="color: {anomaly_color}; font-weight: 700;">{f'{anomaly_val:.0%}' if anomaly is not None else 'N/A'}</span>
                        </td>
                        <td style="font-size: 11px; max-width: 180px; overflow: hidden; text-overflow: ellipsis;" title="{self._esc(', '.join(t.replace('_', ' ') for t in threat_types))}">
                            {threat_summary if threat_summary else '<span style="color: #3fb950;">None</span>'}
                        </td>
                    </tr>
            """

        html += """
                </tbody>
            </table>
            </div>
        </div>
        """

        return html

    def _generate_compliance_details_table(self, results: List[Dict]) -> str:
        """Generate detailed compliance table showing ISO 27001 & GDPR article violations per email"""
        html = """
        <div class="content-section" style="background: #f8f9fa;">
            <h2 class="section-title">
                <span class="section-icon">📋</span>
                Article-Level Compliance Assessment
            </h2>

            <div style="background: white; border-radius: 15px; padding: 30px; margin-bottom: 20px; box-shadow: 0 5px 20px rgba(0,0,0,0.06);">
                <p style="font-size: 16px; line-height: 1.8; color: #6c757d;">
                    This table shows compliance status for each email against specific ISO 27001:2022 controls and GDPR articles.
                    <strong style="color: #3fb950;">✅ PASS</strong> indicates compliance,
                    <strong style="color: #f85149;">❌ FAIL</strong> indicates violation,
                    <strong style="color: #d29922;">❌ NOTIFY</strong> indicates breach notification required.
                </p>
            </div>

            <table class="breach-table">
                <thead>
                    <tr>
                        <th style="min-width: 200px;">Email Address</th>
                        <th>GDPR Art. 5<br><small>(Data Protection)</small></th>
                        <th>GDPR Art. 32<br><small>(Security)</small></th>
                        <th>GDPR Art. 33<br><small>(Breach Notice)</small></th>
                        <th>ISO A.5.7<br><small>(Threat Intel)</small></th>
                        <th>ISO A.5.15<br><small>(Access Control)</small></th>
                        <th>ISO A.8.24<br><small>(Encryption)</small></th>
                        <th>Email Auth<br><small>(SPF/DMARC/DKIM)</small></th>
                    </tr>
                </thead>
                <tbody>
        """

        # Sort by email
        sorted_results = sorted(results, key=lambda x: x.get('email', ''))

        for result in sorted_results:
            email = result.get('email', 'N/A')
            breach_info = result.get('breach_info') or {}
            dns = result.get('dns_security') or {}
            risk_level = result.get('risk_level', 'unknown')

            # GDPR Article 5 - Data Protection Principles (violated if personal data at risk)
            gdpr_art5 = '❌ FAIL' if (breach_info.get('found') or risk_level in ['critical', 'high']) else '✅ PASS'
            gdpr_art5_color = '#f85149' if '❌' in gdpr_art5 else '#3fb950'
            gdpr_art5_reason = 'Personal data at risk' if '❌' in gdpr_art5 else 'Data protected'

            # GDPR Article 32 - Security of Processing (violated if breached)
            gdpr_art32 = '❌ FAIL' if breach_info.get('found') else '✅ PASS'
            gdpr_art32_color = '#f85149' if '❌' in gdpr_art32 else '#3fb950'
            gdpr_art32_reason = 'Security breach detected' if '❌' in gdpr_art32 else 'No breaches'

            # GDPR Article 33 - Breach Notification (requires notification if high/critical breach)
            requires_notification = breach_info.get('found') and breach_info.get('severity') in ['high', 'critical']
            gdpr_art33 = '❌ NOTIFY' if requires_notification else '✅ N/A'
            gdpr_art33_color = '#d29922' if '❌' in gdpr_art33 else '#3fb950'
            gdpr_art33_reason = 'Notification required within 72h' if requires_notification else 'No notification needed'

            # ISO 27001:2022 A.5.7 - Threat Intelligence (violated if high/critical risk)
            iso_a57 = '❌ FAIL' if risk_level in ['critical', 'high'] else '✅ PASS'
            iso_a57_color = '#f85149' if '❌' in iso_a57 else '#3fb950'
            iso_a57_reason = f'High-risk threats detected' if '❌' in iso_a57 else 'No threats'

            # ISO 27001:2022 A.5.15 - Access Control (pass if not breached and low risk)
            iso_a515 = '✅ PASS' if (not breach_info.get('found') and risk_level in ['low', 'minimal']) else '❌ FAIL'
            iso_a515_color = '#3fb950' if '✅' in iso_a515 else '#f85149'
            iso_a515_reason = 'Access controlled' if '✅' in iso_a515 else 'Access compromised'

            # ISO 27001:2022 A.8.24 - Cryptographic Controls (pass if not breached)
            iso_a824 = '✅ PASS' if not breach_info.get('found') else '❌ FAIL'
            iso_a824_color = '#3fb950' if '✅' in iso_a824 else '#f85149'
            iso_a824_reason = 'Encryption intact' if '✅' in iso_a824 else 'Encryption compromised'

            # Email Authentication (pass if SPF + DMARC + DKIM)
            email_auth = '✅ PASS' if (dns.get('spf') and dns.get('dmarc') and dns.get('dkim')) else '❌ FAIL'
            email_auth_color = '#3fb950' if '✅' in email_auth else '#f85149'
            missing_auth = []
            if not dns.get('spf'): missing_auth.append('SPF')
            if not dns.get('dmarc'): missing_auth.append('DMARC')
            if not dns.get('dkim'): missing_auth.append('DKIM')
            email_auth_reason = f'Missing: {", ".join(missing_auth)}' if missing_auth else 'Fully configured'

            html += f"""
                    <tr>
                        <td class="email-cell" style="font-size: 13px;">{self._esc(email)}</td>
                        <td style="text-align: center;">
                            <div style="color: {gdpr_art5_color}; font-weight: 700; font-size: 14px;">{gdpr_art5}</div>
                            <div style="font-size: 11px; color: #6c757d; margin-top: 5px;">{gdpr_art5_reason}</div>
                        </td>
                        <td style="text-align: center;">
                            <div style="color: {gdpr_art32_color}; font-weight: 700; font-size: 14px;">{gdpr_art32}</div>
                            <div style="font-size: 11px; color: #6c757d; margin-top: 5px;">{gdpr_art32_reason}</div>
                        </td>
                        <td style="text-align: center;">
                            <div style="color: {gdpr_art33_color}; font-weight: 700; font-size: 14px;">{gdpr_art33}</div>
                            <div style="font-size: 11px; color: #6c757d; margin-top: 5px;">{gdpr_art33_reason}</div>
                        </td>
                        <td style="text-align: center;">
                            <div style="color: {iso_a57_color}; font-weight: 700; font-size: 14px;">{iso_a57}</div>
                            <div style="font-size: 11px; color: #6c757d; margin-top: 5px;">{iso_a57_reason}</div>
                        </td>
                        <td style="text-align: center;">
                            <div style="color: {iso_a515_color}; font-weight: 700; font-size: 14px;">{iso_a515}</div>
                            <div style="font-size: 11px; color: #6c757d; margin-top: 5px;">{iso_a515_reason}</div>
                        </td>
                        <td style="text-align: center;">
                            <div style="color: {iso_a824_color}; font-weight: 700; font-size: 14px;">{iso_a824}</div>
                            <div style="font-size: 11px; color: #6c757d; margin-top: 5px;">{iso_a824_reason}</div>
                        </td>
                        <td style="text-align: center;">
                            <div style="color: {email_auth_color}; font-weight: 700; font-size: 14px;">{email_auth}</div>
                            <div style="font-size: 11px; color: #6c757d; margin-top: 5px;">{email_auth_reason}</div>
                        </td>
                    </tr>
            """

        html += """
                </tbody>
            </table>

            <div style="background: white; border-radius: 15px; padding: 30px; margin-top: 30px; box-shadow: 0 5px 20px rgba(0,0,0,0.06);">
                <h3 style="font-size: 20px; margin-bottom: 20px; color: #1e3c72;">📚 Compliance Framework Reference</h3>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 25px; line-height: 1.8;">
                    <div>
                        <h4 style="font-size: 16px; margin-bottom: 15px; color: #1e3c72;">GDPR Articles</h4>
                        <div style="font-size: 14px; color: #6c757d;">
                            <strong>Article 5:</strong> Principles relating to processing of personal data<br>
                            <strong>Article 32:</strong> Security of processing (implement appropriate security measures)<br>
                            <strong>Article 33:</strong> Notification of data breach to supervisory authority (within 72 hours)
                        </div>
                    </div>
                    <div>
                        <h4 style="font-size: 16px; margin-bottom: 15px; color: #1e3c72;">ISO 27001:2022 Controls</h4>
                        <div style="font-size: 14px; color: #6c757d;">
                            <strong>A.5.7:</strong> Threat intelligence (identify and analyze threats)<br>
                            <strong>A.5.15:</strong> Access control (restrict access to information)<br>
                            <strong>A.8.24:</strong> Use of cryptography (protect confidentiality and integrity)
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """

        return html

    def _generate_breach_details_section(self, results: List[Dict]) -> str:
        """Generate detailed breach information section"""
        breached_emails = [r for r in results if (r.get('breach_info') or {}).get('found')]

        if not breached_emails:
            return ""

        html = """
        <div class="content-section" style="background: #fef6f6;">
            <h2 class="section-title">
                <span class="section-icon">🚨</span>
                Breach Intelligence Report
            </h2>
        """

        for result in breached_emails[:20]:  # Show top 20 breached emails
            email = result.get('email', 'N/A')
            breach_info = result.get('breach_info') or {}
            severity = breach_info.get('severity', 'medium')
            count = breach_info.get('count', 0)

            severity_color = {
                'critical': '#f85149',
                'high': '#d29922',
                'medium': '#d29922',
                'low': '#3fb950'
            }.get(severity, '#d29922')

            html += f"""
            <div style="background: white; border-radius: 20px; padding: 40px; margin-bottom: 30px; border-left: 8px solid {severity_color}; box-shadow: 0 5px 20px rgba(0,0,0,0.08);">
                <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 25px;">
                    <div>
                        <h3 style="font-size: 24px; color: #1e3c72; margin-bottom: 10px;">{self._esc(email)}</h3>
                        <div style="display: flex; gap: 15px; align-items: center;">
                            <span class="badge badge-{severity}">Severity: {severity.upper()}</span>
                            <span style="color: #6c757d; font-size: 14px;">{count} breach(es) detected</span>
                        </div>
                    </div>
                </div>
            """

            # Add breach details
            details_list = breach_info.get('details') or []
            if not isinstance(details_list, list):
                details_list = []

            # Fallback: if details is empty but breaches list has names, build minimal details
            if not details_list and breach_info.get('breaches'):
                breach_names = breach_info['breaches']
                if isinstance(breach_names, list):
                    for bname in breach_names[:10]:
                        if isinstance(bname, str) and bname.strip():
                            details_list.append({'name': bname, 'breach_date': 'Unknown', 'data_classes': []})

            if details_list:
                html += """
                <div style="margin-top: 30px;">
                    <h4 style="font-size: 18px; color: #1e3c72; margin-bottom: 20px; display: flex; align-items: center; gap: 10px;">
                        <span>🔍</span> Breach Details
                    </h4>
                """

                esc = self._esc
                for breach in details_list[:5]:
                    if isinstance(breach, list):
                        # Handle XposedOrNot nested list format [["Adobe","LinkedIn"]]
                        for breach_name in breach[:5]:
                            if isinstance(breach_name, str):
                                html += f"""
                        <div style="background: #f8f9fa; border-radius: 12px; padding: 20px; margin-bottom: 15px;">
                            <div style="font-weight: 700; font-size: 16px; color: #1e3c72; margin-bottom: 8px;">
                                {esc(breach_name)}
                            </div>
                            <div style="color: #6c757d; font-size: 14px; margin-bottom: 5px;">
                                <strong>Source:</strong> XposedOrNot (Dark Web)
                            </div>
                        </div>
                                """
                    elif isinstance(breach, dict):
                        breach_date = esc(breach.get('breach_date', ''))
                        date_display = f"<span style='color: #6c757d; font-weight: 400; font-size: 14px;'>({breach_date})</span>" if breach_date and breach_date != 'Unknown' else ""
                        html += f"""
                        <div style="background: #f8f9fa; border-radius: 12px; padding: 20px; margin-bottom: 15px;">
                            <div style="font-weight: 700; font-size: 16px; color: #1e3c72; margin-bottom: 8px;">
                                {esc(breach.get('name', 'Unknown'))}
                                {date_display}
                            </div>
                        """

                        if breach.get('domain'):
                            html += f"""
                            <div style="color: #6c757d; font-size: 14px; margin-bottom: 5px;">
                                <strong>Domain:</strong> {esc(breach['domain'])}
                            </div>
                            """

                        if breach.get('pwn_count') is not None and breach.get('pwn_count') != 0:
                            try:
                                pwn_display = f"{int(breach['pwn_count']):,}"
                            except (TypeError, ValueError):
                                pwn_display = esc(breach['pwn_count'])
                            html += f"""
                            <div style="color: #6c757d; font-size: 14px; margin-bottom: 5px;">
                                <strong>Affected Accounts:</strong> {pwn_display}
                            </div>
                            """

                        data_classes = breach.get('data_classes') or []
                        if isinstance(data_classes, list) and data_classes:
                            data_list = ', '.join(esc(dc) for dc in data_classes[:8])
                            html += f"""
                            <div style="color: #f85149; font-size: 14px; margin-bottom: 5px;">
                                <strong>Compromised Data:</strong> {data_list}
                            </div>
                            """

                        if breach.get('description'):
                            html += f"""
                            <div style="color: #6c757d; font-size: 13px; margin-top: 8px; font-style: italic;">
                                {esc(breach['description'])}
                            </div>
                            """

                        html += "</div>"

                html += "</div>"

            # Add MITRE ATT&CK Techniques
            mitre_details = result.get('mitre_details', [])
            if mitre_details:
                html += """
                <div style="background: #f0f4ff; border-radius: 15px; padding: 25px; margin-top: 25px; border: 2px solid #58a6ff;">
                    <h4 style="font-size: 18px; color: #1e3c72; margin-bottom: 15px; display: flex; align-items: center; gap: 10px;">
                        <span>🎯</span> MITRE ATT&CK Techniques
                    </h4>
                """

                for technique in mitre_details[:5]:  # Show top 5 techniques
                    similarity = technique.get('similarity', 0)
                    confidence_color = '#3fb950' if similarity > 85 else '#d29922' if similarity > 70 else '#d29922'

                    tech_desc = technique.get('description', '')
                    desc_html = ''
                    if tech_desc:
                        truncated = tech_desc[:200] + '...' if len(tech_desc) > 200 else tech_desc
                        desc_html = f'<div style="color: #495057; font-size: 12px; margin-top: 5px;">{self._esc(truncated)}</div>'

                    html += f"""
                    <div style="background: white; border-radius: 10px; padding: 15px; margin-bottom: 12px; border-left: 4px solid {confidence_color};">
                        <div style="font-weight: 700; font-size: 15px; color: #1e3c72; margin-bottom: 5px;">
                            {self._esc(str(technique.get('id', 'N/A')))}: {self._esc(str(technique.get('name', 'Unknown')))}
                        </div>
                        <div style="color: #6c757d; font-size: 13px; margin-bottom: 3px;">
                            <strong>Tactic:</strong> {self._esc(str(technique.get('tactic', 'Unknown')))} |
                            <strong>Severity:</strong> {self._esc(str(technique.get('severity', 'N/A')).upper())}
                        </div>
                        <div style="color: {confidence_color}; font-size: 13px; font-weight: 600;">
                            <strong>Confidence:</strong> {similarity:.1f}%
                        </div>
                        {desc_html}
                    </div>
                    """

                html += "</div>"

            # Add mitigation steps
            if breach_info.get('mitigation_steps') and isinstance(breach_info['mitigation_steps'], list):
                html += """
                <div style="background: linear-gradient(135deg, #3fb950, #00ffb8); color: white; border-radius: 15px; padding: 25px; margin-top: 25px;">
                    <h4 style="font-size: 18px; margin-bottom: 15px; display: flex; align-items: center; gap: 10px;">
                        <span>🛡️</span> Recommended Actions
                    </h4>
                    <ol style="margin-left: 20px; line-height: 2;">
                """

                for step in breach_info['mitigation_steps'][:5]:
                    html += f"<li style='margin-bottom: 8px;'>{self._esc(str(step))}</li>"

                html += """
                    </ol>
                </div>
                """

            # Password breach info (if checked)
            password_breach = result.get('password_breach') or {}
            if password_breach and password_breach.get('found'):
                html += """
                <div style="background: rgba(248,81,73,0.1); border-radius: 15px; padding: 25px; margin-top: 25px; border: 2px solid #f85149;">
                    <h4 style="font-size: 18px; color: #f85149; margin-bottom: 15px; display: flex; align-items: center; gap: 10px;">
                        <span>🔑</span> Password Breach Detected
                    </h4>
                    <p style="color: #333; font-size: 14px; margin-bottom: 10px;">The associated password was found in known dark web breach databases.</p>
                    <p style="color: #f85149; font-size: 14px; font-weight: 600;">Recommendation: Change this password immediately on all accounts.</p>
                    <p style="color: #6c757d; font-size: 12px; margin-top: 10px;">Source: XposedOrNot (free dark web monitoring) | Privacy: Keccak-512 + k-anonymity</p>
                </div>
                """

            html += "</div>"

        html += "</div>"
        return html

    def _generate_recommendations_section(self, stats: Dict) -> str:
        """Generate actionable recommendations"""
        recommendations = []

        if stats['critical'] > 0:
            recommendations.append(f"<strong>URGENT:</strong> {stats['critical']} email(s) require immediate password changes and 2FA implementation")

        if stats['breached'] > 0:
            recommendations.append(f"Monitor {stats['breached']} breached email account(s) for suspicious activity and unauthorized access")

        if stats['high'] > 0:
            recommendations.append(f"Review and strengthen security posture for {stats['high']} high-risk email account(s)")

        recommendations.append("Implement regular password rotation policy (every 90 days)")
        recommendations.append("Enable Two-Factor Authentication (2FA) on all critical accounts")
        recommendations.append("Conduct security awareness training for users with compromised credentials")
        recommendations.append("Deploy email monitoring solution for real-time threat detection")

        if stats['breach_percentage'] > 20:
            recommendations.append("Consider comprehensive security audit given high breach rate")

        html = """
        <div class="content-section">
            <div class="recommendations">
                <h3>
                    <span>💡</span>
                    Security Recommendations
                </h3>
                <ul class="recommendation-list">
        """

        for rec in recommendations:
            html += f'<li class="recommendation-item">{rec}</li>'

        html += """
                </ul>
            </div>
        </div>
        """

        return html
