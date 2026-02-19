import os
import sys
import re
import logging
import warnings
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Optional
from collections import Counter

# Configure logging
warnings.filterwarnings('ignore')

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
from .MitreAttackFramework import MitreAttackFramework
from .MachineLearningEngine import MachineLearningEngine
from .ThreatIntelligenceEngine import ThreatIntelligenceEngine
from .DisposableEmailDetector import DisposableEmailDetector
from .TyposquattingDetector import TyposquattingDetector
from .DGADetector import DGADetector


class EmailSecurityAnalyzer:
    """Main email security analysis engine with enhanced MITRE mapping"""

    def __init__(self, config: ApplicationConfig):
        self.config = config
        self.mitre = MitreAttackFramework(config)  # Now uses enhanced framework
        self.ml_engine = MachineLearningEngine(config)
        self.threat_intel = ThreatIntelligenceEngine(config)
        self.disposable_detector = DisposableEmailDetector()
        self.typosquat_detector = TyposquattingDetector()
        self.dga_detector = DGADetector()
        self.analysis_cache = {}

    def analyze_email(
        self,
        email_address: str,
        full_analysis: bool = True,
        *,
        enable_ml: Optional[bool] = None,
        enable_threat_intel: Optional[bool] = None,
        password: Optional[str] = None,
    ) -> Dict:
        """Comprehensive email security analysis.

        Args:
            email_address: Email address to analyze.
            full_analysis: When False, performs a fast analysis (skips DNS/WHOIS/breach checks).
            enable_ml: Optional per-run override to disable ML predictions.
            enable_threat_intel: Optional per-run override to disable threat-intel lookups.
        """

        result = {
            'email': email_address,
            'timestamp': datetime.now().isoformat(),
            'risk_score': 0,
            'risk_level': 'low',
            'threats': [],
            'vulnerabilities': [],
            'breaches': {},
            'password_breach': {},
            'ml_predictions': {},
            'mitre_techniques': [],
            'dns_security': {},
            'domain_reputation': {},
            'dnsbl': {},
            'cert_transparency': {},
            'gravatar': {},
            'threatfox': {},
            'parked_domain': {},
            'dga_analysis': {},
            'recommendations': [],
            'detailed_analysis': {}
        }

        # Practical email length limit (RFC 5321 specifies 254, we allow up to 320 for edge cases)
        if len(email_address) > 320:
            result['risk_score'] = 100
            result['risk_level'] = 'critical'
            result['threats'].append({
                'type': 'invalid_format',
                'description': 'Email address exceeds maximum length',
                'severity': 'critical'
            })
            return result

        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, email_address):
            result['risk_score'] = 100
            result['risk_level'] = 'critical'
            result['threats'].append({
                'type': 'invalid_format',
                'description': 'Invalid email address format',
                'severity': 'critical'
            })
            return result

        domain = email_address.split('@')[1].lower()

        # Effective toggles
        threat_intel_enabled = True if enable_threat_intel is None else bool(enable_threat_intel)
        threat_intel_enabled = threat_intel_enabled and bool(getattr(self.config, 'enable_threat_feeds', True))

        deep_scan = bool(full_analysis) and threat_intel_enabled

        breach_enabled = deep_scan and bool(getattr(self.config, 'enable_breach_check', True))
        dns_enabled = deep_scan and bool(getattr(self.config, 'enable_dns', True))
        whois_enabled = deep_scan and bool(getattr(self.config, 'enable_whois', True))

        # ML predictions (cannot force-enable if globally disabled)
        ml_enabled = bool(getattr(self.config, 'enable_ml', True))
        if enable_ml is not None:
            ml_enabled = ml_enabled and bool(enable_ml)

        dnsbl_enabled = deep_scan and bool(getattr(self.config, 'enable_dnsbl', True))
        cert_ct_enabled = deep_scan and bool(getattr(self.config, 'enable_cert_transparency', True))
        gravatar_enabled = deep_scan and bool(getattr(self.config, 'enable_gravatar_check', True))
        threatfox_enabled = deep_scan and bool(getattr(self.config, 'enable_threatfox', True))
        parked_enabled = deep_scan and bool(getattr(self.config, 'enable_parked_detection', True))
        dga_enabled = bool(getattr(self.config, 'enable_dga_detection', True))

        result['analysis_flags'] = {
            'threat_intel': threat_intel_enabled,
            'deep_scan': deep_scan,
            'dns': dns_enabled,
            'whois': whois_enabled,
            'breach': breach_enabled,
            'ml': ml_enabled,
            'dnsbl': dnsbl_enabled,
            'cert_transparency': cert_ct_enabled,
            'gravatar': gravatar_enabled,
            'threatfox': threatfox_enabled,
            'parked_detection': parked_enabled,
            'dga_detection': dga_enabled,
        }

        # Run network lookups in parallel for speed
        def _check_domain_reputation():
            try:
                return self.threat_intel.check_domain_reputation(
                    domain, enabled=threat_intel_enabled, enable_whois=whois_enabled
                )
            except Exception as e:
                logger.debug(f"Domain reputation check failed: {e}")
                return {'score': 50, 'category': 'unknown', 'age': None, 'flags': []}

        def _check_dns():
            if not dns_enabled:
                return {}
            try:
                return self.threat_intel.check_dns_security(domain, enabled=True)
            except Exception as e:
                logger.debug(f"DNS security check failed: {e}")
                return {}

        def _check_breach():
            try:
                return self.threat_intel.check_email_breach(email_address, enabled=breach_enabled)
            except Exception as e:
                logger.debug(f"Breach check failed: {e}")
                return {
                    'found': False, 'count': 0, 'breaches': [], 'severity': 'low',
                    'details': [], 'mitigation_steps': [], 'data_classes': [], 'breach_dates': []
                }

        def _check_password():
            if not (password and breach_enabled):
                return {'found': False, 'details': '', 'recommendation': ''}
            try:
                return self.threat_intel.check_password_breach(password)
            except Exception as e:
                logger.debug(f"Password breach check failed: {e}")
                return {'found': False, 'details': 'Password breach check failed', 'recommendation': ''}

        def _check_dnsbl():
            if not dnsbl_enabled:
                return {}
            try:
                return self.threat_intel.check_dnsbl(domain, enabled=True)
            except Exception as e:
                logger.debug(f"DNSBL check failed: {e}")
                return {}

        def _check_cert_transparency():
            if not cert_ct_enabled:
                return {}
            try:
                return self.threat_intel.check_cert_transparency(domain, enabled=True)
            except Exception as e:
                logger.debug(f"Certificate transparency check failed: {e}")
                return {}

        def _check_gravatar():
            if not gravatar_enabled:
                return {}
            try:
                return self.threat_intel.check_gravatar(email_address, enabled=True)
            except Exception as e:
                logger.debug(f"Gravatar check failed: {e}")
                return {}

        def _check_threatfox():
            if not threatfox_enabled:
                return {}
            try:
                return self.threat_intel.check_threatfox(domain, enabled=True)
            except Exception as e:
                logger.debug(f"ThreatFox check failed: {e}")
                return {}

        def _check_parked():
            if not parked_enabled:
                return {}
            try:
                return self.threat_intel.check_parked_domain(domain, enabled=True)
            except Exception as e:
                logger.debug(f"Parked domain check failed: {e}")
                return {}

        with ThreadPoolExecutor(max_workers=10) as executor:
            fut_domain = executor.submit(_check_domain_reputation)
            fut_dns = executor.submit(_check_dns)
            fut_breach = executor.submit(_check_breach)
            fut_password = executor.submit(_check_password)
            fut_dnsbl = executor.submit(_check_dnsbl)
            fut_cert = executor.submit(_check_cert_transparency)
            fut_gravatar = executor.submit(_check_gravatar)
            fut_threatfox = executor.submit(_check_threatfox)
            fut_parked = executor.submit(_check_parked)

            # Timeouts prevent indefinite hangs if network calls get stuck
            try:
                result['domain_reputation'] = fut_domain.result(timeout=15)
            except Exception:
                result['domain_reputation'] = {'score': 50, 'category': 'unknown', 'age': None, 'flags': []}
            try:
                result['dns_security'] = fut_dns.result(timeout=12)
            except Exception:
                result['dns_security'] = {}
            try:
                result['breach_info'] = fut_breach.result(timeout=20)
            except Exception:
                result['breach_info'] = {
                    'found': False, 'count': 0, 'breaches': [], 'severity': 'low',
                    'details': [], 'mitigation_steps': [], 'data_classes': [], 'breach_dates': []
                }
            try:
                result['password_breach'] = fut_password.result(timeout=12)
            except Exception:
                result['password_breach'] = {'found': False, 'details': '', 'recommendation': ''}
            try:
                result['dnsbl'] = fut_dnsbl.result(timeout=15)
            except Exception:
                result['dnsbl'] = {}
            try:
                result['cert_transparency'] = fut_cert.result(timeout=12)
            except Exception:
                result['cert_transparency'] = {}
            try:
                result['gravatar'] = fut_gravatar.result(timeout=8)
            except Exception:
                result['gravatar'] = {}
            try:
                result['threatfox'] = fut_threatfox.result(timeout=12)
            except Exception:
                result['threatfox'] = {}
            try:
                result['parked_domain'] = fut_parked.result(timeout=8)
            except Exception:
                result['parked_domain'] = {}

        # Keep both keys for backwards compatibility
        result['breaches'] = result.get('breach_info') or {}

        # DGA detection (local, fast — no network call needed)
        if dga_enabled:
            try:
                result['dga_analysis'] = self.dga_detector.check_domain(domain)
            except Exception as e:
                logger.debug(f"DGA detection failed: {e}")
                result['dga_analysis'] = {}

        features = self.extract_features(email_address, domain, result)
        result['_ml_features'] = features  # Store per-result for feedback loop
        result['_ml_ensemble_score'] = 0.5  # Default; updated after prediction
        if ml_enabled:
            try:
                result['ml_predictions'] = self.ml_engine.predict_ensemble(features)
            except Exception as e:
                logger.error(f"ML prediction failed: {e}")
                result['ml_predictions'] = {
                    'ensemble': 0.5, 'is_malicious': False,
                    'anomaly_score': 0.0, 'precision_threshold': 0.5
                }
        else:
            result['ml_predictions'] = {'ensemble': 0.5}
        result['_ml_ensemble_score'] = result['ml_predictions'].get('ensemble', 0.5)

        # Always compute threats/risk/recommendations (even in fast mode)
        self.analyze_threats(result, email_address, domain)
        self.map_to_mitre(result)

        result['risk_score'] = self.calculate_risk_score(result)

        if result['risk_score'] >= 80:
            result['risk_level'] = 'critical'
        elif result['risk_score'] >= 60:
            result['risk_level'] = 'high'
        elif result['risk_score'] >= 40:
            result['risk_level'] = 'medium'
        elif result['risk_score'] >= 20:
            result['risk_level'] = 'low'
        else:
            result['risk_level'] = 'minimal'

        result['recommendations'] = self.generate_recommendations(result)
        result['detailed_analysis'] = self.generate_detailed_analysis(result)

        return result

    def extract_features(self, email: str, domain: str, analysis: Dict) -> np.ndarray:
        """Extract 44 real features for ML (no zero-padding).

        Feature order MUST match MachineLearningEngine.FEATURE_NAMES exactly.
        """
        features = []

        local_part = email.split('@')[0]

        # --- Email address features (10) ---
        features.append(len(email))
        features.append(len(local_part))
        features.append(1 if any(c.isdigit() for c in local_part) else 0)
        features.append(local_part.count('.'))
        features.append(local_part.count('_'))
        features.append(local_part.count('-'))
        features.append(1 if local_part and local_part[0].isdigit() else 0)
        features.append(len(set(local_part)))
        features.append(1 if local_part.lower() != local_part else 0)
        features.append(sum(1 for c in local_part if not c.isalnum()))

        # --- Domain features (9) ---
        features.append(len(domain))
        features.append(domain.count('.'))
        features.append(domain.count('-'))
        features.append(1 if any(c.isdigit() for c in domain) else 0)
        features.append(1 if domain.startswith('www.') else 0)
        features.append(len(domain.split('.')[-1]))
        features.append(1 if domain.endswith(('.tk', '.ml', '.ga', '.cf')) else 0)
        dom_rep = analysis.get('domain_reputation') or {}
        dom_age = dom_rep.get('age', None)
        dom_score = dom_rep.get('score', 50)

        try:
            dom_age_val = float(dom_age) if dom_age is not None else 365.0
        except (TypeError, ValueError):
            dom_age_val = 365.0

        try:
            dom_score_val = float(dom_score) if dom_score is not None else 50.0
        except (TypeError, ValueError):
            dom_score_val = 50.0

        features.append(dom_age_val)
        features.append(dom_score_val)

        # --- Domain flags (1) ---
        features.append(1 if 'phishing' in (analysis.get('domain_reputation') or {}).get('flags', []) else 0)

        # --- DNS features (6) ---
        dns = analysis.get('dns_security') or {}
        features.append(1 if dns.get('spf') else 0)
        features.append(1 if dns.get('dmarc') else 0)
        features.append(1 if dns.get('dkim') else 0)
        features.append(1 if dns.get('mx') else 0)
        features.append(1 if dns.get('dnssec') else 0)
        features.append(len(dns.get('issues', [])))

        # --- Breach features (3) ---
        features.append(1 if (analysis.get('breaches') or {}).get('found') else 0)
        breach_count = (analysis.get('breaches') or {}).get('count', 0)
        try:
            breach_count = int(breach_count) if breach_count is not None else 0
        except (TypeError, ValueError):
            breach_count = 0
        features.append(breach_count)

        dns_score = dns.get('score', 0)
        try:
            dns_score = float(dns_score) if dns_score is not None else 0.0
        except (TypeError, ValueError):
            dns_score = 0.0
        features.append(dns_score)

        # --- DNS extra (1) ---
        features.append(1 if (dns.get('records') or {}).get('a') else 0)

        # --- Pattern features (10) ---
        suspicious_words = ['admin', 'security', 'update', 'verify', 'suspend', 'urgent', 'click', 'winner']
        features.append(sum(1 for word in suspicious_words if word in local_part.lower()))
        features.append(sum(1 for word in suspicious_words if word in domain.lower()))
        features.append(1 if re.search(r'\d{4,}', local_part) else 0)
        features.append(1 if re.search(r'[A-Z]{5,}', local_part) else 0)
        features.append(1 if len(local_part) > 0 and local_part.count(local_part[0]) > len(local_part) * 0.5 else 0)
        features.append(1 if any(c in local_part for c in ['$', '%', '!', '#']) else 0)
        features.append(1 if len(set(domain.split('.'))) != len(domain.split('.')) else 0)
        features.append(1 if any(len(part) > 20 for part in domain.split('.')) else 0)
        features.append(1 if domain.count('.') > 3 else 0)
        features.append(1 if re.search(r'[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}', domain) else 0)

        # --- NEW features (4) ---
        # Disposable email domain detection
        features.append(1 if self.disposable_detector.is_disposable(domain) else 0)
        # Typosquatting risk score (0.0 - 1.0)
        features.append(self.typosquat_detector.get_typosquat_score(domain))
        # Shannon entropy of local part (high = random/automated)
        features.append(self.typosquat_detector.get_shannon_entropy(local_part))
        # Is a known free email provider
        features.append(1 if self.typosquat_detector.is_free_email_provider(domain) else 0)

        result = np.array(features, dtype=np.float64)
        return result

    def analyze_threats(self, result: Dict, email: str, domain: str):
        """Analyze potential threats with enhanced MITRE mapping"""

        dom_rep = result.get('domain_reputation') or {}
        if 'phishing' in (dom_rep.get('flags') or []):
            result['threats'].append({
                'type': 'phishing',
                'description': 'Domain associated with phishing attacks',
                'severity': 'critical',
                'confidence': 0.9
            })

        if 'malware' in (dom_rep.get('flags') or []):
            result['threats'].append({
                'type': 'malware',
                'description': 'Domain associated with malware distribution',
                'severity': 'critical',
                'confidence': 0.9
            })

        if 'spam' in (dom_rep.get('flags') or []):
            result['threats'].append({
                'type': 'spam',
                'description': 'Domain associated with spam',
                'severity': 'medium',
                'confidence': 0.8
            })

        dns = result.get('dns_security') or {}
        dns_checked = result.get('analysis_flags', {}).get('dns', False)
        if dns_checked:
            if not dns.get('spf'):
                result['vulnerabilities'].append({
                    'type': 'missing_spf',
                    'description': 'No SPF record - vulnerable to email spoofing',
                    'severity': 'medium',
                    'remediation': 'Configure SPF record for the domain'
                })

            if not dns.get('dmarc'):
                result['vulnerabilities'].append({
                    'type': 'missing_dmarc',
                    'description': 'No DMARC policy - limited email authentication',
                    'severity': 'medium',
                    'remediation': 'Implement DMARC policy'
                })

            if not dns.get('dnssec'):
                result['vulnerabilities'].append({
                    'type': 'missing_dnssec',
                    'description': 'DNSSEC not enabled - vulnerable to DNS spoofing',
                    'severity': 'low',
                    'remediation': 'Enable DNSSEC for the domain'
                })

        ml_preds = result.get('ml_predictions') or {}
        ml_score = ml_preds.get('ensemble', 0.5)
        # Use the model's precision-optimized threshold (is_malicious flag)
        # Require corroboration for moderate scores; allow standalone only for very high scores
        is_malicious = ml_preds.get('is_malicious', False)
        has_other_threats = any(t['type'] not in ('ml_high_risk', 'ml_medium_risk') for t in result.get('threats', []))
        if is_malicious and ml_score > 0.95:
            # Very high confidence: flag even without corroboration
            result['threats'].append({
                'type': 'ml_high_risk',
                'description': 'Machine learning models detected high risk patterns',
                'severity': 'high',
                'confidence': ml_score
            })
        elif is_malicious and ml_score > 0.85 and has_other_threats:
            # High confidence + corroborated by other signals
            result['threats'].append({
                'type': 'ml_high_risk',
                'description': 'Machine learning models detected high risk patterns',
                'severity': 'high',
                'confidence': ml_score
            })
        elif is_malicious and ml_score > 0.8:
            # Moderate confidence: flag as medium risk
            result['threats'].append({
                'type': 'ml_medium_risk',
                'description': 'Machine learning models detected suspicious patterns',
                'severity': 'medium',
                'confidence': ml_score
            })

        # Isolation Forest anomaly detection
        anomaly_score = ml_preds.get('anomaly_score', 0.0)
        if anomaly_score > 0.7:
            result['threats'].append({
                'type': 'anomaly_detected',
                'description': f'Unsupervised anomaly detection flagged unusual patterns (score: {anomaly_score:.2f})',
                'severity': 'medium',
                'confidence': anomaly_score
            })

        breaches = result.get('breaches') or {}
        if breaches.get('found'):
            severity = breaches.get('severity') or 'medium'
            result['threats'].append({
                'type': 'data_breach',
                'description': f'Email found in {breaches.get("count", 0)} data breach(es)',
                'severity': severity,
                'confidence': 1.0
            })

        dom_age = dom_rep.get('age')
        if dom_age is not None and isinstance(dom_age, (int, float)) and dom_age < 30:
            result['threats'].append({
                'type': 'new_domain',
                'description': 'Very new domain (less than 30 days old)',
                'severity': 'medium',
                'confidence': 0.8
            })

        if 'suspicious_tld' in (dom_rep.get('flags') or []):
            result['threats'].append({
                'type': 'suspicious_tld',
                'description': 'Domain uses suspicious top-level domain',
                'severity': 'medium',
                'confidence': 0.7
            })

        # Disposable email detection (domain already set from function parameter)
        if domain and self.disposable_detector.is_disposable(domain):
            result['threats'].append({
                'type': 'disposable_email',
                'description': 'Email uses a known disposable/temporary email service',
                'severity': 'high',
                'confidence': 0.95
            })

        # Typosquatting detection
        if domain:
            typosquat_result = self.typosquat_detector.check_domain(domain)
            if typosquat_result.get('is_typosquat'):
                target = typosquat_result.get('target_domain', 'unknown')
                attack_type = typosquat_result.get('attack_type', 'unknown')
                result['threats'].append({
                    'type': 'typosquatting',
                    'description': f'Domain appears to impersonate {target} ({attack_type})',
                    'severity': 'critical',
                    'confidence': typosquat_result.get('similarity', 0.8)
                })
            result['typosquat_info'] = typosquat_result

        # DNSBL blacklist detection
        dnsbl = result.get('dnsbl') or {}
        if dnsbl.get('listed'):
            listed_count = dnsbl.get('listed_count', 0)
            result['threats'].append({
                'type': 'dnsbl_listed',
                'description': f'Domain mail server listed in {listed_count} DNS blacklist(s)',
                'severity': 'high',
                'confidence': 0.9
            })

        # ThreatFox IOC detection
        threatfox = result.get('threatfox') or {}
        if threatfox.get('found'):
            ioc_count = threatfox.get('ioc_count', 0)
            malware_names = [ioc.get('malware', '') for ioc in threatfox.get('iocs', [])[:3] if ioc.get('malware')]
            malware_str = ', '.join(malware_names) if malware_names else 'unknown'
            result['threats'].append({
                'type': 'threatfox_ioc',
                'description': f'Domain found in ThreatFox IOC database ({ioc_count} IOC(s): {malware_str})',
                'severity': 'critical',
                'confidence': 0.95
            })

        # DGA detection
        dga = result.get('dga_analysis') or {}
        if dga.get('is_dga'):
            result['threats'].append({
                'type': 'dga_detected',
                'description': f'Domain name appears algorithmically generated (DGA score: {float(dga.get("dga_score") or 0):.2f})',
                'severity': 'high',
                'confidence': float(dga.get('dga_score') or 0.65)
            })

        # Parked domain detection
        parked = result.get('parked_domain') or {}
        if parked.get('is_parked'):
            result['threats'].append({
                'type': 'parked_domain',
                'description': 'Domain appears to be parked or for sale',
                'severity': 'medium',
                'confidence': 0.7
            })

        # Additional DNS vulnerabilities
        if dns_checked:
            dns_data = result.get('dns_security') or {}
            if not dns_data.get('bimi'):
                result['vulnerabilities'].append({
                    'type': 'missing_bimi',
                    'description': 'No BIMI record - no verified brand indicator for email',
                    'severity': 'low',
                    'remediation': 'Configure BIMI record for brand verification'
                })
            if not dns_data.get('mta_sts'):
                result['vulnerabilities'].append({
                    'type': 'missing_mta_sts',
                    'description': 'No MTA-STS policy - email transport not enforcing TLS',
                    'severity': 'low',
                    'remediation': 'Implement MTA-STS policy for enforced TLS'
                })
            if not dns_data.get('tls_rpt'):
                result['vulnerabilities'].append({
                    'type': 'missing_tls_rpt',
                    'description': 'No TLS-RPT policy - no reporting for mail transport failures',
                    'severity': 'low',
                    'remediation': 'Implement TLS-RPT policy for transport security visibility'
                })

    def map_to_mitre(self, result: Dict):
        """Map threats to MITRE ATT&CK — fully dynamic via semantic search,
        rule-based fallback only when sentence-transformers unavailable."""
        techniques = []
        technique_details = []
        seen_ids = set()

        if self.mitre.semantic_enabled:
            # --- Fully dynamic: sentence-transformer maps threats to 811 MITRE techniques ---

            # 1) Overall semantic search with combined threat description
            threat_description = self.create_threat_description(result)
            if threat_description.strip():
                semantic_techniques = self.mitre.find_techniques_by_description(threat_description, top_k=5)
                for tech in semantic_techniques:
                    if tech['similarity'] > 40 and tech['id'] not in seen_ids:
                        seen_ids.add(tech['id'])
                        techniques.append(tech['id'])
                        technique_details.append({
                            'id': tech['id'],
                            'name': tech['name'],
                            'tactic': tech['tactic'],
                            'severity': tech['severity'],
                            'description': tech['description'],
                            'similarity': tech['similarity']
                        })

            # 2) Per-threat semantic search for targeted results
            for threat in result.get('threats', []):
                desc = threat.get('description', '')
                if desc:
                    per_threat_results = self.mitre.find_techniques_by_description(desc, top_k=3)
                    for tech in per_threat_results:
                        if tech['similarity'] > 40 and tech['id'] not in seen_ids:
                            seen_ids.add(tech['id'])
                            techniques.append(tech['id'])
                            technique_details.append({
                                'id': tech['id'],
                                'name': tech['name'],
                                'tactic': tech['tactic'],
                                'severity': tech['severity'],
                                'description': tech['description'],
                                'similarity': tech['similarity']
                            })

        else:
            # --- Offline fallback: rule-based mapping (no sentence-transformers) ---
            for threat in result.get('threats', []):
                threat_techniques = self.mitre.map_threat_to_techniques(threat['type'])
                for tech_id in threat_techniques:
                    if tech_id not in seen_ids:
                        seen_ids.add(tech_id)
                        techniques.append(tech_id)
                        tech_info = self.mitre.get_technique_details(tech_id)
                        if tech_info:
                            technique_details.append({
                                'id': tech_id,
                                'name': tech_info.get('name'),
                                'tactic': tech_info.get('tactic'),
                                'severity': tech_info.get('severity'),
                                'description': tech_info.get('description'),
                                'similarity': 100  # Rule-based match
                            })

        result['mitre_techniques'] = list(set(techniques))
        # Sort by similarity descending so reports show most relevant first
        technique_details.sort(key=lambda t: t.get('similarity', 0), reverse=True)
        result['mitre_details'] = technique_details

    def create_threat_description(self, result: Dict) -> str:
        """Create comprehensive threat description for semantic search"""
        descriptions = []

        for threat in result.get('threats', []):
            descriptions.append(threat.get('description', ''))

        for vulnerability in result.get('vulnerabilities', []):
            descriptions.append(vulnerability.get('description', ''))

        # Enhanced breach description for better MITRE mapping
        breach_info = result.get('breach_info') or {}
        if breach_info.get('found'):
            breach_count = breach_info.get('count', 0)
            severity = breach_info.get('severity', 'medium')
            data_classes = breach_info.get('data_classes', [])

            # Create detailed breach description for MITRE mapping
            breach_desc = f"Email found in {breach_count} data breach(es) "
            breach_desc += f"Credentials potentially compromised with {severity} severity "

            if 'Passwords' in data_classes:
                breach_desc += "Passwords exposed enabling credential stuffing and account takeover "
            if 'Email addresses' in data_classes:
                breach_desc += "Email addresses exposed for targeted phishing campaigns "
            if any(dc in data_classes for dc in ['Credit cards', 'Bank account numbers', 'Social security numbers']):
                breach_desc += "Financial data exposed enabling identity theft and fraud "

            # Add breach names for context
            if breach_info.get('breaches'):
                breach_desc += f"Breaches: {', '.join(str(b) for b in breach_info['breaches'][:3])} "

            descriptions.append(breach_desc)

        # Keep backward compatibility
        if (result.get('breaches') or {}).get('found') and not breach_info.get('found'):
            descriptions.append("Credentials potentially compromised in data breach")

        domain_rep = result.get('domain_reputation') or {}
        if domain_rep.get('flags'):
            descriptions.append(f"Domain has indicators: {', '.join(str(f) for f in domain_rep['flags'])}")

        return " ".join(descriptions)

    def calculate_risk_score(self, result: Dict) -> int:
        """Calculate comprehensive risk score"""
        import math
        score = 0

        if (result.get('breaches') or {}).get('found'):
            breach_count = (result.get('breaches') or {}).get('count', 0)
            score += min(breach_count * 5, 25)

        if (result.get('password_breach') or {}).get('found'):
            score += 20

        for threat in (result.get('threats') or []):
            sev = threat.get('severity', 'low')
            if sev == 'critical':
                score += 15
            elif sev == 'high':
                score += 10
            elif sev == 'medium':
                score += 5
            else:
                score += 2

        def to_float(value, default: float) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return float(default)

        flags = result.get('analysis_flags', {}) or {}

        if flags.get('threat_intel', True):
            rep_score = to_float((result.get('domain_reputation') or {}).get('score', 50), 50.0)
            if not math.isnan(rep_score):
                score += int((50 - rep_score) * 0.4)

        if flags.get('dns', False):
            dns_score = to_float((result.get('dns_security') or {}).get('score', 0), 0.0)
            if not math.isnan(dns_score):
                score += int((100 - dns_score) * 0.1)

        if flags.get('ml', True):
            ml_score = to_float((result.get('ml_predictions') or {}).get('ensemble', 0.5), 0.5)
            if not math.isnan(ml_score):
                # Only add ML-based risk when model is confident AND corroborated
                # Subtract 0.5 baseline so neutral predictions add 0 risk
                ml_risk = max(0.0, ml_score - 0.5) * 2  # Scale 0.5-1.0 → 0.0-1.0
                score += int(ml_risk * 15)  # 15% max ML weight (conservative for synthetic-trained model)

        # Advanced check signals
        if (result.get('dnsbl') or {}).get('listed'):
            score += 15

        if (result.get('threatfox') or {}).get('found'):
            score += 20

        if (result.get('dga_analysis') or {}).get('is_dga'):
            score += 10

        if (result.get('parked_domain') or {}).get('is_parked'):
            score += 5

        return max(0, min(score, 100))

    def generate_recommendations(self, result: Dict) -> List[str]:
        """Generate security recommendations"""
        recommendations = []
        risk_level = result['risk_level']

        if risk_level == 'critical':
            recommendations.extend([
                "🚨 IMMEDIATE ACTION REQUIRED",
                "🔐 Change password immediately on all accounts using this email",
                "🔒 Enable two-factor authentication (2FA) on all critical accounts",
                "📧 Consider abandoning this email address for sensitive accounts",
                "🔍 Check all accounts for unauthorized access or suspicious activity",
                "💳 Monitor financial statements and credit reports",
                "🛡️ Use a password manager to create unique passwords for each account"
            ])
        elif risk_level == 'high':
            recommendations.extend([
                "⚠️ HIGH RISK DETECTED",
                "🔑 Update passwords on all important accounts",
                "🔒 Enable 2FA wherever possible",
                "📱 Review and update account recovery options",
                "📊 Monitor account activity closely",
                "🛡️ Consider using a more secure email provider"
            ])
        elif risk_level == 'medium':
            recommendations.extend([
                "🟡 MODERATE RISK IDENTIFIED",
                "🔑 Consider updating passwords on sensitive accounts",
                "🛡️ Review and strengthen security settings",
                "📊 Monitor account activity regularly",
                "🔒 Enable 2FA on important accounts"
            ])
        else:
            recommendations.extend([
                "✅ Low risk detected",
                "🔑 Continue using strong, unique passwords",
                "👀 Stay vigilant for phishing attempts",
                "🔒 Consider enabling 2FA for added security"
            ])

        if (result.get('breaches') or {}).get('found'):
            count = (result.get('breaches') or {}).get('count', 0)
            recommendations.append(f"📚 Email found in {count} data breach(es) - change all passwords")

        if (result.get('password_breach') or {}).get('found'):
            recommendations.append("🔑 CRITICAL: The checked password was found in breach databases - stop using it immediately")
            recommendations.append("🔐 Generate a new unique password using a password manager")

        dns = result.get('dns_security') or {}
        dns_checked = result.get('analysis_flags', {}).get('dns', False)
        if dns_checked:
            if not dns.get('spf'):
                recommendations.append("📧 Domain lacks SPF record - contact domain administrator")

            if not dns.get('dmarc'):
                recommendations.append("🛡️ No DMARC policy - email authentication is limited")

        dom_rep = result.get('domain_reputation') or {}
        if 'very_new_domain' in (dom_rep.get('flags') or []):
            recommendations.append("🆕 Very new domain - exercise extra caution")

        if 'suspicious_tld' in (dom_rep.get('flags') or []):
            recommendations.append("⚠️ Suspicious domain extension - verify legitimacy")

        ml_score = (result.get('ml_predictions') or {}).get('ensemble', 0.5)
        if ml_score > 0.7:
            recommendations.append("🤖 AI models detect high risk patterns - be extremely cautious")

        # Add MITRE-based recommendations if semantic search found matches
        if result.get('mitre_details'):
            high_confidence_techniques = [t for t in result['mitre_details'] if t.get('similarity', 0) > 85]
            if high_confidence_techniques:
                recommendations.append(
                    f"🎯 MITRE ATT&CK: {len(high_confidence_techniques)} techniques identified - review security posture")

        # Advanced check recommendations
        if (result.get('dnsbl') or {}).get('listed'):
            count = (result.get('dnsbl') or {}).get('listed_count', 0)
            recommendations.append(f"🚫 Domain mail server is blacklisted in {count} DNSBL(s) - emails may be blocked")

        if (result.get('threatfox') or {}).get('found'):
            recommendations.append("☠️ Domain found in ThreatFox IOC database - do NOT interact with emails from this domain")

        if (result.get('dga_analysis') or {}).get('is_dga'):
            recommendations.append("🤖 Domain name appears algorithmically generated (possible malware C2) - exercise extreme caution")

        if (result.get('parked_domain') or {}).get('is_parked'):
            recommendations.append("🅿️ Domain appears to be parked/for sale - unlikely to be a legitimate sender")

        if not (result.get('gravatar') or {}).get('has_profile'):
            if risk_level in ('high', 'critical'):
                recommendations.append("👤 No Gravatar profile found - may indicate a throwaway or fake account")

        dns_data = result.get('dns_security') or {}
        if dns_checked and dns_data and not dns_data.get('mta_sts'):
            recommendations.append("🔐 Domain lacks MTA-STS policy - email transport may not enforce TLS encryption")

        return recommendations

    def generate_detailed_analysis(self, result: Dict) -> Dict:
        """Generate detailed analysis report"""
        analysis = {
            'summary': {},
            'technical_details': {},
            'threat_analysis': {},
            'security_posture': {},
            'mitre_mapping': {}  # New section for MITRE details
        }

        analysis['summary'] = {
            'overall_risk': result['risk_level'],
            'risk_score': result['risk_score'],
            'total_threats': len(result.get('threats', [])),
            'total_vulnerabilities': len(result.get('vulnerabilities', [])),
            'breach_status': 'COMPROMISED' if (result.get('breaches') or {}).get('found') else 'CLEAN',
            'domain_trust': (result.get('domain_reputation') or {}).get('category', 'unknown')
        }

        analysis['technical_details'] = {
            'domain_age_days': (result.get('domain_reputation') or {}).get('age'),
            'dns_security_score': (result.get('dns_security') or {}).get('score', 0),
            'ml_confidence': (result.get('ml_predictions') or {}).get('ensemble', 0.5),
            'mitre_techniques_count': len(result.get('mitre_techniques', []))
        }

        threat_counts = Counter(t.get('severity', 'low') for t in (result.get('threats') or []))
        analysis['threat_analysis'] = {
            'critical_threats': threat_counts.get('critical', 0),
            'high_threats': threat_counts.get('high', 0),
            'medium_threats': threat_counts.get('medium', 0),
            'low_threats': threat_counts.get('low', 0)
        }

        dns = result.get('dns_security') or {}
        analysis['security_posture'] = {
            'email_authentication': {
                'spf': dns.get('spf', False),
                'dmarc': dns.get('dmarc', False),
                'dkim': dns.get('dkim', False)
            },
            'dns_security': {
                'dnssec': dns.get('dnssec', False),
                'mx_records': dns.get('mx', False)
            },
            'domain_reputation': {
                'score': (result.get('domain_reputation') or {}).get('score', 0),
                'flags': (result.get('domain_reputation') or {}).get('flags', [])
            }
        }

        # Add MITRE mapping details
        if result.get('mitre_details'):
            analysis['mitre_mapping'] = {
                'semantic_search_enabled': self.mitre.semantic_enabled,
                'total_techniques': len(result.get('mitre_techniques', [])),
                'high_confidence_matches': len([t for t in result.get('mitre_details', [])
                                                if t.get('similarity', 0) > 85]),
                'tactics_covered': list(set(t.get('tactic', 'Unknown')
                                            for t in result.get('mitre_details', [])))
            }

        # Advanced security checks summary
        analysis['advanced_checks'] = {
            'dnsbl_listed': (result.get('dnsbl') or {}).get('listed', False),
            'dnsbl_count': (result.get('dnsbl') or {}).get('listed_count', 0),
            'cert_transparency': (result.get('cert_transparency') or {}).get('found', False),
            'cert_count': (result.get('cert_transparency') or {}).get('cert_count', 0),
            'gravatar_profile': (result.get('gravatar') or {}).get('has_profile', False),
            'threatfox_found': (result.get('threatfox') or {}).get('found', False),
            'threatfox_iocs': (result.get('threatfox') or {}).get('ioc_count', 0),
            'parked_domain': (result.get('parked_domain') or {}).get('is_parked', False),
            'dga_detected': (result.get('dga_analysis') or {}).get('is_dga', False),
            'dga_score': (result.get('dga_analysis') or {}).get('dga_score', 0.0),
            'bimi': (result.get('dns_security') or {}).get('bimi', False),
            'mta_sts': (result.get('dns_security') or {}).get('mta_sts', False),
            'tls_rpt': (result.get('dns_security') or {}).get('tls_rpt', False),
        }

        return analysis

    def submit_feedback(self, email: str, user_label: int, features=None, ensemble_score: float = 0.5):
        """Submit user feedback for ML retraining.

        Args:
            email: The email address that was analyzed
            user_label: 0=safe, 1=malicious
            features: The ML feature array for this specific email (avoids race conditions)
            ensemble_score: The ML ensemble score for this specific result (avoids stale prediction_history)
        """
        if features is None:
            logger.warning("submit_feedback called without features — skipping")
            return
        if self.ml_engine:
            self.ml_engine.submit_feedback(email, features, ensemble_score, user_label)
