import threading
import time
import logging
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List
from collections import Counter

import numpy as np

warnings.filterwarnings('ignore')

# Library modules should not configure the root logger — main.py owns that
logger = logging.getLogger(__name__)

# Import module dependencies
from .ApplicationConfig import ApplicationConfig
from .EmailSecurityAnalyzer import EmailSecurityAnalyzer

class BulkProcessingEngine:
    """High-performance bulk email processing"""

    def __init__(self, analyzer: EmailSecurityAnalyzer, config: ApplicationConfig):
        self.analyzer = analyzer
        self.config = config
        self.results = []
        self.statistics = {}
        self.cancel_flag = threading.Event()
        self._results_lock = threading.Lock()

    def process_email_list(self, emails: List[str], progress_callback=None) -> Dict:
        """Process list of emails with multi-threading"""
        self.results = []
        self.cancel_flag.clear()
        total = len(emails)
        start_time = time.time()

        logger.info(f"Starting bulk analysis of {total} emails")

        # Scale workers based on list size to avoid API rate limits
        max_workers = min(int(self.config.max_workers), 5)

        # Submit individual emails for per-email progress updates
        completed = 0
        executor = ThreadPoolExecutor(max_workers=max_workers)
        try:
            futures = {}
            for email in emails:
                if self.cancel_flag.is_set():
                    break
                future = executor.submit(self._analyze_single, email)
                futures[future] = email

            for future in as_completed(futures):
                if self.cancel_flag.is_set():
                    # Cancel remaining pending futures so shutdown doesn't block
                    for f in futures:
                        f.cancel()
                    break

                try:
                    result = future.result(timeout=60)
                    with self._results_lock:
                        self.results.append(result)
                        completed += 1
                except Exception as e:
                    email_addr = futures[future]
                    with self._results_lock:
                        self.results.append({
                            'email': email_addr,
                            'error': str(e),
                            'risk_score': -1
                        })
                        completed += 1
                    logger.error(f"Analysis error for {email_addr}: {e}")

                if progress_callback:
                    progress_callback(completed, total)
        finally:
            # Use wait=False + cancel_futures=True to avoid hanging on cancel
            executor.shutdown(wait=False, cancel_futures=True)

        # Calculate statistics
        self.calculate_statistics()

        elapsed = time.time() - start_time
        logger.info(f"Processed {len(self.results)} emails in {elapsed:.2f} seconds")

        return {
            'results': self.results,
            'statistics': self.statistics,
            'processing_time': elapsed,
            'emails_processed': len(self.results)
        }

    def _analyze_single(self, email: str) -> Dict:
        """Analyze a single email (called from thread pool)"""
        try:
            return self.analyzer.analyze_email(email, full_analysis=True)
        except Exception as e:
            return {
                'email': email,
                'error': str(e),
                'risk_score': -1
            }

    def process_batch(self, emails: List[str]) -> List[Dict]:
        """Process a batch of emails"""
        batch_results = []

        for email in emails:
            if self.cancel_flag.is_set():
                break

            try:
                result = self.analyzer.analyze_email(email, full_analysis=True)
                batch_results.append(result)
            except Exception as e:
                batch_results.append({
                    'email': email,
                    'error': str(e),
                    'risk_score': -1
                })

        return batch_results

    def calculate_statistics(self):
        """Calculate comprehensive statistics"""
        valid_results = [r for r in self.results if r.get('risk_score', -1) >= 0]

        self.statistics = {
            'total': len(self.results),
            'valid': len(valid_results),
            'failed': len(self.results) - len(valid_results),
            'critical': sum(1 for r in valid_results if r.get('risk_level') == 'critical'),
            'high': sum(1 for r in valid_results if r.get('risk_level') == 'high'),
            'medium': sum(1 for r in valid_results if r.get('risk_level') == 'medium'),
            'low': sum(1 for r in valid_results if r.get('risk_level') == 'low'),
            'minimal': sum(1 for r in valid_results if r.get('risk_level') == 'minimal'),
            'avg_risk': np.mean([r.get('risk_score', 0) for r in valid_results]) if valid_results else 0,
            'with_threats': sum(1 for r in valid_results if r.get('threats')),
            'with_breaches': sum(1 for r in valid_results if (r.get('breach_info') or {}).get('found')),
            'domain_stats': self.calculate_domain_statistics(valid_results),
            'threat_distribution': self.calculate_threat_distribution(valid_results)
        }

    def calculate_domain_statistics(self, results: List[Dict]) -> Dict:
        """Calculate domain-based statistics"""
        domains = []
        for r in results:
            email = r.get('email', '')
            if '@' in email:
                parts = email.split('@')
                if len(parts) == 2 and parts[1]:
                    domains.append(parts[1])
        domain_counts = Counter(domains)

        return {
            'unique_domains': len(set(domains)),
            'top_domains': domain_counts.most_common(10),
            'suspicious_domains': sum(1 for r in results
                                      if 'suspicious_tld' in ((r.get('domain_reputation') or {}).get('flags') or []))
        }

    def calculate_threat_distribution(self, results: List[Dict]) -> Dict:
        """Calculate threat distribution"""
        all_threats = []
        for r in results:
            all_threats.extend([t.get('type', 'unknown') for t in (r.get('threats') or [])])

        return Counter(all_threats)

    def cancel(self):
        """Cancel processing"""
        self.cancel_flag.set()
