"""
Typosquatting / Homoglyph Detector

Detects email domains that are visually similar to legitimate email providers,
which is a common phishing and abuse technique.

Uses Levenshtein distance and homoglyph mapping to identify:
  - Misspelled domains (gmial.com, yahooo.com)
  - Character substitution (g00gle.com, micros0ft.com)
  - Unicode homoglyphs (gοοgle.com using Greek 'o')
  - TLD swaps (gmail.co instead of gmail.com)
"""

import logging
import math
import re
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Top email providers and their domains (targets for typosquatting)
LEGITIMATE_PROVIDERS: Dict[str, str] = {
    # Major providers
    'gmail.com': 'Google Gmail',
    'yahoo.com': 'Yahoo Mail',
    'outlook.com': 'Microsoft Outlook',
    'hotmail.com': 'Microsoft Hotmail',
    'live.com': 'Microsoft Live',
    'msn.com': 'Microsoft MSN',
    'aol.com': 'AOL',
    'icloud.com': 'Apple iCloud',
    'me.com': 'Apple Me',
    'mac.com': 'Apple Mac',
    'protonmail.com': 'ProtonMail',
    'proton.me': 'Proton',
    'zoho.com': 'Zoho',
    'yandex.com': 'Yandex',
    'mail.com': 'Mail.com',
    'gmx.com': 'GMX',
    'gmx.net': 'GMX',
    'fastmail.com': 'Fastmail',
    'tutanota.com': 'Tutanota',
    'tuta.io': 'Tuta',

    # ISP email
    'comcast.net': 'Comcast',
    'verizon.net': 'Verizon',
    'att.net': 'AT&T',
    'cox.net': 'Cox',
    'charter.net': 'Charter',
    'sbcglobal.net': 'SBCGlobal',
    'earthlink.net': 'EarthLink',
    'bellsouth.net': 'BellSouth',

    # Business email
    'microsoft.com': 'Microsoft',
    'google.com': 'Google',
    'apple.com': 'Apple',
    'amazon.com': 'Amazon',
    'facebook.com': 'Facebook',
    'paypal.com': 'PayPal',
    'chase.com': 'Chase Bank',
    'bankofamerica.com': 'Bank of America',
    'wellsfargo.com': 'Wells Fargo',
    'citi.com': 'Citibank',
    'linkedin.com': 'LinkedIn',
    'twitter.com': 'Twitter/X',
    'netflix.com': 'Netflix',
    'ebay.com': 'eBay',
    'dropbox.com': 'Dropbox',
}

# Known free email providers (not suspicious)
FREE_EMAIL_PROVIDERS = {
    'gmail.com', 'yahoo.com', 'outlook.com', 'hotmail.com', 'live.com',
    'msn.com', 'aol.com', 'icloud.com', 'me.com', 'mac.com',
    'protonmail.com', 'proton.me', 'zoho.com', 'yandex.com',
    'mail.com', 'gmx.com', 'gmx.net', 'fastmail.com',
    'tutanota.com', 'tuta.io', 'yahoo.co.uk', 'yahoo.co.in',
    'yahoo.co.jp', 'googlemail.com', 'outlook.co.uk',
    'hotmail.co.uk', 'hotmail.fr', 'hotmail.de', 'hotmail.it',
    'live.co.uk', 'live.fr', 'live.de', 'live.it',
    'ymail.com', 'rocketmail.com',
    'comcast.net', 'verizon.net', 'att.net', 'cox.net',
    'charter.net', 'sbcglobal.net', 'earthlink.net', 'bellsouth.net',
    'optonline.net', 'frontier.com', 'windstream.net',
    'mail.ru', 'inbox.ru', 'bk.ru', 'list.ru',
    'web.de', 'gmx.de', 'freenet.de', 't-online.de',
    'orange.fr', 'wanadoo.fr', 'laposte.net', 'free.fr',
    'virgilio.it', 'libero.it', 'alice.it', 'tin.it',
    'rediffmail.com', '163.com', '126.com', 'sina.com',
    'naver.com', 'daum.net', 'hanmail.net',
}

# Unicode homoglyph mapping (characters that look like ASCII equivalents)
HOMOGLYPHS = {
    'a': ['\u0430', '\u00e0', '\u00e1', '\u00e2', '\u00e3', '\u00e4'],  # Cyrillic а, etc.
    'c': ['\u0441', '\u00e7', '\u0188'],  # Cyrillic с
    'e': ['\u0435', '\u00e8', '\u00e9', '\u00ea', '\u00eb'],  # Cyrillic е
    'i': ['\u0456', '\u00ec', '\u00ed', '\u00ee', '\u00ef', '\u0131'],  # Cyrillic і
    'o': ['\u043e', '\u00f2', '\u00f3', '\u00f4', '\u00f5', '\u00f6', '\u0030'],  # Cyrillic о, digit 0
    'p': ['\u0440'],  # Cyrillic р
    's': ['\u0455'],  # Cyrillic ѕ
    'x': ['\u0445'],  # Cyrillic х
    'y': ['\u0443'],  # Cyrillic у
    'n': ['\u0578'],  # Armenian ո
    'l': ['1', '\u006c', '\u0131'],  # digit 1, lowercase L
    '0': ['o', 'O', '\u043e', '\u041e'],  # letter o/O, Cyrillic о/О
    '1': ['l', 'I', '\u006c', '\u0049'],  # letter l/I
}


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Compute the Levenshtein (edit) distance between two strings."""
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def _normalized_similarity(s1: str, s2: str) -> float:
    """Return similarity ratio 0.0-1.0 based on Levenshtein distance."""
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    distance = _levenshtein_distance(s1, s2)
    return 1.0 - (distance / max_len)


class TyposquattingDetector:
    """Detects typosquatting and homoglyph attacks on email domains."""

    def __init__(self):
        self.legitimate_domains = dict(LEGITIMATE_PROVIDERS)
        self.free_providers = set(FREE_EMAIL_PROVIDERS)
        logger.info(f"TyposquattingDetector initialized with {len(self.legitimate_domains)} target domains")

    def is_free_email_provider(self, domain: str) -> bool:
        """Check if a domain is a known free email provider."""
        return domain.lower().strip() in self.free_providers

    def check_domain(self, domain: str) -> Dict:
        """
        Check a domain for typosquatting indicators.

        Returns:
            Dict with keys:
                - is_typosquat: bool - True if likely a typosquat
                - similarity: float - 0.0 to 1.0 similarity to closest legitimate domain
                - target_domain: str - the legitimate domain it most resembles
                - target_provider: str - the provider name
                - distance: int - Levenshtein edit distance
                - has_homoglyphs: bool - contains Unicode lookalike characters
                - attack_type: str - type of attack detected (if any)
        """
        domain = domain.lower().strip()
        result = {
            'is_typosquat': False,
            'similarity': 0.0,
            'target_domain': '',
            'target_provider': '',
            'distance': 999,
            'has_homoglyphs': False,
            'attack_type': 'none',
        }

        # If it IS a legitimate domain, it's not a typosquat
        if domain in self.legitimate_domains:
            result['similarity'] = 0.0
            result['target_domain'] = domain
            result['target_provider'] = self.legitimate_domains[domain]
            result['distance'] = 0
            return result

        # Check for homoglyphs first
        has_homoglyphs = self._check_homoglyphs(domain)
        result['has_homoglyphs'] = has_homoglyphs

        # Find the closest legitimate domain
        best_similarity = 0.0
        best_domain = ''
        best_provider = ''
        best_distance = 999

        for legit_domain, provider in self.legitimate_domains.items():
            # Compare full domain
            sim = _normalized_similarity(domain, legit_domain)
            dist = _levenshtein_distance(domain, legit_domain)

            if sim > best_similarity:
                best_similarity = sim
                best_domain = legit_domain
                best_provider = provider
                best_distance = dist

            # Also compare just the base domain (before TLD)
            domain_base = domain.rsplit('.', 1)[0] if '.' in domain else domain
            legit_base = legit_domain.rsplit('.', 1)[0] if '.' in legit_domain else legit_domain

            base_sim = _normalized_similarity(domain_base, legit_base)
            if base_sim > best_similarity:
                best_similarity = base_sim
                best_domain = legit_domain
                best_provider = provider
                best_distance = _levenshtein_distance(domain, legit_domain)

        result['similarity'] = best_similarity
        result['target_domain'] = best_domain
        result['target_provider'] = best_provider
        result['distance'] = best_distance

        # Determine if this is a typosquat
        if has_homoglyphs and best_similarity > 0.7:
            result['is_typosquat'] = True
            result['attack_type'] = 'homoglyph'
        elif best_distance == 1 and best_similarity >= 0.85:
            # Single character edit from a known domain - very suspicious
            result['is_typosquat'] = True
            result['attack_type'] = 'single_char_edit'
        elif best_distance == 2 and best_similarity >= 0.75:
            # Two character edits - suspicious (covers character swaps like gmial→gmail)
            result['is_typosquat'] = True
            result['attack_type'] = 'double_char_edit'
        elif best_similarity >= 0.90 and domain != best_domain:
            # Very high similarity
            result['is_typosquat'] = True
            result['attack_type'] = 'high_similarity'
        elif self._check_tld_swap(domain, best_domain):
            result['is_typosquat'] = True
            result['attack_type'] = 'tld_swap'

        return result

    def _check_homoglyphs(self, domain: str) -> bool:
        """Check if the domain contains Unicode homoglyph characters."""
        for char in domain:
            # Check if character is non-ASCII
            if ord(char) > 127:
                return True
            # Check common digit-letter substitutions in the base domain
        domain_base = domain.rsplit('.', 1)[0] if '.' in domain else domain
        # Patterns like g00gle, micros0ft, yah00
        if re.search(r'[a-z]0[a-z]', domain_base):  # letter-zero-letter
            return True
        if re.search(r'[a-z]1[a-z]', domain_base) and '1' not in domain_base.split('.')[0][:1]:
            return True
        return False

    def _check_tld_swap(self, domain: str, target: str) -> bool:
        """Check if this is the same base domain with a different TLD."""
        if '.' not in domain or '.' not in target:
            return False

        domain_parts = domain.rsplit('.', 1)
        target_parts = target.rsplit('.', 1)

        # Same base, different TLD
        if domain_parts[0] == target_parts[0] and domain_parts[1] != target_parts[1]:
            return True

        return False

    def get_typosquat_score(self, domain: str) -> float:
        """
        Return a typosquatting risk score 0.0 (safe) to 1.0 (definitely typosquat).

        This is the main feature for ML integration.
        """
        result = self.check_domain(domain)

        if not result['is_typosquat']:
            # Even if not flagged, return the raw similarity as a signal
            # But dampen it - only high similarities should contribute
            if result['similarity'] > 0.7 and result['distance'] <= 3:
                return result['similarity'] * 0.5  # Partial signal
            return 0.0

        # It IS a typosquat - score by attack type
        if result['attack_type'] == 'homoglyph':
            return 0.95
        elif result['attack_type'] == 'single_char_edit':
            return 0.90
        elif result['attack_type'] == 'tld_swap':
            return 0.85
        elif result['attack_type'] == 'high_similarity':
            return 0.80
        elif result['attack_type'] == 'double_char_edit':
            return 0.75

        return result['similarity']

    def get_shannon_entropy(self, text: str) -> float:
        """
        Calculate Shannon entropy of a string.

        High entropy suggests random/automated generation.
        Typical human-chosen local parts: 2.5-4.0
        Random generated strings: 4.0-5.0+
        """
        if not text:
            return 0.0

        length = len(text)
        freq = {}
        for char in text:
            freq[char] = freq.get(char, 0) + 1

        entropy = 0.0
        for count in freq.values():
            p = count / length
            if p > 0:
                entropy -= p * math.log2(p)

        return entropy
