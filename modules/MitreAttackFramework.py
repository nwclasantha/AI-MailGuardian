import os
import logging
from typing import Dict, List

import numpy as np

# Prefer PyTorch-only transformer stack (avoid TensorFlow import side-effects)
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# Check for PyTorch availability (for semantic search)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Check for sentence-transformers (for semantic search)
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Check for FAISS (for similarity search)
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

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
from .MITRETAXIIConnection import MITRETAXIIConnection
from .TechniqueRetriever import TechniqueRetriever

class MitreAttackFramework:
    """Enhanced MITRE ATT&CK Framework with semantic search capabilities"""

    def __init__(self, config: ApplicationConfig):
        self.config = config
        self.techniques = {}
        self.tactics = {}
        self.connection = None
        self.retriever = None
        self.semantic_enabled = False

        # Initialize MITRE data
        self.initialize_framework()

    def initialize_framework(self):
        """Initialize MITRE framework with advanced capabilities if available"""
        try:
            # Try to load advanced MITRE data
            self.connection = MITRETAXIIConnection(self.config)

            if self.connection.memory_store:
                # If the connection is already in built-in fallback mode, use our built-in framework
                if isinstance(self.connection.memory_store, dict):
                    logger.info("Using built-in MITRE framework (offline fallback)")
                    self.load_builtin_framework()
                    self.semantic_enabled = False
                    self.retriever = None

                else:
                    self.retriever = TechniqueRetriever(self.connection.memory_store, self.config)

                    if self.retriever.techniques:
                        # Convert to dictionary format for compatibility
                        self.load_techniques_from_retriever()

                        # If conversion failed (e.g., missing external refs), fall back to built-in techniques
                        if not self.techniques:
                            logger.warning("No MITRE techniques mapped from data source; using built-in framework")
                            self.load_builtin_framework()
                            self.semantic_enabled = False
                            self.retriever = None

                        else:
                            if self.retriever.model and self.retriever.index:
                                self.semantic_enabled = True
                                logger.info("MITRE ATT&CK framework initialized with semantic search")
                            else:
                                logger.info("MITRE ATT&CK framework initialized with rule-based matching")
                    else:
                        logger.warning("No techniques loaded, using built-in data")
                        self.load_builtin_framework()
                        self.retriever = None
            else:
                logger.warning("No MITRE data available, using built-in framework")
                self.load_builtin_framework()

        except Exception as e:
            logger.error(f"Error initializing MITRE framework: {e}")
            self.load_builtin_framework()

        # Always load tactics
        self.load_tactics()

    def load_techniques_from_retriever(self):
        """Convert retriever techniques to framework format"""
        for technique in self.retriever.techniques:
            # Extract MITRE ID
            external_refs = technique.get('external_references', [])
            mitre_id = None
            for ref in external_refs:
                if ref.get('source_name') == 'mitre-attack':
                    mitre_id = ref.get('external_id')
                    break

            if mitre_id:
                # Extract tactic from kill chain phases
                tactic = 'Unknown'
                kill_chain_phases = technique.get('kill_chain_phases', [])
                if kill_chain_phases:
                    tactic = kill_chain_phases[0].get('phase_name', 'unknown').replace('-', ' ').title()

                # Determine severity based on tactic
                severity_map = {
                    'Initial Access': 'high',
                    'Execution': 'high',
                    'Persistence': 'high',
                    'Privilege Escalation': 'high',
                    'Defense Evasion': 'medium',
                    'Credential Access': 'high',
                    'Discovery': 'low',
                    'Lateral Movement': 'high',
                    'Collection': 'medium',
                    'Command And Control': 'medium',
                    'Exfiltration': 'high',
                    'Impact': 'critical'
                }
                severity = severity_map.get(tactic, 'medium')

                self.techniques[mitre_id] = {
                    'name': technique.get('name', 'Unknown'),
                    'tactic': tactic,
                    'severity': severity,
                    'description': technique.get('description', '')
                }

    def load_builtin_framework(self):
        """Load built-in MITRE ATT&CK data as fallback"""
        self.techniques = {
            # Initial Access
            'T1566': {'name': 'Phishing', 'tactic': 'Initial Access', 'severity': 'high',
                      'description': 'Adversaries send phishing messages to gain access'},
            'T1566.001': {'name': 'Spearphishing Attachment', 'tactic': 'Initial Access', 'severity': 'high',
                          'description': 'Phishing with malicious attachments'},
            'T1566.002': {'name': 'Spearphishing Link', 'tactic': 'Initial Access', 'severity': 'high',
                          'description': 'Phishing with malicious links'},
            'T1566.003': {'name': 'Spearphishing via Service', 'tactic': 'Initial Access', 'severity': 'high',
                          'description': 'Phishing through third-party services'},
            'T1598': {'name': 'Phishing for Information', 'tactic': 'Reconnaissance', 'severity': 'medium',
                      'description': 'Phishing to collect information'},
            'T1190': {'name': 'Exploit Public-Facing Application', 'tactic': 'Initial Access', 'severity': 'critical',
                      'description': 'Exploiting internet-facing applications'},
            'T1133': {'name': 'External Remote Services', 'tactic': 'Initial Access', 'severity': 'high',
                      'description': 'Using remote services for access'},

            # Execution
            'T1204': {'name': 'User Execution', 'tactic': 'Execution', 'severity': 'medium',
                      'description': 'User executes malicious content'},
            'T1204.001': {'name': 'Malicious Link', 'tactic': 'Execution', 'severity': 'high',
                          'description': 'User clicks malicious link'},
            'T1204.002': {'name': 'Malicious File', 'tactic': 'Execution', 'severity': 'high',
                          'description': 'User opens malicious file'},

            # Persistence
            'T1078': {'name': 'Valid Accounts', 'tactic': 'Persistence', 'severity': 'high',
                      'description': 'Using legitimate credentials'},
            'T1053': {'name': 'Scheduled Task/Job', 'tactic': 'Persistence', 'severity': 'medium',
                      'description': 'Scheduling malicious tasks'},
            'T1543': {'name': 'Create or Modify System Process', 'tactic': 'Persistence', 'severity': 'high',
                      'description': 'Modifying system processes'},

            # Privilege Escalation
            'T1055': {'name': 'Process Injection', 'tactic': 'Privilege Escalation', 'severity': 'high',
                      'description': 'Injecting code into processes'},
            'T1068': {'name': 'Exploitation for Privilege Escalation', 'tactic': 'Privilege Escalation',
                      'severity': 'critical', 'description': 'Exploiting vulnerabilities for privileges'},

            # Defense Evasion
            'T1027': {'name': 'Obfuscated Files or Information', 'tactic': 'Defense Evasion', 'severity': 'medium',
                      'description': 'Hiding malicious content'},
            'T1070': {'name': 'Indicator Removal', 'tactic': 'Defense Evasion', 'severity': 'medium',
                      'description': 'Removing attack indicators'},

            # Credential Access
            'T1110': {'name': 'Brute Force', 'tactic': 'Credential Access', 'severity': 'high',
                      'description': 'Password guessing attacks'},
            'T1555': {'name': 'Credentials from Password Stores', 'tactic': 'Credential Access', 'severity': 'high',
                      'description': 'Stealing stored passwords'},
            'T1539': {'name': 'Steal Web Session Cookie', 'tactic': 'Credential Access', 'severity': 'high',
                      'description': 'Stealing authentication cookies'},

            # Discovery
            'T1087': {'name': 'Account Discovery', 'tactic': 'Discovery', 'severity': 'low',
                      'description': 'Discovering user accounts'},
            'T1083': {'name': 'File and Directory Discovery', 'tactic': 'Discovery', 'severity': 'low',
                      'description': 'Exploring file systems'},

            # Collection
            'T1114': {'name': 'Email Collection', 'tactic': 'Collection', 'severity': 'high',
                      'description': 'Collecting email data'},
            'T1005': {'name': 'Data from Local System', 'tactic': 'Collection', 'severity': 'medium',
                      'description': 'Collecting local data'},

            # Command and Control
            'T1071': {'name': 'Application Layer Protocol', 'tactic': 'Command and Control', 'severity': 'medium',
                      'description': 'Using common protocols for C2'},
            'T1090': {'name': 'Proxy', 'tactic': 'Command and Control', 'severity': 'medium',
                      'description': 'Using proxy connections'},

            # Exfiltration
            'T1041': {'name': 'Exfiltration Over C2 Channel', 'tactic': 'Exfiltration', 'severity': 'high',
                      'description': 'Data theft via C2'},
            'T1567': {'name': 'Exfiltration Over Web Service', 'tactic': 'Exfiltration', 'severity': 'high',
                      'description': 'Data theft via web services'},
            'T1048': {'name': 'Exfiltration Over Alternative Protocol', 'tactic': 'Exfiltration', 'severity': 'high',
                      'description': 'Data theft via alternate protocols'},

            # Impact
            'T1486': {'name': 'Data Encrypted for Impact', 'tactic': 'Impact', 'severity': 'critical',
                      'description': 'Ransomware encryption'},
            'T1490': {'name': 'Inhibit System Recovery', 'tactic': 'Impact', 'severity': 'critical',
                      'description': 'Preventing system recovery'}
        }

    def load_tactics(self):
        """Load MITRE tactics"""
        self.tactics = {
            'reconnaissance': 'Reconnaissance',
            'resource-development': 'Resource Development',
            'initial-access': 'Initial Access',
            'execution': 'Execution',
            'persistence': 'Persistence',
            'privilege-escalation': 'Privilege Escalation',
            'defense-evasion': 'Defense Evasion',
            'credential-access': 'Credential Access',
            'discovery': 'Discovery',
            'lateral-movement': 'Lateral Movement',
            'collection': 'Collection',
            'command-and-control': 'Command and Control',
            'exfiltration': 'Exfiltration',
            'impact': 'Impact'
        }

    def find_techniques_by_description(self, description: str, top_k: int = 5) -> List[Dict]:
        """Find MITRE techniques using semantic search if available"""
        if self.semantic_enabled and self.retriever:
            try:
                logger.info(f"Searching for techniques matching: '{description[:50]}...'")

                # Encode the query
                query_embedding = self.retriever.model.encode(
                    [description],
                    convert_to_tensor=True,
                    device=self.retriever.device,
                    show_progress_bar=False
                )

                if TORCH_AVAILABLE:
                    query_embedding = query_embedding.cpu().numpy().astype(np.float32)
                else:
                    query_embedding = np.array(query_embedding).astype(np.float32)

                # Search using Faiss or numpy
                if FAISS_AVAILABLE and hasattr(self.retriever.index, 'search'):
                    distances, indices = self.retriever.index.search(query_embedding,
                                                                     min(top_k, len(self.retriever.techniques)))
                else:
                    # Numpy-based search
                    similarities = np.dot(self.retriever.index, query_embedding.T).flatten()
                    indices = np.argsort(similarities)[::-1][:top_k]
                    distances = 1 - similarities[indices]
                    indices = indices.reshape(1, -1)
                    distances = distances.reshape(1, -1)

                results = []
                for i in range(len(indices[0])):
                    if indices[0][i] < len(self.retriever.techniques):
                        technique = self.retriever.techniques[indices[0][i]]

                        # Calculate similarity score (0-100)
                        similarity = max(0, 100 - distances[0][i] * 10)

                        # Get MITRE ID
                        external_refs = technique.get('external_references', [])
                        mitre_id = next(
                            (ref['external_id'] for ref in external_refs
                             if ref.get('source_name') == 'mitre-attack'),
                            'Unknown'
                        )

                        if mitre_id in self.techniques:
                            technique_info = self.techniques[mitre_id]
                            results.append({
                                'id': mitre_id,
                                'name': technique_info['name'],
                                'tactic': technique_info['tactic'],
                                'severity': technique_info['severity'],
                                'description': technique_info['description'],
                                'similarity': similarity
                            })

                logger.info(f"Semantic search found {len(results)} matching techniques")
                return results

            except Exception as e:
                logger.error(f"Error in semantic search: {e}")
                # Fall back to rule-based matching

        # Fallback to keyword-based matching
        return self.find_techniques_by_keywords(description)

    def find_techniques_by_keywords(self, text: str) -> List[Dict]:
        """Find techniques using keyword matching (fallback)"""
        results = []
        text_lower = text.lower()

        # Define keyword mappings
        keyword_mappings = {
            'phishing': ['T1566', 'T1566.001', 'T1566.002', 'T1598'],
            'spearphishing': ['T1566.001', 'T1566.002'],
            'attachment': ['T1566.001', 'T1204.002'],
            'link': ['T1566.002', 'T1204.001'],
            'credential': ['T1078', 'T1110', 'T1555', 'T1539'],
            'password': ['T1110', 'T1555'],
            'brute force': ['T1110'],
            'malware': ['T1204', 'T1055', 'T1027'],
            'ransomware': ['T1486', 'T1490'],
            'execution': ['T1204', 'T1204.001', 'T1204.002'],
            'persistence': ['T1078', 'T1053', 'T1543'],
            'exfiltration': ['T1041', 'T1567', 'T1048'],
            'data': ['T1005', 'T1114', 'T1486'],
            'email': ['T1114', 'T1566'],
            'dns': ['T1071'],
            'command': ['T1071', 'T1090']
        }

        matched_techniques = set()

        for keyword, technique_ids in keyword_mappings.items():
            if keyword in text_lower:
                matched_techniques.update(technique_ids)

        for tech_id in matched_techniques:
            if tech_id in self.techniques:
                technique = self.techniques[tech_id]
                results.append({
                    'id': tech_id,
                    'name': technique['name'],
                    'tactic': technique['tactic'],
                    'severity': technique['severity'],
                    'description': technique['description'],
                    'similarity': 80  # Fixed similarity for keyword matches
                })

        return results

    def map_threat_to_techniques(self, threat_type: str, details: Dict = None) -> List[str]:
        """Map threats to MITRE techniques using semantic search if available"""

        # Create a detailed description for semantic search
        description = self.create_threat_description(threat_type, details)

        # Try semantic search first
        if self.semantic_enabled:
            semantic_results = self.find_techniques_by_description(description, top_k=5)
            if semantic_results:
                return [r['id'] for r in semantic_results if r['similarity'] > 70]

        # Fallback to rule-based mapping
        mapping = {
            'phishing': ['T1566', 'T1566.001', 'T1566.002', 'T1598', 'T1204'],
            'spearphishing': ['T1566.001', 'T1566.002', 'T1598'],
            'credential_theft': ['T1078', 'T1110', 'T1555', 'T1539'],
            'malware': ['T1204', 'T1055', 'T1053', 'T1027', 'T1543'],
            'data_breach': ['T1041', 'T1567', 'T1048', 'T1114'],
            'ransomware': ['T1486', 'T1490', 'T1027'],
            'exploitation': ['T1190', 'T1068'],
            'social_engineering': ['T1566', 'T1598', 'T1204.001']
        }

        techniques = mapping.get(threat_type, [])

        # Add context-specific techniques
        if details:
            if details.get('uses_attachment'):
                techniques.extend(['T1566.001', 'T1204.002'])
            if details.get('uses_link'):
                techniques.extend(['T1566.002', 'T1204.001'])
            if details.get('targets_credentials'):
                techniques.extend(['T1078', 'T1110', 'T1555'])

        return list(set(techniques))

    def create_threat_description(self, threat_type: str, details: Dict = None) -> str:
        """Create a detailed description for semantic search"""
        descriptions = {
            'phishing': "Adversary sends deceptive emails to steal credentials or deliver malware",
            'credential_theft': "Attacker attempts to steal user credentials and authentication information",
            'malware': "Malicious software is deployed to compromise systems and steal data",
            'ransomware': "Encryption malware that locks files and demands payment",
            'data_breach': "Unauthorized access and exfiltration of sensitive data",
            'exploitation': "Exploiting vulnerabilities in software or systems"
        }

        base_desc = descriptions.get(threat_type, f"Security threat involving {threat_type}")

        if details:
            if details.get('uses_attachment'):
                base_desc += " using malicious email attachments"
            if details.get('uses_link'):
                base_desc += " using malicious links"
            if details.get('targets_credentials'):
                base_desc += " targeting user credentials and passwords"

        return base_desc

    def get_technique_details(self, technique_id: str) -> Dict:
        """Get detailed information about a technique"""
        return self.techniques.get(technique_id, {})

    def get_techniques_by_tactic(self, tactic: str) -> List[str]:
        """Get all techniques for a specific tactic"""
        return [tid for tid, details in self.techniques.items()
                if details.get('tactic', '').lower() == tactic.lower()]
