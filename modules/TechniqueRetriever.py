import os
import pickle
import logging
import threading
import numpy as np

# Prefer PyTorch-only transformer stack (avoid TensorFlow import side-effects)
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# Check for PyTorch availability
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Library modules should not configure the root logger — main.py owns that
logger = logging.getLogger(__name__)

# Check for sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not available. Advanced MITRE mapping will be limited.")

# Check for FAISS
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("faiss not available. Will use fallback similarity search.")

# Check for STIX2
try:
    from stix2 import Filter
    STIX2_AVAILABLE = True
except ImportError:
    STIX2_AVAILABLE = False
    logger.warning("stix2 not available. MITRE data handling will be limited.")

# Import module dependencies
from .ApplicationConfig import ApplicationConfig

class TechniqueRetriever:
    """Retrieves techniques from the MITRE ATT&CK collection and prepares embeddings."""

    def __init__(self, memory_store, config: ApplicationConfig):
        self.memory_store = memory_store
        self.config = config
        self.embedding_cache_file = config.mitre_cache_dir / "embeddings_cache.pkl"
        self._index_lock = threading.Lock()

        if TORCH_AVAILABLE:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = "cpu"

        logger.info(f"Using device for embeddings: {self.device}")

        # Initialize sentence transformer if available
        if SENTENCE_TRANSFORMERS_AVAILABLE and config.enable_semantic_mitre:
            try:
                self.model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
                logger.info("Loaded sentence transformer model for semantic search")
            except Exception as e:
                logger.warning(f"Could not load sentence transformer: {e}")
                self.model = None
        else:
            self.model = None

        self.techniques = self.get_all_techniques()

        if self.techniques and self.model:
            self.embeddings = self.load_or_compute_embeddings()
            self.index = self.build_faiss_index()
        else:
            logger.info("Using rule-based MITRE mapping (semantic search disabled)")
            self.embeddings = None
            self.index = None

    def get_all_techniques(self):
        """Retrieves all techniques from the ATT&CK collection."""
        try:
            logger.info("Retrieving all techniques from MITRE ATT&CK collection.")

            if STIX2_AVAILABLE and hasattr(self.memory_store, 'query'):
                # Use STIX2 query
                techniques = self.memory_store.query([Filter("type", "=", "attack-pattern")])
                # Filter out deprecated techniques
                active_techniques = [t for t in techniques if not t.get('x_mitre_deprecated', False) and not t.get('revoked', False)]
            elif isinstance(self.memory_store, list):
                # Raw JSON data
                active_techniques = [
                    t for t in self.memory_store
                    if t.get('type') == 'attack-pattern' and not t.get('x_mitre_deprecated', False) and not t.get('revoked', False)
                ]
            elif isinstance(self.memory_store, dict):
                # Built-in techniques dict — should never reach TechniqueRetriever
                # (MitreAttackFramework routes dict stores to load_builtin_framework)
                logger.warning("TechniqueRetriever received built-in dict store — returning empty list")
                active_techniques = []
            else:
                active_techniques = []

            logger.info(f"Successfully retrieved {len(active_techniques)} active techniques.")
            return active_techniques

        except Exception as e:
            logger.error(f"Error retrieving techniques: {e}")
            return []

    def load_or_compute_embeddings(self):
        """Loads precomputed embeddings or computes them if cache is unavailable."""
        if not self.techniques or not self.model:
            return None

        # Check if cache exists and is valid
        if self.embedding_cache_file.exists():
            try:
                logger.info("Loading embeddings from cache.")
                with open(self.embedding_cache_file, 'rb') as f:
                    cached_data = pickle.load(f)

                # Verify cache validity: count match AND unit-vector normalization
                if len(cached_data) == len(self.techniques):
                    sample_norms = np.linalg.norm(cached_data[:min(10, len(cached_data))], axis=1)
                    if np.allclose(sample_norms, 1.0, atol=1e-4):
                        return cached_data
                    else:
                        logger.warning("Cached embeddings are not L2-normalized. Recomputing.")
                else:
                    logger.warning("Cache size mismatch. Recomputing embeddings.")
            except Exception as e:
                logger.warning(f"Could not load embedding cache: {e}")

        logger.info("Computing embeddings for technique descriptions.")
        descriptions = []

        for i, technique in enumerate(self.techniques):
            # Combine name and description for better matching
            name = technique.get('name', '').strip()
            desc = technique.get('description', '').strip()
            if not name and not desc:
                # Fallback to technique ID for empty entries
                ext_refs = technique.get('external_references', [])
                tech_id = next(
                    (ref.get('external_id', '') for ref in ext_refs
                     if ref.get('source_name') == 'mitre-attack'),
                    f"Unknown technique {i}"
                )
                combined = f"MITRE ATT&CK technique {tech_id}"
            elif not name:
                combined = desc
            elif not desc:
                combined = name
            else:
                combined = f"{name}. {desc}"
            descriptions.append(combined)

        # Compute embeddings in batches
        batch_size = 32 if self.device == "cuda" else 16
        embeddings = self.model.encode(
            descriptions,
            batch_size=batch_size,
            convert_to_tensor=True,
            device=self.device,
            show_progress_bar=False  # Disable progress bar for GUI
        )

        if TORCH_AVAILABLE:
            embeddings = embeddings.cpu().numpy().astype(np.float32)
        else:
            embeddings = np.array(embeddings).astype(np.float32)

        # L2-normalize so IndexFlatL2 distances map to cosine similarity:
        # cosine_sim = 1.0 - squared_L2_distance / 2.0  (valid only for unit vectors)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # avoid division by zero
        embeddings = embeddings / norms

        # Save to cache atomically (write tmp then rename to prevent corruption)
        try:
            import tempfile
            tmp_fd, tmp_path = tempfile.mkstemp(
                dir=str(self.embedding_cache_file.parent), suffix='.tmp'
            )
            try:
                with os.fdopen(tmp_fd, 'wb') as f:
                    pickle.dump(embeddings, f)
                # Atomic rename (on Windows, target must not exist)
                if self.embedding_cache_file.exists():
                    self.embedding_cache_file.unlink()
                os.rename(tmp_path, str(self.embedding_cache_file))
                logger.info("Embeddings computed and cached.")
            except Exception:
                # Clean up temp file on failure
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except Exception as e:
            logger.warning(f"Could not save embedding cache: {e}")

        return embeddings

    def build_faiss_index(self):
        """Builds a Faiss index for fast similarity search."""
        if self.embeddings is None or len(self.embeddings) == 0:
            return None

        logger.info("Building Faiss index for similarity search.")
        dimension = self.embeddings.shape[1]

        if FAISS_AVAILABLE:
            # Use Faiss for fast similarity search
            if len(self.embeddings) > 1000:
                # For larger datasets, use IVF index
                nlist = max(2, min(int(np.sqrt(len(self.embeddings))), len(self.embeddings) // 39))
                try:
                    # Keep quantizer as a named variable — FAISS holds a raw C++ pointer
                    # to it; if Python GC's the temporary, use-after-free / segfault occurs
                    quantizer = faiss.IndexFlatL2(dimension)
                    index = faiss.IndexIVFFlat(
                        quantizer,
                        dimension,
                        nlist
                    )
                    index.train(self.embeddings)
                    index.add(self.embeddings)
                    index.nprobe = min(5, nlist)  # Probe multiple clusters for accuracy
                    self._quantizer = quantizer  # prevent GC for lifetime of index
                except Exception:
                    logger.warning("FAISS IVF training failed, falling back to flat index")
                    index = faiss.IndexFlatL2(dimension)
                    index.add(self.embeddings)
            else:
                # For smaller datasets, use simple flat index
                index = faiss.IndexFlatL2(dimension)
                index.add(self.embeddings)
            logger.info(f"Faiss index built successfully with {index.ntotal} vectors.")
        else:
            # Fallback to numpy-based search
            logger.info("Using numpy-based similarity search (Faiss not available)")
            index = self.embeddings  # Store embeddings directly for numpy search

        return index

