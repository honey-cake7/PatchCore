import os

# faiss-cpu and torch both ship an OpenMP runtime; on macOS the duplicate-runtime
# guard aborts the process. These must be set before faiss/torch are imported by
# any test module. Harmless on Linux/cluster.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
