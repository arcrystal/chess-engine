"""Chess-engine package. Adds its own directory to ``sys.path`` so that the
module files can use bare imports (``from constants import ...``) whether
they're invoked as ``python -m src.uci`` or as ``import position`` after
``sys.path.insert(0, 'src')``.
"""

import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)
