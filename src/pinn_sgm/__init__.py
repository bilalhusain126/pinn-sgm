"""
Physics-Informed Neural Networks for Score-Based Generative Models in Finance

This package implements a unified framework combining:
- Fokker-Planck Equation (FPE) for probability density evolution
- Physics-Informed Neural Networks (PINNs) as PDE solvers
- Score-Based Generative Models (SGMs) for data generation

The framework enables theory-constrained score estimation in sparse-data regimes
for quantitative finance applications.

References:
    - PhD Research Document: Chapter 2 (Preliminaries), Chapter 3 (Theory-Constrained Score Estimation)
"""

import logging
from typing import Optional

# Package version
__version__ = "1.0.0"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def setup_logging(
    level: int = logging.INFO,
    format_string: Optional[str] = None,
    filename: Optional[str] = None
) -> None:
    """
    Configure package-wide logging.

    Args:
        level: Logging level (e.g., logging.DEBUG, logging.INFO)
        format_string: Custom format string for log messages
        filename: If provided, logs will be written to this file
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    logging_config = {
        'level': level,
        'format': format_string,
        'datefmt': '%Y-%m-%d %H:%M:%S'
    }

    if filename:
        logging_config['filename'] = filename
        logging_config['filemode'] = 'a'

    logging.basicConfig(**logging_config)


# Public API
from . import config
from . import equations
from . import solvers
from . import nets
from . import utils

__all__ = [
    'config',
    'equations',
    'solvers',
    'nets',
    'utils',
    'setup_logging',
]
