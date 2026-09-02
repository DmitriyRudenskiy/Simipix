"""Позволяет запускать пакет напрямую: python -m imgsim ..."""

import sys

from .cli import main

sys.exit(main())
