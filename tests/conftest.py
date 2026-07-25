import os
import sys

CURRENT_DIR = os.path.dirname(__file__)
SERVICE_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
SOURCE_ROOT = os.path.join(SERVICE_ROOT, "src")
for path in (SERVICE_ROOT, SOURCE_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)
