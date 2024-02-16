import sys

if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
    print("Running in a virtual environment")
else:
    print("Not running in a virtual environment")
