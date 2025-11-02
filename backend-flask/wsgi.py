import sys
import os

# Add the project directory to the sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Activate the virtual environment
activate_this = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.venv', 'bin', 'activate_this.py')
exec(open(activate_this).read(), dict(__file__=activate_this))

from app import app as application