"""Compatibility entrypoint for Streamlit Cloud apps configured to streamlit.py.

The real multipage app entrypoint is Home.py. Some Streamlit Cloud deployments
were created with `streamlit.py` as the main file path and do not expose a way
to edit that path, so this shim delegates to Home.py.
"""

from pathlib import Path
import runpy


runpy.run_path(str(Path(__file__).with_name("Home.py")), run_name="__main__")
