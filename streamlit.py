"""Compatibility entrypoint for Streamlit Cloud apps configured to streamlit.py.

When executed as a script, this delegates to Home.py. When imported as the
`streamlit` package by app modules, it loads the real installed Streamlit package
so this shim does not shadow the dependency.
"""

from pathlib import Path
import importlib.machinery
import importlib.util
import runpy
import site
import sys


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("Home.py")), run_name="__main__")
else:
    search_paths = []
    try:
        search_paths.extend(site.getsitepackages())
    except AttributeError:
        pass
    try:
        search_paths.append(site.getusersitepackages())
    except AttributeError:
        pass

    spec = None
    for package_path in search_paths:
        spec = importlib.machinery.PathFinder.find_spec("streamlit", [package_path])
        if spec and spec.origin and Path(spec.origin).resolve() != Path(__file__).resolve():
            break

    if spec is None or spec.loader is None:
        raise ImportError("Could not locate the installed streamlit package.")

    module = importlib.util.module_from_spec(spec)
    sys.modules[__name__] = module
    spec.loader.exec_module(module)
