from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parent
    app_path = Path(__file__).with_name("app.py")
    venv_streamlit = project_root / ".venv" / "bin" / "streamlit"
    streamlit_exe = (
        venv_streamlit
        if venv_streamlit.exists()
        else Path(sys.executable).with_name("streamlit")
    )
    if not streamlit_exe.exists():
        sys.exit(
            "Streamlit is not installed in this Python environment. "
            "Run: .venv/bin/pip install -r requirements.txt"
        )

    os.execv(str(streamlit_exe), [str(streamlit_exe), "run", str(app_path)])


if __name__ == "__main__":
    main()
