from __future__ import annotations

from getpass import getpass
from pathlib import Path


ENV_PATH = Path(__file__).resolve().parents[1] / ".env"


def _looks_like_placeholder(key: str) -> bool:
    cleaned = key.strip().lower()
    return (
        not cleaned
        or cleaned.startswith("sk-your")
        or "your-key" in cleaned
        or "your_api_key" in cleaned
        or "****" in cleaned
    )


def _set_env_value(lines: list[str], name: str, value: str) -> list[str]:
    new_line = f"{name}={value}"
    replaced = False
    updated: list[str] = []

    for line in lines:
        if line.startswith(f"{name}="):
            updated.append(new_line)
            replaced = True
        else:
            updated.append(line)

    if not replaced:
        if updated and updated[-1].strip():
            updated.append("")
        updated.append(new_line)
    return updated


def main() -> None:
    print("Paste your real OpenAI API key. Input is hidden and will not be printed.")
    key = getpass("OPENAI_API_KEY: ").strip()

    if _looks_like_placeholder(key) or not key.startswith("sk-"):
        raise SystemExit("That does not look like a real OpenAI API key. .env was not changed.")

    lines = ENV_PATH.read_text().splitlines() if ENV_PATH.exists() else []
    updated = _set_env_value(lines, "OPENAI_API_KEY", key)
    ENV_PATH.write_text("\n".join(updated) + "\n")

    print(f"Saved OPENAI_API_KEY to {ENV_PATH}")
    print(".env is ignored by Git, so this will not be pushed to GitHub.")


if __name__ == "__main__":
    main()
