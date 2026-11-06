"""The README names every module and every doc, so a new one can't be left out of it unnoticed.

Added after the README missed the server, the HF export and the vLLM / llama.cpp comparison
(PLAN.md: a step is done only when the README shows what it changed for a reader).
"""

from pathlib import Path

ROOT = Path(__file__).parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")


def _layout() -> str:
    start = README.index("## Repository layout")
    return README[start:README.index("\n## ", start + 1)]


def test_layout_names_every_module():
    pkg = ROOT / "src/tinyllm"
    names = [p.name for p in pkg.glob("*.py") if p.name != "__init__.py"]
    names += [f"{p.name}/" for p in pkg.iterdir() if (p / "__init__.py").exists()]
    missing = [n for n in sorted(names) if f"`{n}`" not in _layout() and f"`src/tinyllm/{n}`" not in _layout()]
    assert not missing, f"add to the README's Repository layout: {missing}"


def test_readme_links_every_doc():
    missing = [p.name for p in sorted((ROOT / "docs").glob("*.md")) if f"(docs/{p.name})" not in README]
    assert not missing, f"link from the README (the docs/ row of Repository layout): {missing}"


def test_readme_images_exist():
    import re
    missing = [m for m in re.findall(r"\]\((docs/img/[^)]+)\)", README) if not (ROOT / m).exists()]
    assert not missing, missing
