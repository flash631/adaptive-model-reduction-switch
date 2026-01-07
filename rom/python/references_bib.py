from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path


_ARXIV_RE = re.compile(r"\barXiv:([0-9]{4}\.[0-9]{4,5})(v\d+)?\b", flags=re.I)
_ARXIV_IN_NAME_RE = re.compile(r"\barXiv([0-9]{4}\.[0-9]{4,5})\b", flags=re.I)


def _pdftotext_first_page(pdf: Path, max_lines: int = 250) -> str:
    p = subprocess.run(
        ["pdftotext", "-f", "1", "-l", "1", str(pdf), "-"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if p.returncode != 0:
        return ""
    return "\n".join(p.stdout.splitlines()[:max_lines])


def _arxiv_id(text: str) -> str:
    m = _ARXIV_RE.search(text)
    return m.group(1) if m else ""


def _entry_from_prefix(prefix: str) -> dict[str, str]:
    # Hardcoded titles/keys to ensure correct citations in the report.
    # Authors are kept as surnames; year is from the reference list.
    entries: dict[str, dict[str, str]] = {
        "01": {
            "key": "Hijazi2019",
            "title": "Data-driven POD-Galerkin reduced order model for turbulent flows",
            "author": "Hijazi and Stabile and Mola and Rozza",
            "year": "2019",
            "type": "misc",
        },
        "02": {
            "key": "Zancanaro2024",
            "title": "A segregated reduced-order model of a pressure-based solver for turbulent compressible flows",
            "author": "Zancanaro and Ngan and Stabile and Rozza",
            "year": "2024",
            "type": "misc",
        },
        "03": {
            "key": "Star2021",
            "title": "Reduced order models for the incompressible Navier-Stokes equations on collocated grids using a discretize-then-project approach",
            "author": "Star and Sanderse and Stabile and Rozza and Degroote",
            "year": "2021",
            "type": "misc",
        },
        "04": {
            "key": "Rozza2018",
            "title": "Advances in reduced order methods for parametric industrial problems in computational fluid dynamics",
            "author": "Rozza and Malik and Demo and Tezzele and Girfoglio",
            "year": "2018",
            "type": "misc",
        },
        "05": {
            "key": "Benner2020",
            "title": "Operator inference and physics-informed learning of low-dimensional models for incompressible flows",
            "author": "Benner and Goyal and Heiland and Pontes Duff",
            "year": "2020",
            "type": "misc",
        },
        "06": {
            "key": "Kramer2024",
            "title": "Learning nonlinear reduced models from data with operator inference",
            "author": "Kramer and Peherstorfer and Willcox",
            "year": "2024",
            "type": "article",
            "journal": "Annual Review of Fluid Mechanics",
        },
        "08": {
            "key": "Peherstorfer2016",
            "title": "Online adaptive model reduction for nonlinear systems via low-rank updates",
            "author": "Peherstorfer and Willcox",
            "year": "2016",
            "type": "misc",
        },
        "09": {
            "key": "Conti2024",
            "title": "Multi-fidelity reduced-order surrogate modelling",
            "author": "Conti and Guo and Manzoni and Frangi and Brunton and Kutz",
            "year": "2024",
            "type": "misc",
        },
        "10": {
            "key": "Stabile2017",
            "title": "POD-Galerkin reduced order methods for CFD using finite volume discretisation: vortex shedding around a circular cylinder",
            "author": "Stabile and Hijazi and Mola and Lorenzi and Rozza",
            "year": "2017",
            "type": "misc",
        },
        "11": {
            "key": "SanchezOrtiz2022",
            "title": "Adaptive reduced order model of aeroelastic systems using proper orthogonal decomposition",
            "author": "Sanchez-Ortiz and Quero and Moreno-Ramos",
            "year": "2022",
            "type": "misc",
        },
    }
    if prefix not in entries:
        raise KeyError(f"Unknown reference prefix: {prefix}")
    return entries[prefix]


def _bib_escape(s: str) -> str:
    # Minimal escaping for BibTeX fields.
    return s.replace("{", "\\{").replace("}", "\\}")


def _format_entry(meta: dict[str, str], arxiv: str) -> str:
    entry_type = meta.get("type", "misc")
    key = meta["key"]
    fields: list[str] = [
        f"  title = {{{_bib_escape(meta['title'])}}}",
        f"  author = {{{_bib_escape(meta['author'])}}}",
        f"  year = {{{_bib_escape(meta['year'])}}}",
    ]
    if meta.get("journal"):
        fields.append(f"  journal = {{{_bib_escape(meta['journal'])}}}")

    if arxiv:
        fields += [
            "  archivePrefix = {arXiv}",
            f"  eprint = {{{arxiv}}}",
            f"  url = {{https://arxiv.org/abs/{arxiv}}}",
        ]
    return f"@{entry_type}{{{key},\n" + ",\n".join(fields) + "\n}\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", default="references")
    ap.add_argument("--out", default="report/report.bib")
    ap.add_argument("--strict", action="store_true", help="Fail if PDFs are missing")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    refs_dir = (root / args.refs).resolve()
    out_path = (root / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(refs_dir.glob("*.pdf"))
    if not pdfs and args.strict:
        raise RuntimeError(f"No PDFs found in {refs_dir}")

    # Map found PDFs by prefix (e.g., "01", "02", ...).
    pdf_by_prefix: dict[str, Path] = {}
    for pdf in pdfs:
        prefix = pdf.name.split("_", 1)[0]
        pdf_by_prefix[prefix] = pdf

    out_entries: list[str] = []
    for prefix in ["01", "02", "03", "04", "05", "06", "08", "09", "10", "11"]:
        meta = _entry_from_prefix(prefix)
        arxiv = ""
        pdf = pdf_by_prefix.get(prefix)
        if pdf is not None:
            text = _pdftotext_first_page(pdf)
            arxiv = _arxiv_id(text)
            if not arxiv:
                m = _ARXIV_IN_NAME_RE.search(pdf.name)
                arxiv = m.group(1) if m else ""
        out_entries.append(_format_entry(meta, arxiv=arxiv))

    out_path.write_text("\n".join(out_entries), encoding="utf-8")
    print(f"[references_bib] Wrote: {out_path} ({len(out_entries)} entries)")


if __name__ == "__main__":
    main()
