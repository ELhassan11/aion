"""Generate a small test DOCX (stdlib only) for trying the DOCX extractor.

Usage:
    python scripts/make_test_docx.py --output /tmp/test.docx
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, tostring

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def _p(body: Element, text: str) -> None:
    p = SubElement(body, f"{{{W}}}p")
    r = SubElement(p, f"{{{W}}}r")
    t = SubElement(r, f"{{{W}}}t")
    t.text = text


def _table(body: Element, rows: list[list[str]]) -> None:
    tbl = SubElement(body, f"{{{W}}}tbl")
    for row in rows:
        tr = SubElement(tbl, f"{{{W}}}tr")
        for cell in row:
            tc = SubElement(tr, f"{{{W}}}tc")
            p = SubElement(tc, f"{{{W}}}p")
            r = SubElement(p, f"{{{W}}}r")
            t = SubElement(r, f"{{{W}}}t")
            t.text = cell


def make_docx(output: Path) -> None:
    document = Element(f"{{{W}}}document")
    body = SubElement(document, f"{{{W}}}body")
    _p(body, "قائمة أسعار متجر النيل - تجريبي")
    _p(body, "هذه الفقرة للتحقق من استخراج النصوص العربية من ملفات Word.")
    _table(
        body,
        [
            ["المنتج", "السعر", "الكمية"],
            ["هاتف ذكي", "8500 جنيه", "12"],
            ["سماعات", "1200 جنيه", "30"],
        ],
    )
    _p(body, "نهاية المستند التجريبي.")

    document_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        + tostring(document, encoding="unicode")
    )
    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>"""
    rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""

    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types)
        z.writestr("_rels/.rels", rels)
        z.writestr("word/document.xml", document_xml)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a test DOCX file")
    ap.add_argument("--output", required=True, help="output .docx path")
    args = ap.parse_args()
    make_docx(Path(args.output))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
