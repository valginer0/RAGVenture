import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def main():
    # Choose XML file: prefer arg, else tests_durations_full.xml, else tests_durations_junit.xml
    if len(sys.argv) > 1:
        xml_path = Path(sys.argv[1])
    else:
        xml_path = Path("tests_durations_full.xml")
        if not xml_path.exists():
            xml_path = Path("tests_durations_junit.xml")
    if not xml_path.exists():
        raise SystemExit(f"Missing JUnit XML: {xml_path}")

    root = ET.parse(xml_path).getroot()
    items = []
    for tc in root.iter("testcase"):
        try:
            t = float(tc.attrib.get("time", "0") or 0.0)
        except ValueError:
            t = 0.0
        cls = tc.attrib.get("classname", "")
        name = tc.attrib.get("name", "")
        items.append((t, f"{cls}::{name}"))

    items.sort(reverse=True)
    out_path = Path("tests_durations_top50_precise.txt")
    with out_path.open("w", encoding="utf-8") as f:
        for t, n in items[:50]:
            f.write(f"{t:.6f}s {n}\n")

    print(f"Wrote top durations to {out_path}")


if __name__ == "__main__":
    main()
