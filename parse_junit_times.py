import sys
import xml.etree.ElementTree as ET


def main():
    if len(sys.argv) != 2:
        print("Usage: python parse_junit_times.py <file.xml>")
        raise SystemExit(2)
    xml_file = sys.argv[1]
    root = ET.parse(xml_file).getroot()
    rows = []
    for tc in root.iter("testcase"):
        try:
            t = float(tc.attrib.get("time", "0") or 0.0)
        except ValueError:
            t = 0.0
        name = f"{tc.attrib.get('classname','')}::{tc.attrib.get('name','')}"
        rows.append((t, name))
    for t, name in sorted(rows, reverse=True):
        print(f"{t:.6f}s {name}")


if __name__ == "__main__":
    main()
