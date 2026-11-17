"""Downloads real background noise from DEMAND (Thiemann, Ito & Vincent 2013; CC BY 4.0; https://zenodo.org/records/1227121).

    python batch/fetch_noise.py            # ~610 MB download, keeps ~60 MB of 16 kHz mono WAV in batch/noise/

Each DEMAND zip holds 16 microphone channels of one environment; only channel 1 is kept. Use them with
--augment bg-real-kitchen, bg-real-sink, bg-real-living, bg-real-cafe, bg-real-traffic, bg-real-office.
DEMAND has no laundry-room recording; "sink" (DWASHING) is washing dishes.
"""
import tempfile
import zipfile
from pathlib import Path

import httpx

URL = "https://zenodo.org/records/1227121/files/{code}_16k.zip"
REAL = {"real-kitchen": "DKITCHEN", "real-sink": "DWASHING", "real-living": "DLIVING",
        "real-cafe": "PCAFETER", "real-traffic": "STRAFFIC", "real-office": "OOFFICE"}
DIR = Path(__file__).resolve().parent / "noise"


def main() -> None:
    DIR.mkdir(exist_ok=True)
    for name, code in REAL.items():
        dest = DIR / f"{name}.wav"
        if dest.exists():
            print(f"have {name}")
            continue
        print(f"downloading {code} -> {name} ...", flush=True)
        with tempfile.TemporaryDirectory() as tmp:
            zpath = Path(tmp) / "env.zip"
            with httpx.stream("GET", URL.format(code=code), follow_redirects=True, timeout=120) as r, open(zpath, "wb") as f:
                r.raise_for_status()
                for chunk in r.iter_bytes(1 << 20):
                    f.write(chunk)
            with zipfile.ZipFile(zpath) as z:
                member = next(n for n in z.namelist() if n.endswith("ch01.wav"))  # one named member only
                dest.write_bytes(z.read(member))
        print(f"  saved {dest.name} ({dest.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
