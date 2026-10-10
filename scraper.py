"""Usage: python scraper.py 70 72 73  -> saves CSV files into the data/ folder.
Timeout-tolerant: automatic retries, progress saved per page, resumable by re-running the same command."""
import os, sys, time
import requests, pandas as pd
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
from tqdm import tqdm

BASE = "https://graduation.apps.binus.ac.id/edition/wisuda-"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36"}
FILTER = {"SearchGraduate": "", "GraduatesDegree": 3, "GraduatesFaculty": 6}  # S1, School of Computer Science

session = requests.Session()
session.headers.update(HEADERS)
session.mount("https://", HTTPAdapter(max_retries=Retry(
    total=5, connect=5, read=5, backoff_factor=2,  # waits 2, 4, 8, 16 seconds...
    status_forcelist=[429, 500, 502, 503, 504])))


def soup_of(url, params=None):
    r = session.get(url, params=params, timeout=(10, 30))
    r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser")


def thesis_title(url):
    try:
        meta = soup_of(url).select("div.graduates-meta")
        return meta[-1].select_one(".table-cell .the-value").get_text(strip=True)
    except Exception:
        return None  # rows without a title are dropped later in build_data.py


def parse_list(ul):
    rows = []
    for li in tqdm(ul.select("li.the-student"), leave=False):
        try:
            gpa = li.select_one(".gpa-row .col-xs-6 .value").get_text(strip=True)
        except Exception:
            gpa = None
        rows.append({
            "Name": li.select_one("span.student-name").get_text(strip=True),
            "GPA": gpa,
            "Program": li.select_one("span.program-name").get_text(strip=True),
            "Thesis Title": thesis_title(li.select_one("a.photo")["href"]),
        })
        time.sleep(0.5)  # be polite to the server; increase if timeouts are frequent
    return rows


def scrape(url, out):
    if os.path.exists(out):
        print(f"  {out} already exists, skipping")
        return
    part = out + ".part"
    rows = pd.read_csv(part).to_dict("records") if os.path.exists(part) else []
    page = max((r["page"] for r in rows), default=0) + 1
    if rows:
        print(f"  resuming from page {page} ({len(rows)} rows already saved)")

    while True:
        print(f"  page {page}")
        try:
            ul = soup_of(url, {**FILTER, "page": page}).find("ul", id="the-graduates")
            if not ul:
                break
            new = parse_list(ul)
        except requests.RequestException as e:
            print(f"  FAILED on page {page}: {e.__class__.__name__}")
            print("  Progress saved. Re-run the same command to resume.")
            return
        if not new:
            break
        rows += [{**r, "page": page} for r in new]
        pd.DataFrame(rows).to_csv(part, index=False)  # checkpoint after every page
        page += 1

    pd.DataFrame(rows).drop(columns="page", errors="ignore").to_csv(out, index=False)
    if os.path.exists(part):
        os.remove(part)
    print(f"  saved {len(rows)} rows -> {out}")


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    for ed in sys.argv[1:] or [input("Graduation edition number: ")]:
        for kind, path in [("Graduates", "graduates"), ("Outstanding-Graduates", "outstanding-graduates")]:
            print(f"Edition {ed} - {kind}")
            scrape(f"{BASE}{ed}/{path}", f"data/{ed}-{kind}.csv")
