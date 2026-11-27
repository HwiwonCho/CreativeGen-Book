"""Collect real prose pages and match their OCR lines to the archive transcription."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import time
import xml.etree.ElementTree as ET
from urllib.request import Request, urlopen

from PIL import Image


BOOKS = ("strangestoriesfr00pusuuoft", "adventuresoftoms00twaiiala", "newforestitshist00wiseuoft",
         "handbookoffieldg00coue", "birdsofnorfolkwi01stev", "musicofwaterscol00smituoft",
         "historyofbritish1828flem", "buddhistbirth00daviuoft",
         "adventuresofsher00doylrich", "goldsmithshistor00golduoft", "historyofcoggesh00beauuoft")
LEAVES = (38, 49, 64, 77, 92, 109)
# Both OCR sources misread the year on this line in Fleming's volume.
REJECTED_LINES = {("historyofbritish1828flem/n49.jpg", (374,1347,1831,1391))}


def fetch(url, path):
    if path.exists():
        return path.read_bytes()
    for attempt in range(3):
        try:
            with urlopen(Request(url,headers={"User-Agent":"KAIST-course-student-book-research/1.0"}),timeout=35) as response:
                data = response.read()
            tmp = path.with_suffix(path.suffix+".part")
            tmp.write_bytes(data)
            tmp.replace(path)
            return data
        except OSError:
            if attempt == 2:
                raise
            time.sleep(1+attempt)


def sources(project):
    rows = {}
    for name in ("book_spreads.json","book_print_extra.json"):
        for row in json.loads((project/"data/sources"/name).read_text(encoding="utf-8"))["spreads"]:
            rows[row["id"]] = row
    return [rows[k] for k in BOOKS]


def download(folder, books):
    tasks = []
    errors = []
    def download_book(book):
        name = book["id"]
        local = folder/name
        local.mkdir(exist_ok=True)
        rights = json.loads(fetch(book["rights_url"],local/"rights.json"))["result"]
        if rights != "NOT_IN_COPYRIGHT" or book["rights_statement"] != rights:
            raise ValueError(f"Unverified rights for {name}: {rights!r}")
        url = f"https://archive.org/download/{name}/{name}_djvu.txt"
        fetch(url,local/"transcript.txt")
        return [(f"https://archive.org/download/{name}/page/n{leaf}_w1400.jpg",local/f"n{leaf}.jpg") for leaf in LEAVES]
    def page(job):
        url,path = job
        raw = fetch(url,path)
        with Image.open(io.BytesIO(raw)) as image:
            image.verify()
        return path
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {b["id"]:pool.submit(download_book,b) for b in books}
        for name,f in futures.items():
            try:
                tasks.extend(f.result())
            except Exception as exc:
                errors.append((name,str(exc)))
        futures = [(path,pool.submit(page,(url,path))) for url,path in tasks]
        for path,f in futures:
            try:
                f.result()
            except Exception as exc:
                errors.append((str(path),str(exc)))
    (folder/"download-errors.json").write_text(json.dumps(errors,indent=2))
    print(f"{len(tasks)} pages scheduled, {len(errors)} errors",flush=True)
    if errors:
        raise RuntimeError(f"Some sources could not be downloaded; see {folder/'download-errors.json'}")


def normalize(s):
    return " ".join(s.translate(str.maketrans({"’":"'","‘":"'","“":'"',"”":'"',"—":"-","–":"-","ﬁ":"fi","ﬂ":"fl"})).split())


def read_page(path):
    hocr = path.with_suffix(".hocr")
    if not hocr.exists():
        res = subprocess.run(["tesseract",str(path),"stdout","-l","eng","--psm","3","-c","hocr_char_boxes=1","hocr"],
                             capture_output=True,check=True,timeout=120)
        hocr.write_bytes(res.stdout)
    root = ET.fromstring(hocr.read_bytes())
    def box(node, name):
        return list(map(int,re.search(r"\b"+name+r" (\d+) (\d+) (\d+) (\d+)",node.get("title")).groups()))
    lines = []
    for line in root.findall(".//*[@class='ocr_line']"):
        words = []
        for word in line.findall(".//*[@class='ocrx_word']"):
            chars = [{"c":normalize("".join(ch.itertext())), "box":box(ch,"x_bboxes")}
                     for ch in word.findall(".//*[@class='ocrx_cinfo']")]
            if not chars:
                continue
            words.append(dict(text="".join(ch["c"] for ch in chars),box=box(word,"bbox"),chars=chars,
                              conf=int(re.search(r"x_wconf (\d+)",word.get("title")).group(1))))
        if words:
            lines.append(words)
    return lines


def extract(folder, books):
    records = []
    seen = set()
    for book in books:
        name = book["id"]
        local = folder/name
        rights = json.loads((local/"rights.json").read_text(encoding="utf-8"))["result"]
        if rights != "NOT_IN_COPYRIGHT" or rights != book["rights_statement"]:
            raise ValueError(f"Unverified rights for {name}")
        transcript = " "+normalize((local/"transcript.txt").read_text(encoding="utf-8-sig",errors="replace"))+" "
        split = ("test" if name=="adventuresofsher00doylrich" else
                 "validation" if name=="goldsmithshistor00golduoft" else book["split"])
        paths = sorted(local/f"n{leaf}.jpg" for leaf in LEAVES)
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(path)
        with ThreadPoolExecutor(max_workers=3) as pool:
            pages = list(pool.map(read_page, paths))
        for path, lines in zip(paths,pages):
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            for words in lines:
                text=normalize(" ".join(w["text"] for w in words))
                if (not text.isascii() or not 10<=len(text)<=90 or len(words)<3 or not re.search("[a-z]",text)
                        or " "+text+" " not in transcript or min(float(w["conf"]) for w in words)<60):
                    continue
                if text.casefold() in seen:
                    continue
                if any(len(ch["c"])!=1 for w in words for ch in w["chars"]):
                    continue
                boxes=[w["box"] for w in words]
                box=[min(b[0] for b in boxes),min(b[1] for b in boxes),max(b[2] for b in boxes),max(b[3] for b in boxes)]
                if (path.relative_to(folder).as_posix(),tuple(box)) in REJECTED_LINES:
                    continue
                if box[3]-box[1]<10 or box[2]-box[0]<45:
                    continue
                seen.add(text.casefold())
                records.append(dict(split=split,book=name,image=path.relative_to(folder).as_posix(),
                                    sha256=digest,text=text,box=box,words=words,
                                    source=f"https://archive.org/download/{name}/page/{path.stem}_w1400.jpg",
                                    transcript=f"https://archive.org/download/{name}/{name}_djvu.txt",rights=book["rights_url"]))
        print(name,split,sum(r["book"]==name for r in records),flush=True)
    if {r["split"] for r in records}!={"train","validation","test"}:
        raise ValueError("Missing a book-disjoint partition")
    tmp=folder/"lines.tmp"
    tmp.write_text("".join(json.dumps(r,ensure_ascii=True)+"\n" for r in records),encoding="utf-8")
    tmp.replace(folder/"lines.jsonl")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("download","extract"))
    p.add_argument("folder",type=Path)
    args=p.parse_args()
    args.folder.mkdir(parents=True,exist_ok=True)
    selected=sources(Path(__file__).resolve().parents[1])
    (download if args.action=="download" else extract)(args.folder,selected)
