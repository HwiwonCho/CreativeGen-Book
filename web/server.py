"""Team web interface for the book pipeline. Run with the training environment's Python."""
import argparse
import hashlib
import hmac
from http.cookies import SimpleCookie, CookieError
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
import unicodedata
from urllib.parse import urlsplit

from architecture import book, joint_book

ROOT = Path(__file__).resolve().parents[1]
PAGE = Path(__file__).with_name("index.html")
PAGE_SIZE = 768
WEEK = 7 * 24 * 60 * 60
COOKIE = "creativegen_session"
JOB_PATH = re.compile(r"/api/jobs/([a-f0-9]{32})(?:/(image|download|background))?")
FINAL_IMAGE = Path("spread/final.png")
BACKGROUND_IMAGE = Path("background/background.png")


def team_invite(key):
    return hmac.new(key, b"creativegen-book/team-invite/v1", hashlib.sha256).hexdigest()[:32]


class RequestError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def read_record(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}


class Jobs:
    def __init__(self, folder, model):
        self.folder = folder
        self.folder.mkdir(parents=True, exist_ok=True)
        self.model = model
        config = json.loads((ROOT / "configs/project.json").read_text(encoding="utf-8-sig"))
        self.font = ROOT / config["modules"]["book"]["font"]
        self.lock_file = folder / ".gpu.lock"
        self.guard = threading.Lock()

    def gpu_owner(self):
        import fcntl
        with self.lock_file.open("a+", encoding="ascii") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                lock.seek(0)
                return lock.read()
        return None

    def page_text(self, sentence):
        if not isinstance(sentence, str) or not sentence.strip():
            raise RequestError(400, "책에 넣을 영문 문장이나 짧은 문단을 입력해 주세요.")
        sentence = sentence.replace("\r\n", "\n").replace("\r", "\n").strip()
        sentence = sentence.translate(str.maketrans({"\t": " ", "\u00a0": " ", "\u202f": " ",
                                                     "\u2028": "\n", "\u2029": "\n\n"}))
        letters = [c for c in sentence if c.isalpha()]
        if not letters or any(not unicodedata.name(c, "").startswith("LATIN ") and c != "ʼ" for c in letters):
            raise RequestError(400, "현재 책에는 영문을 인쇄할 수 있어요. 글을 영어로 입력해 주세요.")
        if any(not c.isprintable() and c != "\n" for c in sentence):
            raise RequestError(400, "인쇄할 수 없는 제어 문자가 있어요. 일반 텍스트로 붙여 넣어 주세요.")
        try:
            joint_book.board_conditions(sentence, self.font, PAGE_SIZE)
        except book.TextDoesNotFit as exc:
            raise RequestError(400, "입력한 글이 오른쪽 페이지 한 쪽을 넘어요. 문단이나 줄바꿈을 조금 줄여 주세요.") from exc
        return sentence

    def create(self, sentence, owner):
        sentence = self.page_text(sentence)
        import fcntl
        with self.guard, self.lock_file.open("a+", encoding="ascii") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RequestError(409, "다른 책을 생성 중이에요. 잠시 후 다시 시도해 주세요.")
            if shutil.disk_usage(self.folder).free < 64 * 1024**2:
                raise RequestError(507, "서버 저장 공간이 부족해요. 팀 관리자에게 알려 주세요.")
            job_id = secrets.token_hex(16)
            job = self.folder / job_id
            job.mkdir()
            seed = secrets.randbelow(2**32)
            request = dict(owner=owner, sentence=sentence, seed=seed)
            (job / "request.json").write_text(json.dumps(request), encoding="utf-8")
            lock.seek(0)
            lock.truncate()
            lock.write(job_id)
            lock.flush()
            cmd = [sys.executable, "-u", "-m", "pipeline", "--sentence=" + sentence,
                   "--seed", str(seed), "--output", str(job / "result"), "--model", str(self.model),
                   "--size", str(PAGE_SIZE)]
            # The child owns the GPU lock even if the web server restarts.
            with (job / "run.log").open("wb") as log:
                proc = subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                                        stderr=subprocess.STDOUT, pass_fds=(lock.fileno(),),
                                        start_new_session=True)
            threading.Thread(target=proc.wait, daemon=True).start()
        return dict(id=job_id, state="running", stage="starting", sentence=sentence, seed=seed)

    def status(self, job_id, owner):
        job = self.folder / job_id
        request = read_record(job / "request.json")
        if request.get("owner") != owner:
            raise RequestError(404, "이 브라우저에서 생성한 책을 찾을 수 없어요.")
        with self.guard:
            running = self.gpu_owner() == job_id
            run = read_record(job / "result" / "manifest.json")
        done = run.get("status") == "completed"
        stage = run.get("active") or ("spread" if done else "starting")
        state = "completed" if done else "running" if running and run.get("status") != "failed" else "failed"
        result = dict(id=job_id, state=state, stage=stage, sentence=request["sentence"], seed=request["seed"])
        if state == "failed":
            with (job / "run.log").open("rb") as log:
                log.seek(max(0, os.fstat(log.fileno()).st_size - 8192))
                tail = log.read().lower()
            if any(s in tail for s in (b"sentence does not fit", b"text is too tall", b"encoder length")):
                result["error"] = "문장이 페이지나 모델 입력 길이를 넘었어요. 더 짧은 문장으로 시도해 주세요."
            elif b"no space left" in tail:
                result["error"] = "서버 저장 공간이 부족해요. 팀 관리자에게 알려 주세요."
            else:
                result["error"] = "책 생성이 중단됐어요. 다시 시도해 주세요."
        return result


class Handler(BaseHTTPRequestHandler):
    def setup(self):
        super().setup()
        self.connection.settimeout(20)

    def send_data(self, code, body, kind, extra=()):
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Frame-Options", "DENY")
        for name, value in extra:
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def reply(self, code, data, extra=()):
        self.send_data(code, json.dumps(data, ensure_ascii=False).encode(), "application/json; charset=utf-8", extra)

    def cookies(self):
        jar = SimpleCookie()
        try:
            jar.load(self.headers.get("Cookie", ""))
        except CookieError:
            return None
        token = jar.get(COOKIE)
        if not token:
            return None
        parts = token.value.split(".")
        if len(parts) != 3 or not re.fullmatch("[a-f0-9]{32}", parts[0]) or not re.fullmatch(r"[0-9]{1,12}", parts[1]):
            return None
        signed = ".".join(parts[:2])
        valid = re.fullmatch("[0-9a-f]{64}", parts[2]) and hmac.compare_digest(self.signature(signed), parts[2])
        return parts[0] if valid and int(parts[1]) > time.time() else None

    def signature(self, value):
        return hmac.new(self.server.key, value.encode("ascii"), hashlib.sha256).hexdigest()

    def payload(self):
        if self.headers.get_content_type() != "application/json" or self.headers.get("Transfer-Encoding"):
            raise RequestError(415, "JSON 요청이 필요해요.")
        length = self.headers.get("Content-Length", "")
        if not re.fullmatch("[0-9]{1,12}", length) or int(length) == 0:
            raise RequestError(400, "요청 크기를 확인해 주세요.")
        if int(length) > 16 * 1024:
            raise RequestError(413, "요청 본문이 너무 커요. 책 한 쪽에 인쇄할 분량을 보내 주세요.")
        try:
            value = json.loads(self.rfile.read(int(length)).decode("utf-8"))
        except (UnicodeError, ValueError, RecursionError):
            raise RequestError(400, "요청 내용을 읽을 수 없어요.")
        if not isinstance(value, dict):
            raise RequestError(400, "요청 내용을 확인해 주세요.")
        return value

    def dispatch(self):
        host = self.headers.get("Host", "")
        authority = urlsplit("//" + host)
        origin = self.headers.get("Origin")
        if (authority.hostname not in self.server.hosts or authority.path or authority.username is not None
                or (origin and origin not in {"http://" + host, "https://" + host})):
            raise RequestError(403, "이 주소에서는 접속할 수 없어요.")
        path = urlsplit(self.path).path
        get = self.command == "GET"
        owner = self.cookies()
        if get and path == "/":
            return self.send_data(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        if get and path == "/api/health":
            return self.reply(200, dict(service="creativegen-book", authenticated=bool(owner)))
        if get and path in ("/api/example", "/api/example/background"):
            file = BACKGROUND_IMAGE if path == "/api/example/background" else FINAL_IMAGE
            example = ROOT / "runs/seagull" / file
            if example.is_file():
                return self.send_data(200, example.read_bytes(), "image/png")
        if not get and path == "/api/login":
            auth = self.payload()
            key = auth.get("key")
            invite = auth.get("invite")
            valid_key = isinstance(key, str) and hmac.compare_digest(key.encode(), self.server.key)
            valid_invite = (isinstance(invite, str) and re.fullmatch("[a-f0-9]{32}", invite)
                            and hmac.compare_digest(invite, team_invite(self.server.key)))
            if not (valid_key or valid_invite):
                raise RequestError(401, "팀 접속 코드를 확인해 주세요.")
            signed = f"{owner or secrets.token_hex(16)}.{int(time.time()) + WEEK}"
            cookie = f"{COOKIE}={signed}.{self.signature(signed)}; Path=/; HttpOnly; SameSite=Strict; Max-Age={WEEK}"
            return self.reply(200, {"authenticated": True}, [("Set-Cookie", cookie)])
        if path == "/api/jobs" or JOB_PATH.fullmatch(path):
            if not owner:
                raise RequestError(401, "팀 접속 코드를 입력해 주세요.")
            if not get and path == "/api/jobs":
                return self.reply(202, self.server.jobs.create(self.payload().get("sentence"), owner))
            match = JOB_PATH.fullmatch(path)
            if get and match:
                job_id, asset = match.groups()
                state = self.server.jobs.status(job_id, owner)
                if not asset:
                    return self.reply(200, state)
                if state["state"] != "completed":
                    raise RequestError(409, "완성된 이미지를 아직 열 수 없어요.")
                file = BACKGROUND_IMAGE if asset == "background" else FINAL_IMAGE
                image = self.server.jobs.folder / job_id / "result" / file
                extra = [("Content-Disposition", f'attachment; filename="book-{state["seed"]}.png"')] if asset == "download" else []
                return self.send_data(200, image.read_bytes(), "image/png", extra)
        raise RequestError(404, "페이지를 찾을 수 없어요.")

    def do_GET(self):
        try:
            self.dispatch()
        except RequestError as e:
            self.reply(e.code, {"error": str(e)})
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            pass
        except (ValueError, UnicodeError):
            self.reply(400, {"error": "요청 형식을 확인해 주세요."})
        except OSError as e:
            self.log_error("I/O error: %s", e)
            self.reply(500, {"error": "서버에서 요청을 처리할 수 없어요."})

    do_POST = do_GET


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=18764)
    parser.add_argument("--bind", default="127.0.0.1")
    args = parser.parse_args()
    key = os.environ.get("CREATIVEGEN_WEB_KEY")
    if not key:
        parser.error("Set CREATIVEGEN_WEB_KEY to the team's secret access code before starting.")
    jobs = Jobs(Path(os.environ.get("CREATIVEGEN_WEB_RUNS", ROOT / "runs/web")).resolve(),
                Path(os.environ.get("SDXL_BASE", ROOT / "models/sdxl-base")).resolve())
    with ThreadingHTTPServer((args.bind, args.port), Handler) as server:
        server.key = key.encode()
        server.hosts = {"localhost", "127.0.0.1", args.bind, os.environ.get("CREATIVEGEN_WEB_HOST")}
        server.jobs = jobs
        server.serve_forever()


if __name__ == "__main__":
    main()
