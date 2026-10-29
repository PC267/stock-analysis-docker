#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""账号+密码+Authenticator(TOTP)认证，纯标准库实现。"""
import os, time, hmac, hashlib, base64, secrets, json, struct, threading
from pathlib import Path
from urllib.parse import parse_qs, quote
import yaml

DATA = Path(os.environ.get("AUTH_DATA", "/data/auth"))
USERS = DATA / "users.yaml"
FAILS, FAIL_LOCK = {}, threading.Lock()
MAX_FAILS, FAIL_WINDOW, LOCK_SECONDS = 5, 300, 900

def hash_password(password, iterations=260000):
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations)
    return f"pbkdf2_sha256${iterations}${base64.b64encode(salt).decode()}${base64.b64encode(dk).decode()}"

def check_password(stored, password):
    try:
        algo, iters, s, d = stored.split("$")
        if algo != "pbkdf2_sha256": return False
        dk = hashlib.pbkdf2_hmac("sha256", password.encode(), base64.b64decode(s), int(iters))
        return hmac.compare_digest(dk, base64.b64decode(d))
    except Exception:
        return False

def _b32d(s):
    s = s.upper(); return base64.b32decode(s + "=" * ((8 - len(s) % 8) % 8))

def _totp_at(secret, counter, digits=6):
    h = hmac.new(_b32d(secret), struct.pack(">Q", counter), hashlib.sha1).digest()
    o = h[-1] & 0x0F
    c = ((h[o] & 0x7F) << 24 | h[o+1] << 16 | h[o+2] << 8 | h[o+3]) % (10 ** digits)
    return str(c).zfill(digits)

def verify_totp(secret, code, window=1, step=30, digits=6):
    if not code or not code.isdigit() or len(code) != digits: return False
    now = int(time.time()) // step
    return any(hmac.compare_digest(_totp_at(secret, now + w, digits), code) for w in range(-window, window + 1))

def new_totp_secret():
    return base64.b32encode(secrets.token_bytes(20)).decode().rstrip("=")

def provisioning_uri(secret, username, issuer="WebStockPicker"):
    return f"otpauth://totp/{quote(issuer+':'+username)}?secret={secret}&issuer={quote(issuer)}&algorithm=SHA1&digits=6&period=30"

def load_users():
    return (yaml.safe_load(USERS.read_text(encoding="utf-8")) or {}) if USERS.exists() else {}

def save_users(users):
    DATA.mkdir(parents=True, exist_ok=True)
    t = USERS.with_suffix(".tmp"); t.write_text(yaml.safe_dump(users, allow_unicode=True, sort_keys=False), encoding="utf-8")
    os.chmod(t, 0o600); t.replace(USERS)

def create_user(username, password, totp_secret=None):
    users = load_users()
    if username in users: raise ValueError(f"用户已存在: {username}")
    secret = totp_secret or new_totp_secret()
    recovery = [secrets.token_urlsafe(12) for _ in range(10)]
    users[username] = {"username": username, "password_hash": hash_password(password),
        "totp_secret": secret, "recovery_codes": [hashlib.sha256(c.encode()).hexdigest() for c in recovery],
        "created": int(time.time())}
    save_users(users)
    return secret, provisioning_uri(secret, username), recovery

def delete_user(username):
    users = load_users()
    if username in users: del users[username]; save_users(users); return True
    return False

def verify_credentials(username, password, code):
    users = load_users(); rec = users.get(username)
    if not rec: check_password(hash_password("dummy"), password or ""); return None
    if not check_password(rec["password_hash"], password or ""): return None
    if verify_totp(rec["totp_secret"], code): return rec
    h = hashlib.sha256((code or "").strip().encode()).hexdigest()
    codes = rec.get("recovery_codes", [])
    if h in codes:
        codes.remove(h); users[username]["recovery_codes"] = codes; save_users(users); return rec
    return None

def _b64e(b): return base64.urlsafe_b64encode(b).decode().rstrip("=")
def _b64d(s): return base64.urlsafe_b64decode(s + "=" * ((4 - len(s) % 4) % 4))

def sign_session(payload, secret):
    raw = json.dumps(payload, separators=(",", ":")).encode()
    return f"{_b64e(raw)}.{_b64e(hmac.new(secret.encode(), raw, hashlib.sha256).digest())}"

def verify_session(token, secret, max_age=86400):
    try:
        a, b = token.split("."); raw = _b64d(a)
        if not hmac.compare_digest(_b64d(b), hmac.new(secret.encode(), raw, hashlib.sha256).digest()): return None
        p = json.loads(raw); return p if time.time() - p.get("iat", 0) <= max_age else None
    except Exception:
        return None

def parse_cookies(environ):
    out = {}
    for part in environ.get("HTTP_COOKIE", "").split(";"):
        if "=" in part: k, v = part.strip().split("=", 1); out[k] = v
    return out

def _ip(environ):
    return (environ.get("HTTP_X_FORWARDED_FOR", "").split(",")[0].strip() or environ.get("REMOTE_ADDR", "") or "unknown")

def _rate_ok(ip):
    now = time.time()
    with FAIL_LOCK:
        r = FAILS.get(ip)
        if not r: return True
        if r[2] and now < r[2]: return False
        if now - r[1] > FAIL_WINDOW: FAILS.pop(ip, None); return True
        return True

def _rate_fail(ip):
    now = time.time()
    with FAIL_LOCK:
        r = FAILS.get(ip)
        if not r or now - r[1] > FAIL_WINDOW: FAILS[ip] = [1, now, 0]; return
        r[0] += 1
        if r[0] >= MAX_FAILS: r[2] = now + LOCK_SECONDS

LOGIN_HTML = """<!doctype html><html lang=zh-CN><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><title>登录 - Web选股系统</title>
<style>body{{margin:0;background:#0f172a;color:#e5e7eb;font-family:-apple-system,"Microsoft YaHei",sans-serif}}
.box{{max-width:340px;margin:12vh auto;padding:24px;background:#111827;border:1px solid #1e293b;border-radius:12px}}
h3{{margin:0 0 16px}}input{{width:100%;box-sizing:border-box;background:#0f172a;color:#e5e7eb;border:1px solid #334155;
border-radius:8px;padding:10px 12px;margin:6px 0}}button{{width:100%;background:#2563eb;color:#fff;border:0;
border-radius:8px;padding:10px;margin-top:8px;cursor:pointer}}.err{{color:#f87171;margin:6px 0}}
.tip{{font-size:12px;color:#94a3b8;margin-top:10px}}</style></head><body><form class=box method=post>
<h3>Web选股系统</h3>{err}<input name=username placeholder=用户名 autofocus autocomplete=username>
<input name=password type=password placeholder=密码 autocomplete=current-password>
<input name=code inputmode=numeric autocomplete=one-time-code placeholder="Authenticator 六位码或恢复码">
<input type=hidden name=next value="{next}">
<button>登录</button><p class=tip>丢失设备时可用恢复码代替六位码（一次性）</p>
</form></body></html>"""

class AuthMiddleware:
    def __init__(self, app, secret_key, login_path="/login", logout_path="/logout"):
        self.app = app; self.secret = secret_key
        self.login_path = login_path; self.logout_path = logout_path
        self.public_exact = {"/health", login_path, logout_path, "/favicon.ico"}
        self.public_prefix = ("/static/",)

    def __call__(self, environ, start_response):
        p = environ.get("PATH_INFO", "/")
        if p == self.login_path:  return self._login(environ, start_response)
        if p == self.logout_path: return self._logout(start_response)
        if p in self.public_exact or any(p.startswith(x) for x in self.public_prefix):
            return self.app(environ, start_response)
        if not self._user(environ):
            if p.startswith("/api/"): return self._json401(start_response)
            return self._redirect(environ, start_response)
        return self.app(environ, start_response)

    def _user(self, environ):
        t = parse_cookies(environ).get("wsp_session")
        p = verify_session(t, self.secret) if t else None
        return p.get("user") if p else None

    def _login(self, environ, start_response):
        qs = parse_qs(environ.get("QUERY_STRING", ""))
        nxt = qs.get("next", ["/"])[0]
        if not nxt.startswith("/") or nxt.startswith("//"): nxt = "/"
        err = ""
        if environ.get("REQUEST_METHOD") == "POST":
            ip = _ip(environ)
            if not _rate_ok(ip): err = "尝试过多，请 15 分钟后再试"
            else:
                try:
                    n = int(environ.get("CONTENT_LENGTH") or 0)
                    f = parse_qs(environ["wsgi.input"].read(n).decode("utf-8", "replace"))
                except Exception: f = {}
                u = (f.get("username", [""])[0]).strip(); pw = f.get("password", [""])[0]
                c = (f.get("code", [""])[0]).strip(); nxt = f.get("next", ["/"])[0] or "/"
                if not nxt.startswith("/") or nxt.startswith("//"): nxt = "/"
                if verify_credentials(u, pw, c):
                    ck = f"wsp_session={sign_session({'user': u, 'iat': int(time.time())}, self.secret)}; Path=/; HttpOnly; SameSite=Lax; Max-Age=86400"
                    start_response("302 Found", [("Location", nxt), ("Set-Cookie", ck), ("Content-Type", "text/plain; charset=utf-8")])
                    return [b"ok"]
                _rate_fail(ip); err = "用户名、密码或验证码错误"
        html = LOGIN_HTML.format(err=f'<p class=err>{err}</p>' if err else "", next=nxt.replace('"', "&quot;")).encode()
        start_response("200 OK", [("Content-Type", "text/html; charset=utf-8"), ("Content-Length", str(len(html)))])
        return [html]

    def _logout(self, start_response):
        start_response("302 Found", [("Location", self.login_path),
            ("Set-Cookie", "wsp_session=; Path=/; HttpOnly; SameSite=Lax; Max-Age=0"),
            ("Content-Type", "text/plain; charset=utf-8")])
        return [b"bye"]

    def _json401(self, start_response):
        d = b'{"error":"unauthorized"}'
        start_response("401 Unauthorized", [("Content-Type", "application/json"), ("Content-Length", str(len(d)))])
        return [d]

    def _redirect(self, environ, start_response):
        nxt = environ.get("PATH_INFO", "/") + (("?" + environ["QUERY_STRING"]) if environ.get("QUERY_STRING") else "")
        start_response("302 Found", [("Location", f"{self.login_path}?next={quote(nxt)}"), ("Content-Type", "text/plain; charset=utf-8")])
        return [b"redirect"]
