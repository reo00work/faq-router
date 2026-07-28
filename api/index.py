#!/usr/bin/env python3
"""
faq-router/api/index.py
Vercel Python Runtime向けエントリーポイント。

server.py（ローカル実行用: ThreadingHTTPServerで起動）と同じロジックを、
Vercelのサーバーレス関数として動かすための薄いラッパー。

Vercelの Python Runtime は、api/配下のファイルに
`BaseHTTPRequestHandler` を継承した `handler` という名前のクラスを
置くと、そのままサーバーレス関数としてルーティングする仕様
（1リクエスト = 1関数呼び出し。コンテナは複数リクエストで再利用されうる）。

ロジック本体（load_categories/route）は router.py をそのまま import して
再利用する。router.py 自体は無変更。
"""

import json
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path

# api/ の1つ上（faq-router/ ディレクトリ）を sys.path に追加し、
# 同ディレクトリの router.py をそのまま import できるようにする。
# server.py の `sys.path.insert(0, str(BASE))` と同じ考え方。
BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))

from router import load_categories, route  # noqa: E402

METHOD_LABELS = {
    "keyword": "キーワードルール",
    "tfidf": "TF-IDF 類似度",
    "llm": "LLM 分類",
    "fallback": "フォールバック",
}

# モジュールロード時（コールドスタート時）に1回だけ categories.json を読み込む。
# 同一コンテナがウォームなまま次のリクエストを処理する場合はこの再読み込みは発生しない。
_CATEGORIES = load_categories()


def classify_question(question: str, categories: dict) -> dict:
    """server.py と同一ロジック（router.py の route() をWeb向けに再利用）。"""
    question = (question or "").strip()
    if not question:
        return {
            "question": question,
            "category": None,
            "method": None,
            "method_label": None,
            "response": "",
        }

    category, method, response = route(question, categories)
    return {
        "question": question,
        "category": category,
        "method": method,
        "method_label": METHOD_LABELS.get(method, method),
        "response": response,
    }


def render_index_html() -> bytes:
    html = """<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<title>faq-router — 問い合わせ自動振り分け（デモ）</title>
<style>
  body { font-family: -apple-system, "Hiragino Sans", sans-serif; max-width: 640px; margin: 40px auto; padding: 0 16px; background: #f7f7f8; }
  h1 { font-size: 18px; }
  #log { background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 16px; min-height: 240px; margin-bottom: 12px; }
  .msg { margin-bottom: 14px; white-space: pre-wrap; line-height: 1.6; }
  .msg.user { text-align: right; color: #1a1a1a; }
  .msg.bot { color: #1a1a1a; }
  .meta { font-size: 12px; color: #666; margin-top: 4px; }
  form { display: flex; gap: 8px; }
  input[type=text] { flex: 1; padding: 10px; border: 1px solid #ccc; border-radius: 6px; }
  button { padding: 10px 18px; border: none; border-radius: 6px; background: #2d6cdf; color: #fff; cursor: pointer; }
  button:disabled { background: #999; cursor: default; }
</style>
</head>
<body>
<h1>faq-router — 問い合わせ自動振り分け（デモ・ローカル版）</h1>
<div id="log"><div class="msg bot">お問い合わせ文を入力してください（例: 料金を教えてください）</div></div>
<form id="form">
  <input type="text" id="question" placeholder="問い合わせ文を入力" autocomplete="off">
  <button type="submit" id="send">送信</button>
</form>
<script>
const log = document.getElementById('log');
const form = document.getElementById('form');
const input = document.getElementById('question');
const send = document.getElementById('send');

function addMessage(text, cls) {
  const div = document.createElement('div');
  div.className = 'msg ' + cls;
  div.textContent = text;
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
  return div;
}

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const question = input.value.trim();
  if (!question) return;
  addMessage(question, 'user');
  input.value = '';
  send.disabled = true;
  try {
    const res = await fetch('/route', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({question})
    });
    const data = await res.json();
    const botDiv = addMessage(data.response, 'bot');
    const metaDiv = document.createElement('div');
    metaDiv.className = 'meta';
    metaDiv.textContent = `カテゴリ: ${data.category}（判定方式: ${data.method_label}）`;
    botDiv.appendChild(document.createElement('br'));
    botDiv.appendChild(metaDiv);
  } catch (err) {
    addMessage('通信エラーが発生しました。サーバーが起動しているか確認してください。', 'bot');
  } finally {
    send.disabled = false;
    input.focus();
  }
});
</script>
</body>
</html>
"""
    return html.encode("utf-8")


class handler(BaseHTTPRequestHandler):
    """Vercel Python Runtime が呼び出すエントリークラス（クラス名は固定で `handler`）。"""

    def log_message(self, fmt, *args):
        pass

    def _write_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = render_index_html()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path != "/route":
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            payload = {}
        question = payload.get("question", "") if isinstance(payload, dict) else ""
        result = classify_question(question, _CATEGORIES)
        self._write_json(200, result)
