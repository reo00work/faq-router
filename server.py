#!/usr/bin/env python3
"""
faq-router/server.py
router.py（CLI）をブラウザから使える最小Web版にするローカルサーバー。

目的:
  router.py はターミナル操作（python3 router.py '質問文'）でしか動かず、
  非エンジニアのクライアントに問い合わせ自動振り分けの実演を見せられない。
  本サーバーは router.py のカテゴリ定義ロード・分類ロジック（キーワード → TF-IDF
  → LLM の3段階フォールバック）をそのまま再利用し、ブラウザから問い合わせ文を
  入力するとカテゴリ・判定方式・一次返信文を返すチャットUIを提供する。

依存: 標準ライブラリのみ（http.server）。新規pipパッケージは追加しない。
ANTHROPIC_API_KEY / OPENAI_API_KEY 未設定でも router.py と同じくキーワード＋TF-IDF
のみで動作する（Step 3 の LLM 分類はスキップされる）。

使い方:
    python3 server.py
    python3 server.py --port 8766
    → ブラウザで http://localhost:8766 を開き、問い合わせ文を入力する

他プロセス・テストからの利用:
    from server import classify_question, load_data
    categories = load_data()
    result = classify_question("料金を教えてください", categories)
"""

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))

from router import load_categories, route  # noqa: E402

DEFAULT_PORT = 8766

METHOD_LABELS = {
    "keyword": "キーワードルール",
    "tfidf": "TF-IDF 類似度",
    "llm": "LLM 分類",
    "fallback": "フォールバック",
}


def load_data() -> dict:
    """categories.json を読み込む（router.py の CLI と同一のロード関数を使う）。"""
    return load_categories()


def classify_question(question: str, categories: dict) -> dict:
    """router.py の route() をWeb向けに再利用し、JSONで返せる形に整える。"""
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


def make_handler(categories: dict):
    """categories をクロージャで束縛した BaseHTTPRequestHandler を返す。"""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass  # 標準エラー出力への毎リクエストログを抑制（動作確認時のノイズ削減）

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
            result = classify_question(question, categories)
            self._write_json(200, result)

    return Handler


def run(port: int = DEFAULT_PORT) -> None:
    categories = load_data()
    handler = make_handler(categories)
    server = ThreadingHTTPServer(("localhost", port), handler)
    print(f"faq-router Web版を起動しました: http://localhost:{port}")
    print("Ctrl+C で終了します。")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def parse_port(argv: list[str]) -> int:
    port = DEFAULT_PORT
    for i, a in enumerate(argv):
        if a == "--port" and i + 1 < len(argv):
            port = int(argv[i + 1])
    return port


if __name__ == "__main__":
    run(parse_port(sys.argv[1:]))
