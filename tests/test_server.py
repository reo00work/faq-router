"""
faq-router/tests/test_server.py
server.py（router.py のロジックを再利用するローカルWeb版）を検証する。

実サーバーをスレッドで起動し、urllib.request で実際にHTTPリクエストを送って
GET / と POST /route の応答を確認する（標準ライブラリのみ・新規pip依存なし）。

実行方法:
    python3 -m unittest discover -s faq-router/tests -v
"""
import json
import sys
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from threading import Thread

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server import classify_question, load_data, make_handler  # noqa: E402


class ServerHTTPTest(unittest.TestCase):
    """実際にHTTPサーバーを起動してエンドツーエンドで検証する。"""

    @classmethod
    def setUpClass(cls):
        cls.categories = load_data()
        handler = make_handler(cls.categories)
        # port=0 で空きポートをOSに割り当ててもらう（他プロセスとの衝突回避）
        cls.httpd = ThreadingHTTPServer(("localhost", 0), handler)
        cls.port = cls.httpd.server_address[1]
        cls.thread = Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://localhost:{cls.port}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.thread.join(timeout=2)

    def _post_route(self, question: str) -> tuple[int, dict]:
        data = json.dumps({"question": question}).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/route",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as res:
            return res.status, json.loads(res.read().decode("utf-8"))

    def test_get_index_returns_html_with_chat_form(self):
        with urllib.request.urlopen(f"{self.base_url}/") as res:
            self.assertEqual(res.status, 200)
            self.assertIn("text/html", res.headers.get("Content-Type", ""))
            body = res.read().decode("utf-8")
        self.assertIn("faq-router", body)
        self.assertIn("<form", body)  # 問い合わせ入力フォームが存在する

    def test_route_keyword_question_returns_category_and_response(self):
        status, body = self._post_route("料金を教えてください")
        self.assertEqual(status, 200)
        self.assertEqual(body["category"], "料金")
        self.assertEqual(body["method"], "keyword")
        self.assertTrue(body["response"])

    def test_route_tfidf_question_returns_sonota(self):
        status, body = self._post_route("ちょっと相談があります")
        self.assertEqual(status, 200)
        self.assertEqual(body["category"], "その他")

    def test_route_empty_question_returns_empty_result(self):
        status, body = self._post_route("")
        self.assertEqual(status, 200)
        self.assertIsNone(body["category"])
        self.assertEqual(body["response"], "")

    def test_unknown_path_returns_404(self):
        req = urllib.request.Request(f"{self.base_url}/no-such-path")
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(req)
        self.assertEqual(ctx.exception.code, 404)


class ClassifyQuestionUnitTest(unittest.TestCase):
    """HTTPを介さずロジック単体を検証する。"""

    @classmethod
    def setUpClass(cls):
        cls.categories = load_data()

    def test_classify_question_returns_expected_keys(self):
        result = classify_question("納期はどのくらいですか？", self.categories)
        self.assertEqual(
            set(result.keys()),
            {"question", "category", "method", "method_label", "response"},
        )
        self.assertEqual(result["category"], "納期")


if __name__ == "__main__":
    unittest.main()
