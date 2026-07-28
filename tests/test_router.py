"""
faq-router/tests/test_router.py
router.py の route() / keyword_match() / tfidf_classify() を検証する。

LLM 分類（ANTHROPIC_API_KEY / OPENAI_API_KEY 使用）は対象外
（APIキー未設定環境を前提とし、ローカル完結のロジックのみ検証する）。

実行方法:
    python3 -m unittest discover -s faq-router/tests -v
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from router import (  # noqa: E402
    build_tfidf_index,
    keyword_match,
    load_categories,
    route,
    tfidf_classify,
)


class KeywordMatchTest(unittest.TestCase):
    def setUp(self):
        self.categories = load_categories()

    def test_single_keyword_hit(self):
        """「料金」キーワードに一致する問い合わせは料金カテゴリと判定される。"""
        result = keyword_match("料金を教えてください", self.categories)
        self.assertEqual(result, "料金")

    def test_competing_categories_picks_highest_hit_count(self):
        """複数カテゴリのキーワードが競合する場合、ヒット数が多い方が勝つ。"""
        question = "料金と価格について、あと予算も教えてください。ついでに納期も知りたいです。"
        # 料金カテゴリは「料金」「価格」「予算」の3語がヒット、納期カテゴリは「納期」の1語のみ
        result = keyword_match(question, self.categories)
        self.assertEqual(result, "料金")

    def test_no_keyword_hit_returns_none(self):
        """どのカテゴリのキーワードにも一致しない場合は None を返す。"""
        result = keyword_match("既存の基盤とつなげることは可能でしょうか", self.categories)
        self.assertIsNone(result)


class TfidfClassifyTest(unittest.TestCase):
    def setUp(self):
        self.categories = load_categories()
        self.idf, self.cat_vecs = build_tfidf_index(self.categories)

    def test_classifies_paraphrase_without_keyword_match(self):
        """キーワード非一致だが代表文と意味的に近い言い回しは TF-IDF で分類できる。"""
        question = "既存の基盤とつなげることは可能でしょうか"
        self.assertIsNone(keyword_match(question, self.categories))

        best, scores = tfidf_classify(question, self.idf, self.cat_vecs)
        self.assertEqual(best, "技術質問")
        self.assertGreaterEqual(scores[best], 0.05)


class RouteTest(unittest.TestCase):
    def setUp(self):
        self.categories = load_categories()

    def test_route_keyword_priority(self):
        """キーワードルールが優先され、判定方法は "keyword" になる。"""
        cat, method, response = route("料金を教えてください", self.categories)
        self.assertEqual(cat, "料金")
        self.assertEqual(method, "keyword")
        self.assertTrue(response)

    def test_route_falls_back_to_tfidf_when_no_keyword_match(self):
        """キーワード非一致でも TF-IDF スコアが閾値以上ならそのカテゴリに分類される。"""
        cat, method, response = route("既存の基盤とつなげることは可能でしょうか", self.categories)
        self.assertEqual(cat, "技術質問")
        self.assertEqual(method, "tfidf")
        self.assertTrue(response)

    def test_route_ambiguous_input_falls_back_to_other(self):
        """どのカテゴリにも該当しない曖昧な入力は「その他」にフォールバックする。"""
        cat, method, response = route("猫が好きです", self.categories)
        self.assertEqual(cat, "その他")
        self.assertEqual(method, "fallback")
        self.assertTrue(response)


class CategoriesResponseTest(unittest.TestCase):
    def test_all_categories_have_non_empty_response(self):
        """categories.json の4カテゴリ全てで response が空文字列でないことを確認する。"""
        categories = load_categories()
        self.assertEqual(
            set(categories.keys()), {"料金", "納期", "技術質問", "その他"}
        )
        for cat_name, cat_data in categories.items():
            with self.subTest(category=cat_name):
                self.assertTrue(cat_data["response"].strip())


if __name__ == "__main__":
    unittest.main()
