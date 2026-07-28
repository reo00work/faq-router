#!/usr/bin/env python3
"""
faq-router/router.py
問い合わせ文をカテゴリ（料金/納期/技術質問/その他）に振り分け、定型一次返信文を返す CLI。

使い方:
  python router.py '料金を教えてください'
  python router.py --all              # サンプル問い合わせを一括テスト
  python router.py --debug '納期は？' # スコア詳細を表示

分類ロジック（優先順位）:
  1. キーワードルール  — categories.json の keywords にヒットしたら即決定
  2. TF-IDF 類似度   — カテゴリ代表文との TF-IDF コサイン類似度で判定
  3. LLM 分類        — ANTHROPIC_API_KEY / OPENAI_API_KEY が設定されている場合のみ
  ※ APIキー未設定でも完全ローカルで動作します（標準ライブラリのみ・追加パッケージ不要）
"""

import json
import math
import os
import re
import sys
from pathlib import Path

CATEGORIES_PATH = Path(__file__).parent / "categories.json"

SAMPLE_QUESTIONS = [
    "料金を教えてください",
    "費用はどのくらいかかりますか？",
    "見積もりをお願いしたいです",
    "納期はどのくらいですか？",
    "いつ完成しますか？",
    "スケジュールを確認したい",
    "どんな技術を使っていますか？",
    "既存システムとの連携はできますか？",
    "APIとの連携は可能ですか？",
    "ちょっと相談があります",
    "詳しく話を聞きたい",
    "コストはどれくらいですか",
]


# ---------------------------------------------------------------------------
# テキスト処理
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    """日本語・英数字をトークン化。標準ライブラリのみで動作。"""
    return re.findall(r"[a-z0-9]+|[぀-ゟ]+|[゠-ヿ]+|[一-鿿]+", text.lower())


# ---------------------------------------------------------------------------
# カテゴリ定義ロード
# ---------------------------------------------------------------------------

def load_categories(path: Path = CATEGORIES_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))["categories"]


# ---------------------------------------------------------------------------
# Step 1: キーワードルール
# ---------------------------------------------------------------------------

def keyword_match(question: str, categories: dict) -> str | None:
    """
    categories.json の keywords にヒットしたカテゴリを返す。
    「その他」は keywords が空なのでスキップ。
    複数ヒットの場合はヒット数が最多のカテゴリを選ぶ。
    """
    scores: dict[str, int] = {}
    for cat_name, cat_data in categories.items():
        if cat_name == "その他":
            continue
        count = sum(1 for kw in cat_data["keywords"] if kw in question)
        if count > 0:
            scores[cat_name] = count
    if not scores:
        return None
    return max(scores, key=lambda k: scores[k])


# ---------------------------------------------------------------------------
# Step 2: TF-IDF コサイン類似度
# ---------------------------------------------------------------------------

def build_tfidf_index(categories: dict) -> tuple[dict, dict]:
    """
    全カテゴリの代表文（examples）からカテゴリ別 TF-IDF 平均ベクトルを構築。
    返り値: (idf辞書, {カテゴリ名: L2正規化済みベクトル})
    """
    all_docs: list[list[str]] = []
    doc_cat_map: list[str] = []

    for cat_name, cat_data in categories.items():
        for ex in cat_data["examples"]:
            tokens = tokenize(ex)
            all_docs.append(tokens)
            doc_cat_map.append(cat_name)

    N = len(all_docs)
    df: dict[str, int] = {}
    for tokens in all_docs:
        for t in set(tokens):
            df[t] = df.get(t, 0) + 1

    idf = {t: math.log((N + 1) / (cnt + 1)) + 1.0 for t, cnt in df.items()}

    cat_vecs: dict[str, dict] = {k: {} for k in categories}
    cat_counts: dict[str, int] = {k: 0 for k in categories}

    for tokens, cat_name in zip(all_docs, doc_cat_map):
        total = len(tokens) or 1
        tf: dict[str, float] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        for t, c in tf.items():
            cat_vecs[cat_name][t] = cat_vecs[cat_name].get(t, 0.0) + (c / total) * idf.get(t, 0.1)
        cat_counts[cat_name] += 1

    for cat_name in cat_vecs:
        n = cat_counts[cat_name] or 1
        vec = {t: v / n for t, v in cat_vecs[cat_name].items()}
        norm = math.sqrt(sum(v ** 2 for v in vec.values())) or 1.0
        cat_vecs[cat_name] = {t: v / norm for t, v in vec.items()}

    return idf, cat_vecs


def tfidf_classify(question: str, idf: dict, cat_vecs: dict) -> tuple[str, dict[str, float]]:
    """
    TF-IDF ベクトル化してコサイン類似度でカテゴリ判定。
    返り値: (最スコアカテゴリ名, {カテゴリ名: スコア})
    """
    tokens = tokenize(question)
    if not tokens:
        return "その他", {}

    total = len(tokens)
    tf: dict[str, float] = {}
    for t in tokens:
        tf[t] = tf.get(t, 0) + 1

    q_vec = {t: (c / total) * idf.get(t, 0.1) for t, c in tf.items()}
    norm = math.sqrt(sum(v ** 2 for v in q_vec.values())) or 1.0
    q_vec = {t: v / norm for t, v in q_vec.items()}

    scores: dict[str, float] = {}
    for cat_name, cat_vec in cat_vecs.items():
        scores[cat_name] = sum(q_vec.get(t, 0.0) * v for t, v in cat_vec.items())

    best = max(scores, key=lambda k: scores[k])
    return best, scores


# ---------------------------------------------------------------------------
# Step 3: LLM 分類（APIキーがある場合のみ）
# ---------------------------------------------------------------------------

def llm_classify(question: str, category_names: list[str]) -> str | None:
    """
    LLM にカテゴリ分類を依頼。APIキー未設定なら None を返す。
    ANTHROPIC_API_KEY を優先し、次に OPENAI_API_KEY を試みる。
    """
    cats_str = "・".join(category_names)
    prompt = (
        f"次の問い合わせ文を以下のカテゴリのいずれかに分類してください。\n"
        f"カテゴリ: {cats_str}\n\n"
        f"問い合わせ: {question}\n\n"
        f"カテゴリ名のみを回答してください（説明・句読点不要）:"
    )

    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            import anthropic
            client = anthropic.Anthropic()
            msg = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=32,
                messages=[{"role": "user", "content": prompt}],
            )
            result = msg.content[0].text.strip()
            if result in category_names:
                return result
        except Exception as e:
            print(f"[LLM] Anthropic API エラー: {e}", file=sys.stderr)

    if os.environ.get("OPENAI_API_KEY"):
        try:
            import openai
            client = openai.OpenAI()
            resp = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=32,
            )
            result = resp.choices[0].message.content.strip()
            if result in category_names:
                return result
        except Exception as e:
            print(f"[LLM] OpenAI API エラー: {e}", file=sys.stderr)

    return None


# ---------------------------------------------------------------------------
# メイン分類関数
# ---------------------------------------------------------------------------

TFIDF_THRESHOLD = 0.05  # これ以上でTF-IDFを採用

def route(question: str, categories: dict, debug: bool = False) -> tuple[str, str, str]:
    """
    問い合わせを分類し (カテゴリ名, 判定方法, 一次返信文) を返す。
    判定方法: "keyword" | "tfidf" | "llm" | "fallback"
    """
    category_names = list(categories.keys())

    # Step 1: キーワードルール
    kw_result = keyword_match(question, categories)
    if kw_result:
        if debug:
            print(f"  [判定] キーワードルール → {kw_result}", file=sys.stderr)
        return kw_result, "keyword", categories[kw_result]["response"]

    # Step 2: TF-IDF 類似度
    idf, cat_vecs = build_tfidf_index(categories)
    tfidf_result, scores = tfidf_classify(question, idf, cat_vecs)

    if debug:
        for cat, score in sorted(scores.items(), key=lambda x: -x[1]):
            print(f"  [TF-IDF] {cat}: {score:.4f}", file=sys.stderr)

    top_score = scores.get(tfidf_result, 0.0)
    if top_score >= TFIDF_THRESHOLD:
        if debug:
            print(f"  [判定] TF-IDF → {tfidf_result} (score={top_score:.4f})", file=sys.stderr)
        return tfidf_result, "tfidf", categories[tfidf_result]["response"]

    # Step 3: LLM（APIキーがある場合のみ）
    llm_result = llm_classify(question, category_names)
    if llm_result:
        if debug:
            print(f"  [判定] LLM → {llm_result}", file=sys.stderr)
        return llm_result, "llm", categories[llm_result]["response"]

    # Fallback: その他
    if debug:
        print(f"  [判定] fallback → その他 (TF-IDF max={top_score:.4f})", file=sys.stderr)
    return "その他", "fallback", categories["その他"]["response"]


# ---------------------------------------------------------------------------
# CLI エントリポイント
# ---------------------------------------------------------------------------

METHOD_LABELS = {
    "keyword": "キーワードルール",
    "tfidf":   "TF-IDF 類似度",
    "llm":     "LLM 分類",
    "fallback": "フォールバック",
}

METHOD_ABBR = {"keyword": "KW", "tfidf": "TF", "llm": "LM", "fallback": "FB"}


def main() -> None:
    args = sys.argv[1:]
    debug = "--debug" in args
    show_all = "--all" in args
    args = [a for a in args if a not in ("--debug", "--all")]

    categories = load_categories()

    if show_all:
        print("=" * 60)
        print("サンプル一括テスト")
        print("=" * 60)
        for q in SAMPLE_QUESTIONS:
            cat, method, response = route(q, categories, debug=debug)
            abbr = METHOD_ABBR[method]
            print(f"\n[{abbr}] 問い合わせ: {q}")
            print(f"  → カテゴリ: {cat}")
            for line in response.split("\n"):
                print(f"  | {line}")
        print()
        print("=" * 60)
        print(f"合計: {len(SAMPLE_QUESTIONS)} 件処理")
        return

    if not args:
        print(__doc__)
        sys.exit(0)

    question = " ".join(args)

    if debug:
        print(f"[DEBUG] 入力: {question}", file=sys.stderr)

    cat, method, response = route(question, categories, debug=debug)

    print(f"\n問い合わせ: {question}")
    print(f"カテゴリ  : 【{cat}】 （判定方法: {METHOD_LABELS[method]}）")
    print()
    print("一次返信文:")
    print("-" * 50)
    print(response)
    print("-" * 50)


if __name__ == "__main__":
    main()
