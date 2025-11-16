import streamlit as st
import pandas as pd
import numpy as np
import os
import re
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# .envファイルをロードして環境変数を設定
load_dotenv()

# --- Gemini API v1 エンドポイントを明示的に設定 ---
# APIキーを環境変数から読み込み
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    # .env ファイルが読み込めていないか、キーが設定されていない
    st.error("APIキーが見つかりません。.envファイルに GOOGLE_API_KEY が正しく設定されているか確認してください。")
    st.stop()
else:
    try:
        # v1beta (古いAPI) ではなく v1 (新しいAPI) のエンドポイントを明示的に指定
        genai.configure(
            api_key=api_key,
            client_options={'api_endpoint': 'generativelanguage.googleapis.com'}
        )
        st.write("Gemini API v1 エンドポイントを明示的に設定しました。") # 動作確認用
    except TypeError:
        st.warning("ライブラリが古いようです。デフォルト設定で接続します（エラーが再発する可能性があります）。")
        genai.configure(api_key=api_key)
    except Exception as e:
        st.error(f"Gemini APIの設定に失敗しました: {e}")
        st.stop()
# --- 追加ブロックここまで ---


# CSVファイルを読み込む関数
@st.cache_data
def load_data(csv_file_path):
    """
    指定されたパスからCSVファイルを読み込み、DataFrameを返す。
    ファイルが見つからない場合はエラーを表示して停止する。
    """
    try:
        df = pd.read_csv(csv_file_path)
        return df
    except FileNotFoundError:
        st.error(f"エラー: CSVファイル ({csv_file_path}) が見つかりません。")
        st.info(f"app.py と同じフォルダに {csv_file_path} を配置してください。")
        st.stop()  # ファイルがないと続行できないためアプリを停止
    except Exception as e:
        st.error(f"データの読み込み中に予期せぬエラーが発生しました: {e}")
        st.stop()

# TF-IDFモデルを構築する関数
@st.cache_resource
def build_tfidf_model(texts):
    """
    与えられたテキストのリストからTF-IDFベクトルライザとTF-IDF行列を構築する。
    """
    st.write("TF-IDFモデルを構築中...") # 動作確認用
    vectorizer = TfidfVectorizer(
        max_features=5000,  # 語彙数を5000に制限（メモリと速度のため）
        max_df=0.95,        # 95%以上の文書に出現する単語は無視
        min_df=2            # 2回未満しか出現しない単語は無視
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    st.write("TF-IDFモデル構築完了。") # 動作確認用
    return tfidf_matrix, vectorizer

# SentenceTransformerの埋め込みモデルを取得する関数
@st.cache_resource
def get_embedding_model():
    """
    SentenceTransformerの埋め込みモデル（多言語対応）をロードする。
    """
    st.write("埋め込みモデル（SentenceTransformer）をロード中...")
    # 高性能な多言語対応モデルを使用
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    st.write("埋め込みモデルのロード完了。")
    return model

# テキストデータをベクトル化する関数
@st.cache_resource
def build_embedding_model(texts, _model):
    """
    与えられたテキストのリストとモデルを使って、埋め込みベクトルを計算する。
    """
    st.write("テキストのベクトル化（埋め込み）を計算中...")
    embeddings = _model.encode(texts, show_progress_bar=True)
    st.write("テキストのベクトル化完了。")
    return embeddings

# Geminiモデルを取得する関数 (v1 API強制指定版)
# @st.cache_resource  # ← キャッシュを停止（コメントアウト）
def get_gemini_model():
    """
    Gemini (標準モデル) のモデルをロードする。
    """
    st.write("生成AIモデル（Gemini）をロード中...")
    
    # v1 APIエンドポイントは app.py 冒頭の genai.configure() でグローバル設定済み
    
    # 安定版の 'gemini-pro' に変更
    model = genai.GenerativeModel(
        'gemini-pro'
    ) 
    st.write("生成AIモデルのロード完了。")
    return model

# ハイブリッド検索を行う関数
def hybrid_search(query, tfidf_vectorizer, tfidf_matrix, embedding_model, embeddings, top_n=5):
    """
    TF-IDF (キーワード) と SBERT (意味) の両方を使ってハイブリッド検索を行う。
    """
    
    # 1. TF-IDFスコアの計算 (キーワード検索)
    query_tfidf = tfidf_vectorizer.transform([query])
    tfidf_scores = cosine_similarity(query_tfidf, tfidf_matrix).flatten()
    
    # 2. SBERTスコアの計算 (意味検索)
    query_embedding = embedding_model.encode([query])
    semantic_scores = cosine_similarity(query_embedding, embeddings).flatten()
    
    # 3. ハイブリッドスコアの計算 (重み付け平均)
    hybrid_scores = (0.5 * tfidf_scores) + (0.5 * semantic_scores)
    
    # 4. スコアの高い順にソートし、インデックスとスコアを取得
    top_indices = hybrid_scores.argsort()[::-1][:top_n]
    
    # [(インデックス1, スコア1), (インデックス2, スコア2), ...] の形式で返す
    results = [(int(i), float(hybrid_scores[i])) for i in top_indices]
    
    return results

# Geminiモデルを使って応答を生成する関数 (RAG)
def respond_with_gemini(query, model, results, texts, top_n=3):
    """
    RAG (Retrieval-Augusted Generation) を実行する。
    検索結果（コンテキスト）を基に、Geminiモデルが回答を生成する。
    """
    
    # 1. 検索結果から上位 top_n 件の「記事本文」を取得
    context_list = []
    for (index, score) in results[:top_n]:
        context_list.append(texts[index])
    
    # 2. コンテキストを結合して1つの文字列にする
    context = "\n---\n".join(context_list)
    
    # 3. Geminiへのプロンプト（指示文）を作成
    prompt_template = f"""
あなたは、Yahoo!ニュースの記事について回答するAIアシスタントです。
以下の「参照記事」に書かれている情報**のみ**に基づいて、ユーザーの「質問」に回答してください。

# 制約条件:
- 参照記事に書かれていない事柄や、あなたの一般的な知識で回答してはいけません。
- 参照記事に該当する情報がない場合は、その旨を正直に伝えてください（例：「ご質問に関連する記事が見つかりませんでした。」）。

# 参照記事:
{context}

# 質問:
{query}

# 回答:
"""
    
    # 4. Gemini APIを呼び出して回答を生成
    try:
        response = model.generate_content(prompt_template)
        return response.text
    except Exception as e:
        st.error(f"Geminiでの回答生成中にエラーが発生しました: {e}")
        return "申し訳ありません。回答の生成中にエラーが発生しました。"

# チャット履歴を初期化する関数
def init_chat_history():
    """
    Streamlitのセッション状態を利用してチャット履歴を初期化する。
    """
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "こんにちは。Yahoo!ニュースの記事に関するご質問をどうぞ。"}
        ]

# チャット履歴を表示する関数
def display_chat_history():
    """
    現在のチャット履歴をStreamlitのチャットUIで表示する。
    """
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# --- Streamlitアプリのメイン実行部分 ---
st.title("RAG System")

# STEP 1 & 2: データのロードとテキストの準備
csv_file_path = "yahoo_news_articles.csv"
df = load_data(csv_file_path)

if "text" in df.columns:
    texts = df["text"].fillna("").tolist()
    st.write(f"記事データのロード完了。記事数: {len(texts)} 件") # 動作確認用
else:
    st.error("エラー: 'text' 列がCSVファイルに見つかりません。")
    st.stop()

# STEP 3: TF-IDFモデルを構築
tfidf_matrix, tfidf_vectorizer = build_tfidf_model(texts)

# STEP 4: 埋め込みモデルを構築
embedding_model = get_embedding_model()
embeddings = build_embedding_model(texts, embedding_model)

# STEP 5: Geminiモデルをロード
# v1betaエラーを回避するため、キャッシュ(cache_resource)の代わりに
# session_state を使い、configure直後に生成されたモデルを保持する
if "gemini_model" not in st.session_state:
    st.session_state.gemini_model = get_gemini_model()
model = st.session_state.gemini_model # session_stateからモデルを取得

# STEP 6: チャット履歴の初期化と表示
init_chat_history()
display_chat_history()

# STEP 9: ユーザー入力とRAGの実行
user_input = st.chat_input("質問を入力してください")
if user_input:
    # 1. ユーザーの入力をチャット履歴に追加して表示
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2. RAG (Retrieval-Augmented Generation) の実行
    
    # 2a. ハイブリッド検索 (Retrieval)
    search_results = hybrid_search(
        user_input, 
        tfidf_vectorizer, 
        tfidf_matrix, 
        embedding_model, 
        embeddings, 
        top_n=5 # まず上位5件を検索
    )
    
    # 2b. 回答生成 (Generation)
    response_text = respond_with_gemini(
        user_input, 
        model, 
        search_results, 
        texts, 
        top_n=3 # 検索結果5件のうち、特に精度の高い上位3件をコンテキストとしてLLMに渡す
    )
    
    # 3. AIの応答をチャット履歴に追加して表示
    st.session_state.messages.append({"role": "assistant", "content": response_text})
    with st.chat_message("assistant"):
        st.markdown(response_text)