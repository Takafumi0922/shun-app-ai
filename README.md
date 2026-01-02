# 透析シャント音 AI解析アプリ 🩺

透析患者が自身のシャント音を録音し、**FFT周波数解析** と **Gemini AI診断** でシャントの健全性をチェックするWebアプリです。

## 🌐 デモ

**Streamlit Cloud でアクセス:**  
(デプロイ後にURLを設定)

## ✨ 機能

- 📊 **FFT周波数解析**: 録音音声から周波数スペクトルを可視化
- 🤖 **AI診断**: Gemini 2.5 Flash による専門的な音声評価
- 📱 **スマホ対応**: モバイルブラウザで簡単に録音・解析

## 🚀 デプロイ方法

### 1. GitHubにリポジトリ作成

```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/shunt-sound-analyzer.git
git push -u origin main
```

### 2. Streamlit Cloud でデプロイ

1. [Streamlit Cloud](https://share.streamlit.io/) にアクセス
2. GitHubアカウントでログイン
3. 「New app」からリポジトリを選択
4. Main file: `app.py` を指定
5. 「Deploy」をクリック

### 3. APIキーの設定（Secrets）

Streamlit Cloud の設定画面で、**Secrets** に以下を追加：

```toml
GOOGLE_API_KEY = "YOUR_ACTUAL_API_KEY_HERE"
```

> ⚠️ **重要**: APIキーは絶対にコードにハードコードせず、必ずSecretsで管理してください。

## 📁 ファイル構成

```
shunt app/
├── app.py              # メインアプリケーション
├── requirements.txt    # 依存ライブラリ
├── .gitignore          # Git除外設定
├── .streamlit/
│   └── config.toml     # Streamlit設定
└── README.md           # このファイル
```

## 🔒 セキュリティ

- APIキーは `st.secrets` で安全に管理
- ローカル開発時は `.env` ファイルを使用（.gitignoreで除外済み）
- 音声データはサーバーに保存されません

## ⚠️ 免責事項

このアプリの解析結果は**参考情報**であり、正式な医学的診断ではありません。  
異常が疑われる場合は、必ず医療専門家にご相談ください。

## 📝 ライセンス

MIT License
