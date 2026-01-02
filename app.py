"""
透析シャント音チェッカー
FFT周波数解析とGemini AIによるハイブリッド解析アプリ
"""

import io
import os
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import scipy.io.wavfile as wavfile
import streamlit as st
from dotenv import load_dotenv

# 日本語フォント設定（文字化け対策）
# Windows/Mac/Linux対応
matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'Hiragino Sans', 'Yu Gothic', 'Meiryo', 'MS Gothic', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

# 環境変数の読み込み
load_dotenv()


def get_api_key() -> Optional[str]:
    """
    APIキーを取得する。
    優先順位: st.secrets > 環境変数 > session_state
    """
    # st.secretsから取得を試みる
    try:
        if "GOOGLE_API_KEY" in st.secrets:
            return st.secrets["GOOGLE_API_KEY"]
    except Exception:
        pass
    
    # 環境変数から取得
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key and api_key != "DUMMY_API_KEY_REPLACE_ME":
        return api_key
    
    # セッションステートから取得（ユーザー入力）
    if "user_api_key" in st.session_state and st.session_state.user_api_key:
        return st.session_state.user_api_key
    
    return None


def load_audio_data(audio_bytes: bytes) -> Tuple[int, np.ndarray]:
    """
    音声バイトデータからサンプルレートと波形データを取得する。
    """
    audio_io = io.BytesIO(audio_bytes)
    sample_rate, audio_data = wavfile.read(audio_io)
    
    # ステレオの場合はモノラルに変換
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)
    
    # float型に正規化
    if audio_data.dtype == np.int16:
        audio_data = audio_data.astype(np.float32) / 32768.0
    elif audio_data.dtype == np.int32:
        audio_data = audio_data.astype(np.float32) / 2147483648.0
    
    return sample_rate, audio_data


def perform_fft_analysis(sample_rate: int, audio_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    FFT解析を実行し、周波数とスペクトル強度を返す。
    """
    # FFTを実行
    n = len(audio_data)
    fft_result = np.fft.fft(audio_data)
    frequencies = np.fft.fftfreq(n, 1/sample_rate)
    
    # 正の周波数のみ取得
    positive_mask = frequencies >= 0
    frequencies = frequencies[positive_mask]
    magnitude = np.abs(fft_result[positive_mask])
    
    return frequencies, magnitude


def plot_spectrum(frequencies: np.ndarray, magnitude: np.ndarray) -> plt.Figure:
    """
    周波数スペクトルをプロットする。
    0-3000Hzの範囲を赤〜オレンジ系で表示。
    """
    # 3000Hz以下にフィルタリング
    mask = frequencies <= 3000
    freq_filtered = frequencies[mask]
    mag_filtered = magnitude[mask]
    
    # 正規化
    if mag_filtered.max() > 0:
        mag_normalized = mag_filtered / mag_filtered.max()
    else:
        mag_normalized = mag_filtered
    
    # プロット作成
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # グラデーションカラーで塗りつぶし
    ax.fill_between(freq_filtered, mag_normalized, alpha=0.4, color='#FF6B35')
    ax.plot(freq_filtered, mag_normalized, color='#E63946', linewidth=1.5)
    
    # 周波数帯域の目安を表示
    ax.axvline(x=500, color='#2A9D8F', linestyle='--', alpha=0.5, label='Normal (<500Hz)')
    ax.axvline(x=1000, color='#E9C46A', linestyle='--', alpha=0.5, label='Caution (1kHz)')
    ax.axvline(x=2000, color='#F4A261', linestyle='--', alpha=0.5, label='High Freq (2kHz)')
    
    ax.set_xlabel('Frequency (Hz)', fontsize=12)
    ax.set_ylabel('Amplitude (Normalized)', fontsize=12)
    ax.set_title('Shunt Sound Frequency Spectrum', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 3000)
    ax.set_ylim(0, 1.1)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 背景色を設定
    ax.set_facecolor('#1E1E1E')
    fig.patch.set_facecolor('#0E1117')
    
    # テキスト色を白に
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')
    
    plt.tight_layout()
    return fig


def plot_spectrogram(sample_rate: int, audio_data: np.ndarray) -> plt.Figure:
    """
    スペクトログラムをプロットする。
    連続音か断続音かを時間軸で視覚的に判断するため。
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # スペクトログラムを生成
    # NFFT: FFTの窓サイズ, noverlap: オーバーラップ量
    spectrum, freqs, times, im = ax.specgram(
        audio_data,
        Fs=sample_rate,
        NFFT=1024,
        noverlap=512,
        cmap='plasma',
        vmin=-80,  # dBの下限
        vmax=0     # dBの上限
    )
    
    # 3000Hz以下に制限
    ax.set_ylim(0, 3000)
    
    # カラーバー追加
    cbar = fig.colorbar(im, ax=ax, format='%+2.0f dB')
    cbar.set_label('Intensity (dB)', color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    
    ax.set_xlabel('Time (sec)', fontsize=12)
    ax.set_ylabel('Frequency (Hz)', fontsize=12)
    ax.set_title('Spectrogram - Continuous vs Intermittent Sound', fontsize=14, fontweight='bold')
    
    # 背景色を設定
    ax.set_facecolor('#1E1E1E')
    fig.patch.set_facecolor('#0E1117')
    
    # テキスト色を白に
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')
    
    plt.tight_layout()
    return fig

def analyze_with_gemini(audio_bytes: bytes, api_key: str) -> str:
    """
    Gemini 2.5 Flashで音声を解析する。
    """
    try:
        from google import genai
        from google.genai import types
        
        # クライアント初期化
        client = genai.Client(api_key=api_key)
        
        # プロンプト定義
        system_prompt = """あなたは熟練した透析専門医および臨床工学技士です。
患者が録音したシャント（ブラッドアクセス）の音声を聞いて、専門的な観点から評価を行ってください。

以下の4つの観点で必ず評価を出力してください：

## 1. 📊 音質の評価
録音は明瞭か、ノイズ（衣擦れ、環境音など）が多いかを評価してください。

## 2. 🔊 聞こえる音の特徴
「ゴーゴー（連続性雑音）」「ヒューヒュー（高調音）」「断続的」「拍動性」など、
聞こえる音の特徴を具体的に記述してください。

## 3. 🩺 推定判定
以下のいずれかを示してください：
- ✅ **正常範囲内**: 低音の連続性雑音が主体
- ⚠️ **狭窄の疑いあり**: 高音成分が目立つ、または音が細い
- 🚨 **閉塞の疑いあり**: 音が非常に弱い、または聞こえない
- ❓ **判定不能**: 録音品質が不十分、またはシャント音ではない可能性

## 4. 💡 アドバイス
患者が次にとるべき行動を具体的にアドバイスしてください。
例：「次回の透析時にスタッフに相談してください」「緊急性はありませんが経過観察を」など

---
**重要**: これは参考情報であり、正式な医学的診断ではありません。
必ず医療専門家に相談するよう促してください。"""

        user_prompt = "この音声はシャント（透析用ブラッドアクセス）から録音された音です。上記の観点で評価をお願いします。"
        
        # 音声データをパートとして準備
        audio_part = types.Part.from_bytes(
            data=audio_bytes,
            mime_type="audio/wav"
        )
        
        # コンテンツ生成
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        audio_part,
                        types.Part.from_text(text=user_prompt)
                    ]
                )
            ],
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.3,
            )
        )
        
        return response.text
        
    except Exception as e:
        return f"❌ **AI解析エラー**: {str(e)}\n\n※ APIキーが正しいか確認してください。"


def main():
    """
    メインアプリケーション
    """
    # ページ設定
    st.set_page_config(
        page_title="透析シャント音チェッカー",
        page_icon="🩺",
        layout="centered"
    )
    
    # カスタムCSS
    st.markdown("""
    <style>
    .main-title {
        text-align: center;
        color: #E63946;
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #888;
        font-size: 0.9rem;
        margin-bottom: 2rem;
    }
    .instruction-box {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #E63946;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .warning-box {
        background: #2d2d2d;
        border-left: 4px solid #E9C46A;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 5px 5px 0;
    }
    /* 録音ボタンを大きく表示 */
    [data-testid="stAudioInput"] > div {
        display: flex;
        justify-content: center;
    }
    [data-testid="stAudioInput"] button {
        width: 120px !important;
        height: 120px !important;
        border-radius: 50% !important;
        background: linear-gradient(135deg, #E63946 0%, #FF6B35 100%) !important;
        border: 4px solid #fff !important;
        box-shadow: 0 8px 20px rgba(230, 57, 70, 0.4) !important;
        transition: transform 0.2s, box-shadow 0.2s !important;
    }
    [data-testid="stAudioInput"] button:hover {
        transform: scale(1.05) !important;
        box-shadow: 0 12px 30px rgba(230, 57, 70, 0.6) !important;
    }
    [data-testid="stAudioInput"] button svg {
        width: 50px !important;
        height: 50px !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # タイトル
    st.markdown('<h1 class="main-title">透析シャント音チェッカー 🩺</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Powered by Gemini 2.5 Flash</p>', unsafe_allow_html=True)
    
    # APIキー取得（st.secretsから自動取得）
    api_key = get_api_key()
    
    # インストラクション
    st.markdown("""
    <div class="instruction-box">
        <h3>📋 録音方法</h3>
        <ol>
            <li>スマートフォンのマイクをシャント（腕の血管が太くなっている部分）に <strong>軽く当てて</strong> ください</li>
            <li>できるだけ <strong>静かな環境</strong> で録音してください</li>
            <li>下のボタンを押して <strong>5〜10秒程度</strong> 録音してください</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    # 音声入力
    st.subheader("🎙️ シャント音を録音")
    audio_data = st.audio_input("録音ボタンを押してください", key="audio_recorder")
    
    if audio_data is not None:
        # 録音データを表示
        st.audio(audio_data, format="audio/wav")
        
        # 解析ボタン
        if st.button("🔬 解析を開始", type="primary", use_container_width=True):
            audio_bytes = audio_data.getvalue()
            
            # 解析タブ
            tab1, tab2 = st.tabs(["📊 周波数解析 (FFT)", "🤖 AI診断 (Gemini)"])
            
            with tab1:
                st.subheader("周波数スペクトル分析")
                try:
                    with st.spinner("周波数解析中..."):
                        sample_rate, waveform = load_audio_data(audio_bytes)
                        frequencies, magnitude = perform_fft_analysis(sample_rate, waveform)
                        fig = plot_spectrum(frequencies, magnitude)
                        st.pyplot(fig)
                        plt.close(fig)
                        
                        # 簡易的な数値分析
                        # 低周波（0-500Hz）と高周波（1000-3000Hz）の比率
                        low_freq_mask = (frequencies >= 0) & (frequencies <= 500)
                        high_freq_mask = (frequencies >= 1000) & (frequencies <= 3000)
                        
                        low_power = magnitude[low_freq_mask].sum() if low_freq_mask.any() else 0
                        high_power = magnitude[high_freq_mask].sum() if high_freq_mask.any() else 0
                        
                        total_power = low_power + high_power
                        if total_power > 0:
                            low_ratio = (low_power / total_power) * 100
                            high_ratio = (high_power / total_power) * 100
                        else:
                            low_ratio = high_ratio = 0
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("低周波成分 (0-500Hz)", f"{low_ratio:.1f}%")
                        with col2:
                            st.metric("高周波成分 (1-3kHz)", f"{high_ratio:.1f}%")
                        
                        # 判定メッセージ
                        if low_ratio > 70:
                            st.success("✅ 低周波成分が優勢です（正常なシャント音の傾向）")
                        elif high_ratio > 40:
                            st.warning("⚠️ 高周波成分が多めです（狭窄の可能性を示唆）")
                        else:
                            st.info("ℹ️ 混合型のスペクトルです")
                        
                        # スペクトログラム表示
                        st.markdown("---")
                        st.subheader("スペクトログラム（時間-周波数解析）")
                        st.caption("💡 連続して色がついていれば「連続音」、途切れていれば「断続音」です")
                        
                        fig_spec = plot_spectrogram(sample_rate, waveform)
                        st.pyplot(fig_spec)
                        plt.close(fig_spec)
                            
                except Exception as e:
                    st.error(f"❌ FFT解析エラー: {str(e)}")
            
            with tab2:
                st.subheader("Gemini AI による評価")
                if not api_key:
                    st.warning("⚠️ AI診断を利用するにはAPIキーが必要です。")
                    st.info("サイドバーからGoogle API Keyを設定してください。")
                else:
                    with st.spinner("🤖 AIが音声を分析中..."):
                        result = analyze_with_gemini(audio_bytes, api_key)
                        st.markdown(result)
        
        # 免責事項
        st.markdown("""
        <div class="warning-box">
            <strong>⚠️ 重要な注意事項</strong><br>
            このアプリの解析結果は <strong>参考情報</strong> であり、正式な医学的診断ではありません。<br>
            異常が疑われる場合や心配な場合は、必ず <strong>医療専門家（透析スタッフ・医師）</strong> にご相談ください。
        </div>
        """, unsafe_allow_html=True)
    
    # フッター
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #666; font-size: 0.8rem;'>"
        "© 2026 Shunt Sound Analyzer | 医療機関での正式な検査をお勧めします"
        "</p>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
