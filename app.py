import platform
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import streamlit as st
from wordcloud import WordCloud

SYSTEM_MESSAGES = {"사진", "이모티콘", "동영상", "보이스톡", "삭제된 메시지입니다."}
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
TOKEN_PATTERN = re.compile(r"[가-힣A-Za-z0-9]+")


def get_korean_font_path() -> str:
    system = platform.system()
    if system == "Windows":
        candidates = [
            Path("C:/Windows/Fonts/malgun.ttf"),
            Path("C:/Windows/Fonts/malgunbd.ttf"),
        ]
    elif system == "Darwin":
        candidates = [
            Path("/System/Library/Fonts/Supplemental/AppleGothic.ttf"),
            Path("/Library/Fonts/AppleGothic.ttf"),
            Path("/System/Library/Fonts/AppleSDGothicNeo.ttc"),
        ]
    else:
        candidates = [
            Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),
            Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
        ]

    for path in candidates:
        if path.exists():
            return str(path)
    raise FileNotFoundError("한글 폰트를 찾을 수 없습니다.")


def is_excluded_message(message: str) -> bool:
    text = message.strip()
    if not text:
        return True
    if text in SYSTEM_MESSAGES:
        return True
    if any(keyword in text for keyword in ("보이스톡", "페이스톡")) and len(text) <= 20:
        return True
    if URL_PATTERN.fullmatch(text):
        return True
    if re.fullmatch(r"[\d\s:.-]+", text):
        return True
    return False


def extract_tokens(message: str) -> list[str]:
    cleaned = URL_PATTERN.sub(" ", message)
    tokens = TOKEN_PATTERN.findall(cleaned)
    return [
        token
        for token in tokens
        if len(token) >= 2
        and not token.isdigit()
        and token not in SYSTEM_MESSAGES
    ]


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    if seconds < 60:
        return f"{seconds}초"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}분 {secs}초" if secs else f"{minutes}분"
    hours, mins = divmod(minutes, 60)
    if hours < 24:
        return f"{hours}시간 {mins}분" if mins else f"{hours}시간"
    days, hrs = divmod(hours, 24)
    return f"{days}일 {hrs}시간" if hrs else f"{days}일"


def build_reply_records(chat_df: pd.DataFrame) -> pd.DataFrame:
    """화자가 바뀌는 지점의 답장 간격만 계산. 6시간 이상 공백은 새 대화로 표시."""
    ordered = chat_df.sort_values("Date").reset_index(drop=True)
    records = []
    for i in range(1, len(ordered)):
        prev = ordered.iloc[i - 1]
        curr = ordered.iloc[i]
        if prev["User"] == curr["User"]:
            continue

        gap_seconds = (curr["Date"] - prev["Date"]).total_seconds()
        if gap_seconds < 0:
            continue

        records.append(
            {
                "replier": curr["User"],
                "previous_user": prev["User"],
                "gap_seconds": gap_seconds,
                "reply_time": curr["Date"],
                "previous_time": prev["Date"],
                "previous_message": prev["Message"],
                "reply_message": curr["Message"],
                "is_new_conversation": gap_seconds >= 6 * 3600,
                "is_slow_reply": 3600 <= gap_seconds < 6 * 3600,
            }
        )
    return pd.DataFrame(records)


def apply_chart_layout(fig, *, height=None):
    layout = dict(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=60, b=40),
    )
    if height is not None:
        layout["height"] = height
    fig.update_layout(**layout)
    return fig


st.set_page_config(page_title="카카오톡 대화 분석기", layout="wide")

with st.sidebar:
    st.title("💬 카카오톡 대화 분석기")
    st.caption(
        "카카오톡에서 내보낸 CSV를 올리면 "
        "메시지·시간·단어·답장 속도를 한눈에 볼 수 있어요."
    )
    st.divider()
    uploaded_file = st.file_uploader("CSV 파일 업로드", type=["csv"])

df = None
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    required_columns = {"Date", "User", "Message"}
    if not required_columns.issubset(df.columns):
        with st.sidebar:
            st.error("Date, User, Message 칼럼이 필요합니다.")
        st.error("CSV에는 Date, User, Message 칼럼이 있어야 합니다.")
        st.stop()

    df["Message"] = df["Message"].fillna("").astype(str)
    df["char_count"] = df["Message"].str.len()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).copy()
    df["hour"] = df["Date"].dt.hour
    df["weekday"] = df["Date"].dt.dayofweek
    df["month"] = df["Date"].dt.to_period("M").astype(str)

    weekday_labels = ["월", "화", "수", "목", "금", "토", "일"]
    df["weekday_kr"] = df["weekday"].map(lambda x: weekday_labels[x])

    participants = sorted(df["User"].dropna().unique().tolist())
    date_min = df["Date"].min()
    date_max = df["Date"].max()

    with st.sidebar:
        st.divider()
        st.subheader("기본 정보")
        st.metric("메시지 수", f"{len(df):,}")
        st.markdown("**참여자**")
        for name in participants:
            st.write(f"- {name}")
        st.markdown("**기간**")
        st.write(
            f"{date_min.strftime('%Y-%m-%d')} ~ {date_max.strftime('%Y-%m-%d')}"
        )

with st.sidebar:
    st.divider()
    st.subheader("사용 방법")
    st.markdown(
        """
1. 카카오톡 대화방 → **채팅방 설정** → **대화 내용 내보내기**
2. CSV 파일을 사이드바에 업로드
3. 상단 탭에서 원하는 분석을 확인
4. `Date`, `User`, `Message` 칼럼이 있으면 다른 대화방도 분석 가능
        """
    )

if df is None:
    st.info("👈 사이드바에서 CSV 파일을 업로드하면 분석 결과가 여기에 표시됩니다.")
    st.stop()

user_stats = (
    df.groupby("User", as_index=False)
    .agg(
        message_count=("Message", "count"),
        total_chars=("char_count", "sum"),
        avg_length=("char_count", "mean"),
    )
    .sort_values("message_count", ascending=False)
)
user_stats["avg_length"] = user_stats["avg_length"].round(1)

color_map = px.colors.qualitative.Set2
user_colors = {
    user: color_map[i % len(color_map)] for i, user in enumerate(user_stats["User"])
}

tab_basic, tab_time, tab_words, tab_reply = st.tabs(
    ["📊 기본 통계", "⏰ 시간 분석", "🔤 단어 분석", "⚡ 답장 속도"]
)

# ----- 1. 기본 통계 -----
with tab_basic:
    st.markdown(
        """
        <style>
        div[data-testid="stMetric"] {
            background: linear-gradient(180deg, #ffffff 0%, #f7fafc 100%);
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            padding: 18px 20px;
            box-shadow: 0 8px 20px rgba(15, 23, 42, 0.04);
        }
        div[data-testid="stMetric"] label {
            color: #64748b !important;
            font-weight: 600 !important;
        }
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            color: #0f766e !important;
            font-size: 1.7rem !important;
        }
        .participant-card {
            border-radius: 18px;
            padding: 20px 22px;
            margin-bottom: 8px;
            border: 1px solid transparent;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
        }
        .participant-card h3 {
            margin: 0 0 14px 0;
            font-size: 1.15rem;
        }
        .participant-card .stat-row {
            display: flex;
            justify-content: space-between;
            margin: 8px 0;
            font-size: 0.95rem;
        }
        .participant-card .stat-label { color: #475569; }
        .participant-card .stat-value { font-weight: 700; color: #0f172a; }
        .dashboard-section-title {
            margin: 8px 0 4px 0;
            font-size: 1.05rem;
            font-weight: 700;
            color: #0f172a;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    total_messages = len(df)
    day_span = max((date_max.normalize() - date_min.normalize()).days + 1, 1)
    daily_avg = total_messages / day_span
    avg_length = df["char_count"].mean()
    active_days = df["Date"].dt.date.nunique()

    st.markdown(
        '<p class="dashboard-section-title">📌 한눈에 보는 대화 요약</p>',
        unsafe_allow_html=True,
    )
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("전체 메시지", f"{total_messages:,}")
    with m2:
        st.metric("대화 기간", f"{day_span:,}일")
    with m3:
        st.metric("일평균 메시지", f"{daily_avg:.1f}")
    with m4:
        st.metric("참여자 수", f"{len(participants):,}")

    m5, m6, m7, m8 = st.columns(4)
    with m5:
        st.metric("평균 메시지 길이", f"{avg_length:.1f}자")
    with m6:
        st.metric("총 글자수", f"{int(df['char_count'].sum()):,}")
    with m7:
        st.metric("대화가 있던 날", f"{active_days:,}일")
    with m8:
        st.metric(
            "기간",
            f"{date_min.strftime('%y.%m.%d')}~{date_max.strftime('%y.%m.%d')}",
        )

    st.markdown(
        '<p class="dashboard-section-title">👥 참여자별 카드</p>',
        unsafe_allow_html=True,
    )
    card_palette = [
        ("#ecfdf5", "#10b981", "#064e3b"),
        ("#fff7ed", "#f97316", "#7c2d12"),
        ("#eff6ff", "#3b82f6", "#1e3a8a"),
        ("#fdf2f8", "#ec4899", "#831843"),
        ("#f0fdfa", "#14b8a6", "#134e4a"),
    ]
    participant_cols = st.columns(min(len(participants), 3) or 1)
    total_for_share = max(total_messages, 1)

    for idx, row in enumerate(user_stats.itertuples(index=False)):
        bg, accent, title_color = card_palette[idx % len(card_palette)]
        share = row.message_count / total_for_share * 100
        with participant_cols[idx % len(participant_cols)]:
            st.markdown(
                f"""
                <div class="participant-card"
                     style="background:{bg}; border-color:{accent}33; border-left:6px solid {accent};">
                  <h3 style="color:{title_color};">{row.User}</h3>
                  <div class="stat-row">
                    <span class="stat-label">메시지</span>
                    <span class="stat-value">{row.message_count:,}개</span>
                  </div>
                  <div class="stat-row">
                    <span class="stat-label">비중</span>
                    <span class="stat-value">{share:.1f}%</span>
                  </div>
                  <div class="stat-row">
                    <span class="stat-label">총 글자수</span>
                    <span class="stat-value">{int(row.total_chars):,}자</span>
                  </div>
                  <div class="stat-row">
                    <span class="stat-label">평균 길이</span>
                    <span class="stat-value">{row.avg_length:.1f}자</span>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        '<p class="dashboard-section-title">📈 참여자 비교 차트</p>',
        unsafe_allow_html=True,
    )
    fig_count = px.bar(
        user_stats,
        x="User",
        y="message_count",
        color="User",
        color_discrete_map=user_colors,
        text="message_count",
        title="참여자별 메시지 개수",
        labels={"User": "참여자", "message_count": "메시지 개수"},
    )
    fig_count.update_traces(textposition="outside", texttemplate="%{text:,}")
    apply_chart_layout(fig_count)
    fig_count.update_layout(
        showlegend=False,
        yaxis=dict(gridcolor="rgba(0,0,0,0.08)", zeroline=False),
        xaxis=dict(title=""),
    )
    st.plotly_chart(fig_count, use_container_width=True)

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        fig_chars = px.bar(
            user_stats,
            x="User",
            y="total_chars",
            color="User",
            color_discrete_map=user_colors,
            text="total_chars",
            title="참여자별 총 글자수",
            labels={"User": "참여자", "total_chars": "총 글자수"},
        )
        fig_chars.update_traces(textposition="outside", texttemplate="%{text:,}")
        apply_chart_layout(fig_chars)
        fig_chars.update_layout(
            showlegend=False,
            yaxis=dict(gridcolor="rgba(0,0,0,0.08)", zeroline=False),
            xaxis=dict(title=""),
        )
        st.plotly_chart(fig_chars, use_container_width=True)

    with chart_col2:
        fig_avg = px.bar(
            user_stats,
            x="User",
            y="avg_length",
            color="User",
            color_discrete_map=user_colors,
            text="avg_length",
            title="참여자별 평균 메시지 길이",
            labels={"User": "참여자", "avg_length": "평균 글자수"},
        )
        fig_avg.update_traces(textposition="outside", texttemplate="%{text}")
        apply_chart_layout(fig_avg)
        fig_avg.update_layout(
            showlegend=False,
            yaxis=dict(gridcolor="rgba(0,0,0,0.08)", zeroline=False),
            xaxis=dict(title=""),
        )
        st.plotly_chart(fig_avg, use_container_width=True)

    longest = df.loc[df["char_count"].idxmax()]
    st.markdown(
        '<p class="dashboard-section-title">📝 가장 긴 메시지</p>',
        unsafe_allow_html=True,
    )
    st.success(
        f"**보낸 사람:** {longest['User']}  ·  **길이:** {longest['char_count']:,}자"
    )
    st.info(longest["Message"])

    with st.expander("데이터 미리보기 (처음 10행)"):
        st.dataframe(
            df[["Date", "User", "Message"]].head(10),
            use_container_width=True,
        )

# ----- 2. 시간 분석 -----
with tab_time:
    hour_counts = (
        df.groupby("hour", as_index=False)
        .size()
        .rename(columns={"size": "message_count"})
    )
    hour_counts = (
        pd.DataFrame({"hour": range(24)})
        .merge(hour_counts, on="hour", how="left")
        .fillna(0)
    )
    hour_counts["message_count"] = hour_counts["message_count"].astype(int)

    weekday_counts = (
        df.groupby(["weekday", "weekday_kr"], as_index=False)
        .size()
        .rename(columns={"size": "message_count"})
        .sort_values("weekday")
    )
    weekday_full = pd.DataFrame(
        {"weekday": range(7), "weekday_kr": weekday_labels}
    ).merge(
        weekday_counts[["weekday", "message_count"]],
        on="weekday",
        how="left",
    )
    weekday_full["message_count"] = weekday_full["message_count"].fillna(0).astype(int)

    month_counts = (
        df.groupby("month", as_index=False)
        .size()
        .rename(columns={"size": "message_count"})
        .sort_values("month")
    )

    peak_hour = int(hour_counts.loc[hour_counts["message_count"].idxmax(), "hour"])
    peak_hour_count = int(hour_counts["message_count"].max())
    peak_weekday_row = weekday_full.loc[weekday_full["message_count"].idxmax()]
    peak_weekday = peak_weekday_row["weekday_kr"]
    peak_weekday_count = int(peak_weekday_row["message_count"])

    peak_col1, peak_col2 = st.columns(2)
    with peak_col1:
        st.metric(
            "가장 활발한 시간대",
            f"{peak_hour}시",
            delta=f"{peak_hour_count:,}건",
            delta_color="off",
        )
    with peak_col2:
        st.metric(
            "가장 활발한 요일",
            f"{peak_weekday}요일",
            delta=f"{peak_weekday_count:,}건",
            delta_color="off",
        )

    fig_hour = px.bar(
        hour_counts,
        x="hour",
        y="message_count",
        text="message_count",
        title="시간대별 메시지 분포 (0시~23시)",
        labels={"hour": "시간", "message_count": "메시지 개수"},
        color="message_count",
        color_continuous_scale="Teal",
    )
    fig_hour.update_traces(textposition="outside", texttemplate="%{text:,}")
    apply_chart_layout(fig_hour)
    fig_hour.update_layout(
        yaxis=dict(gridcolor="rgba(0,0,0,0.08)", zeroline=False),
        xaxis=dict(
            title="시간",
            tickmode="linear",
            dtick=1,
            tickvals=list(range(24)),
            ticktext=[f"{h}시" for h in range(24)],
        ),
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig_hour, use_container_width=True)

    time_col1, time_col2 = st.columns(2)
    with time_col1:
        fig_weekday = px.bar(
            weekday_full,
            x="weekday_kr",
            y="message_count",
            text="message_count",
            title="요일별 메시지 분포",
            labels={"weekday_kr": "요일", "message_count": "메시지 개수"},
            color="message_count",
            color_continuous_scale="Purples",
            category_orders={"weekday_kr": weekday_labels},
        )
        fig_weekday.update_traces(textposition="outside", texttemplate="%{text:,}")
        apply_chart_layout(fig_weekday)
        fig_weekday.update_layout(
            yaxis=dict(gridcolor="rgba(0,0,0,0.08)", zeroline=False),
            xaxis=dict(title=""),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_weekday, use_container_width=True)

    with time_col2:
        fig_month = px.line(
            month_counts,
            x="month",
            y="message_count",
            markers=True,
            title="월별 메시지 추이",
            labels={"month": "월", "message_count": "메시지 개수"},
        )
        fig_month.update_traces(
            line=dict(width=3, color="#0D9488"),
            marker=dict(size=9, color="#0F766E"),
            hovertemplate="%{x}<br>메시지: %{y:,}<extra></extra>",
        )
        apply_chart_layout(fig_month)
        fig_month.update_layout(
            yaxis=dict(gridcolor="rgba(0,0,0,0.08)", zeroline=False),
            xaxis=dict(title=""),
        )
        st.plotly_chart(fig_month, use_container_width=True)

# ----- 3. 단어 분석 -----
with tab_words:
    word_df = df[~df["Message"].map(is_excluded_message)].copy()
    word_df["tokens"] = word_df["Message"].map(extract_tokens)
    all_tokens = [token for tokens in word_df["tokens"] for token in tokens]
    word_counter = Counter(all_tokens)
    top15 = word_counter.most_common(15)

    if top15:
        top15_df = pd.DataFrame(top15, columns=["word", "count"])
        fig_words = px.bar(
            top15_df,
            x="count",
            y="word",
            orientation="h",
            text="count",
            title="가장 많이 쓴 단어 TOP 15",
            labels={"word": "단어", "count": "횟수"},
            color="count",
            color_continuous_scale="Sunset",
        )
        fig_words.update_traces(textposition="outside", texttemplate="%{text:,}")
        apply_chart_layout(fig_words, height=560)
        fig_words.update_layout(
            yaxis=dict(categoryorder="total ascending", title=""),
            xaxis=dict(gridcolor="rgba(0,0,0,0.08)", zeroline=False),
            coloraxis_showscale=False,
            margin=dict(t=60, b=40, l=80),
        )
        st.plotly_chart(fig_words, use_container_width=True)
    else:
        st.warning("분석 가능한 단어가 없습니다.")

    st.markdown("#### 참여자별 자주 쓰는 단어 TOP 10")
    user_word_cols = st.columns(min(len(participants), 3) or 1)
    for idx, user in enumerate(participants):
        user_tokens = [
            token
            for tokens in word_df.loc[word_df["User"] == user, "tokens"]
            for token in tokens
        ]
        user_top = Counter(user_tokens).most_common(10)
        with user_word_cols[idx % len(user_word_cols)]:
            st.markdown(f"**{user}**")
            if user_top:
                user_top_df = pd.DataFrame(user_top, columns=["단어", "횟수"])
                fig_user_words = px.bar(
                    user_top_df,
                    x="횟수",
                    y="단어",
                    orientation="h",
                    text="횟수",
                    color="횟수",
                    color_continuous_scale="Tealgrn",
                )
                fig_user_words.update_traces(
                    textposition="outside", texttemplate="%{text:,}"
                )
                apply_chart_layout(fig_user_words, height=360)
                fig_user_words.update_layout(
                    showlegend=False,
                    yaxis=dict(categoryorder="total ascending", title=""),
                    xaxis=dict(gridcolor="rgba(0,0,0,0.08)", zeroline=False),
                    coloraxis_showscale=False,
                    margin=dict(t=20, b=20, l=60, r=20),
                )
                st.plotly_chart(fig_user_words, use_container_width=True)
            else:
                st.caption("분석 가능한 단어가 없습니다.")

    st.markdown("#### 워드클라우드")
    try:
        font_path = get_korean_font_path()
        if word_counter:
            wc = WordCloud(
                font_path=font_path,
                width=1000,
                height=500,
                background_color="white",
                colormap="viridis",
                max_words=150,
                prefer_horizontal=0.85,
            ).generate_from_frequencies(dict(word_counter))

            fig_wc, ax_wc = plt.subplots(figsize=(12, 6))
            ax_wc.imshow(wc, interpolation="bilinear")
            ax_wc.axis("off")
            st.pyplot(fig_wc, clear_figure=True)
            plt.close(fig_wc)
            st.caption(f"사용 폰트: `{font_path}`")
        else:
            st.warning("워드클라우드를 만들 단어가 없습니다.")
    except Exception as exc:
        st.error(f"워드클라우드 생성 중 오류: {exc}")

    st.markdown("#### ㅋ / ㅎ 포함 메시지 비교")
    laugh_stats = (
        df.assign(
            has_k=df["Message"].str.contains("ㅋ", na=False),
            has_h=df["Message"].str.contains("ㅎ", na=False),
        )
        .groupby("User", as_index=False)
        .agg(count_k=("has_k", "sum"), count_h=("has_h", "sum"))
        .rename(columns={"count_k": "ㅋ 포함", "count_h": "ㅎ 포함"})
    )
    laugh_long = laugh_stats.melt(
        id_vars="User",
        value_vars=["ㅋ 포함", "ㅎ 포함"],
        var_name="유형",
        value_name="메시지 수",
    )
    laugh_long["유형"] = laugh_long["유형"].map(
        {
            "ㅋ 포함": "ㅋ 포함 (ㅋㅋ, ㅋㅋㅋ 등)",
            "ㅎ 포함": "ㅎ 포함 (ㅎㅎ 등)",
        }
    )

    fig_laugh = px.bar(
        laugh_long,
        x="User",
        y="메시지 수",
        color="유형",
        barmode="group",
        text="메시지 수",
        title="참여자별 ㅋ / ㅎ 포함 메시지 횟수",
        color_discrete_sequence=["#F59E0B", "#6366F1"],
        labels={"User": "참여자"},
    )
    fig_laugh.update_traces(textposition="outside", texttemplate="%{text:,}")
    apply_chart_layout(fig_laugh)
    fig_laugh.update_layout(
        yaxis=dict(gridcolor="rgba(0,0,0,0.08)", zeroline=False),
        xaxis=dict(title=""),
        legend_title_text="",
    )
    st.plotly_chart(fig_laugh, use_container_width=True)
    st.dataframe(laugh_stats, use_container_width=True, hide_index=True)

# ----- 4. 답장 속도 -----
with tab_reply:
    st.caption(
        "같은 사람이 연속으로 보낸 메시지는 제외하고, "
        "6시간 이상 공백은 새 대화로 보고 평균/TOP5 계산에서 제외합니다. "
        "읽씹은 1시간 이상~6시간 미만 답장만 집계합니다."
    )

    reply_records = build_reply_records(df)
    if reply_records.empty:
        st.warning("답장 간격으로 분석할 대화가 없습니다.")
    else:
        valid_replies = reply_records[~reply_records["is_new_conversation"]].copy()

        if valid_replies.empty:
            st.warning("6시간 미만 답장 기록이 없어 평균 속도를 계산할 수 없습니다.")
        else:
            overall_avg = valid_replies["gap_seconds"].mean()
            reply_speed = (
                valid_replies.groupby("replier", as_index=False)
                .agg(
                    avg_seconds=("gap_seconds", "mean"),
                    reply_count=("gap_seconds", "count"),
                    median_seconds=("gap_seconds", "median"),
                )
                .sort_values("avg_seconds")
            )
            reply_speed["평균 답장 시간"] = reply_speed["avg_seconds"].map(
                format_duration
            )
            reply_speed["중앙값"] = reply_speed["median_seconds"].map(format_duration)

            st.metric("전체 평균 답장 시간", format_duration(overall_avg))

            st.markdown("#### 참여자별 평균 답장 속도")
            fig_reply = px.bar(
                reply_speed,
                x="replier",
                y="avg_seconds",
                color="replier",
                color_discrete_map=user_colors,
                text="평균 답장 시간",
                title="참여자별 평균 답장 속도 (초)",
                labels={"replier": "참여자", "avg_seconds": "평균 답장 시간(초)"},
            )
            fig_reply.update_traces(textposition="outside")
            apply_chart_layout(fig_reply)
            fig_reply.update_layout(
                showlegend=False,
                yaxis=dict(gridcolor="rgba(0,0,0,0.08)", zeroline=False),
                xaxis=dict(title=""),
            )
            st.plotly_chart(fig_reply, use_container_width=True)
            st.dataframe(
                reply_speed.rename(
                    columns={
                        "replier": "참여자",
                        "reply_count": "답장 횟수",
                    }
                )[["참여자", "답장 횟수", "평균 답장 시간", "중앙값"]],
                use_container_width=True,
                hide_index=True,
            )

            st.markdown("#### 가장 빠른 답장 TOP 5")
            fastest = valid_replies.nsmallest(5, "gap_seconds").copy()
            fastest["답장 속도"] = fastest["gap_seconds"].map(format_duration)
            fastest["상대 메시지"] = fastest["previous_message"].str.slice(0, 40)
            fastest["내 답장"] = fastest["reply_message"].str.slice(0, 40)
            st.dataframe(
                fastest.rename(
                    columns={
                        "replier": "답장한 사람",
                        "previous_user": "상대방",
                        "reply_time": "답장 시각",
                    }
                )[
                    [
                        "답장한 사람",
                        "상대방",
                        "답장 속도",
                        "답장 시각",
                        "상대 메시지",
                        "내 답장",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )

        st.markdown("#### 읽씹 카운트 (1시간 이상 답장)")
        slow_counts = (
            reply_records[reply_records["is_slow_reply"]]
            .groupby("replier", as_index=False)
            .size()
            .rename(columns={"replier": "참여자", "size": "읽씹 횟수"})
        )
        slow_counts = (
            pd.DataFrame({"참여자": participants})
            .merge(slow_counts, on="참여자", how="left")
            .fillna(0)
        )
        slow_counts["읽씹 횟수"] = slow_counts["읽씹 횟수"].astype(int)

        fig_slow = px.bar(
            slow_counts,
            x="참여자",
            y="읽씹 횟수",
            color="참여자",
            color_discrete_map=user_colors,
            text="읽씹 횟수",
            title="참여자별 읽씹 횟수 (1시간 이상 ~ 6시간 미만)",
        )
        fig_slow.update_traces(textposition="outside", texttemplate="%{text:,}")
        apply_chart_layout(fig_slow)
        fig_slow.update_layout(
            showlegend=False,
            yaxis=dict(gridcolor="rgba(0,0,0,0.08)", zeroline=False),
            xaxis=dict(title=""),
        )
        st.plotly_chart(fig_slow, use_container_width=True)
        st.dataframe(slow_counts, use_container_width=True, hide_index=True)
        st.metric("전체 읽씹 횟수", f"{int(slow_counts['읽씹 횟수'].sum()):,}")
