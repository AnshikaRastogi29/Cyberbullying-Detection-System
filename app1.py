import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from model import train_models, analyze

st.set_page_config(
    page_title="Cyberbullying Detection - Week 3 Progress",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

@st.cache_resource
def load():
    return train_models()

with st.spinner("🔄 Training ML Models..."):
    art = load()

accuracies = art['accuracies']
reports = art['reports']
le = art['le']
df = art['df']
cm = art['cm']

with st.sidebar:
    st.markdown("## 🛡️ CyberGuard AI")
    
    st.markdown("---")
    page = st.radio("📌 Navigation", [
        "🏠 Detection Demo",
        "📊 Model Performance",
        "📈 Data Insights"
    ])
    st.markdown("---")
    st.markdown("**Dataset:** 47,692 Tweets")
    st.markdown(f"**Best Accuracy:** {max(accuracies.values())}%")

# ══════════════════════════════════════════════════════
if page == "🏠 Detection Demo":
    st.markdown("""
    <div class="hero">
        <h1>🛡️ Cyberbullying Detection System</h1>
        <p>Multi-Class Detection with Sentiment & Emotion Analysis</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="stat-card"><h2>47K+</h2><p>Tweets Analyzed</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="stat-card"><h2>{max(accuracies.values())}%</h2><p>Best Accuracy</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="stat-card"><h2>6</h2><p>Categories</p></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-title">📝 Enter a Comment</div>', unsafe_allow_html=True)

    if "tweet_text" not in st.session_state:
        st.session_state.tweet_text = ""

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("✅ Normal Example", use_container_width=True):
            st.session_state.tweet_text = "Have a wonderful day! Stay kind and positive!"
    with c2:
        if st.button("🚨 Bullying Example", use_container_width=True):
            st.session_state.tweet_text = "You are so stupid and ugly, nobody likes you"
    with c3:
        if st.button("🕌 Religion Example", use_container_width=True):
            st.session_state.tweet_text = "Go back to your country, your religion doesn't belong here"

    tweet = st.text_area("Comment:", height=120, key="tweet_text")

    if st.button("🔍 ANALYZE", use_container_width=True, type="primary"):
        if tweet.strip() == "":
            st.warning("⚠️ Please enter a comment!")
        else:
            result = analyze(tweet, art)
            pred = result['prediction']

            st.markdown("---")
            st.markdown('<div class="section-title">🚨 Detection Result</div>', unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                if pred == 'not_cyberbullying':
                    st.markdown('<div class="result-safe">✅ SAFE COMMENT</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="result-danger">🚨 {pred.replace("_"," ").upper()}</div>', unsafe_allow_html=True)
            with col2:
                st.metric("Confidence Score", f"{result['confidence']}%")
                st.metric("Category", pred.replace('_', ' ').title())

            st.markdown('<div class="section-title">💬 Sentiment Analysis</div>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Sentiment", result['sentiment'])
            with col2:
                st.metric("Polarity Score", result['polarity'])

            st.markdown('<div class="section-title">😡 Emotion Detection</div>', unsafe_allow_html=True)
            st.metric("Dominant Emotion", result['emotion'])

# ══════════════════════════════════════════════════════
elif page == "📊 Model Performance":
    st.markdown('<div class="section-title">📊 Model Comparison</div>', unsafe_allow_html=True)

    cols = st.columns(len(accuracies))
    for col, (name, acc) in zip(cols, accuracies.items()):
        with col:
            st.markdown(f'<div class="stat-card"><h2>{acc}%</h2><p>{name}</p></div>', unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(accuracies.keys(), accuracies.values(), color=['#667eea','#764ba2','#0f3460'])
    ax.set_ylabel('Accuracy %')
    st.pyplot(fig)

    st.markdown('<div class="section-title">🔢 Confusion Matrix</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

# ══════════════════════════════════════════════════════
elif page == "📈 Data Insights":
    st.markdown('<div class="section-title">📈 Dataset Overview</div>', unsafe_allow_html=True)

    counts = df['cyberbullying_type'].value_counts()
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.pie(counts.values, labels=counts.index, autopct='%1.1f%%')
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        counts.plot(kind='bar', ax=ax)
        st.pyplot(fig)

    st.dataframe(df[['tweet_text','cyberbullying_type']].sample(10), use_container_width=True)