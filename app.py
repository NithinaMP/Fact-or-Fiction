import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime
import re
from transformers import pipeline
import torch

# Page Configuration
st.set_page_config(
    page_title="Fact-or-Fiction | AI News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize session state
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'classifier' not in st.session_state:
    st.session_state.classifier = None

# Load Hugging Face Model (cached)
@st.cache_resource
def load_model():
    """Load pre-trained fake news detection model from Hugging Face"""
    try:
        # Get Hugging Face token from secrets
        hf_token = st.secrets.get("HUGGINGFACE_TOKEN", None)
        
        # Using a popular fake news detection model
        model_name = "hamzab/roberta-fake-news-classification"
        
        classifier = pipeline(
            "text-classification",
            model=model_name,
            token=hf_token,
            device=0 if torch.cuda.is_available() else -1
        )
        return classifier, None
    except Exception as e:
        return None, str(e)

# Load model silently in background
if not st.session_state.model_loaded:
    classifier, error = load_model()
    if classifier:
        st.session_state.classifier = classifier
        st.session_state.model_loaded = True

# Title and Description
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    st.markdown("""
    <div style='text-align: center; padding: 20px 0;'>
        <h1 style='font-size: 48px; margin-bottom: 10px; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: bold;'>
            🔍 Fact-or-Fiction
        </h1>
        <p style='font-size: 20px; color: #888; margin-top: 0;'>
            AI-Powered News Credibility Analyzer
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Basic Pattern Analysis (Fallback)
def analyze_patterns(text):
    """Basic pattern-based analysis"""
    text_lower = text.lower()
    score = 50
    indicators = {'positive': [], 'negative': []}
    
    # Positive patterns
    positive = {
        r'according to|research shows|study found': ('Uses authoritative sources', 10),
        r'university|institute|journal|professor': ('References academic sources', 8),
        r'\d{4}|\d{1,2}/\d{1,2}/\d{2,4}': ('Contains specific dates', 5),
        r'however|although|despite': ('Shows balanced perspective', 7),
        r'reported by|published in': ('Cites publications', 6),
    }
    
    # Negative patterns
    negative = {
        r'shocking|unbelievable|miracle|secret': ('Sensational language', -15),
        r'!!!+': ('Excessive punctuation', -10),
        r'click here|share now|viral': ('Clickbait phrases', -12),
        r'conspiracy|cover-up': ('Conspiracy language', -15),
        r'100%|absolutely|definitely': ('Absolute claims', -10),
    }
    
    for pattern, (reason, weight) in positive.items():
        if re.search(pattern, text_lower):
            score += weight
            indicators['positive'].append(reason)
    
    for pattern, (reason, weight) in negative.items():
        if re.search(pattern, text_lower):
            score += weight
            indicators['negative'].append(reason)
    
    score = max(0, min(100, score))
    return score, indicators

# AI-Powered Analysis using Hugging Face
def analyze_with_ai(text):
    """Analyze text using Hugging Face model"""
    try:
        if st.session_state.classifier is None:
            return None
        
        # Get prediction from model
        result = st.session_state.classifier(text[:512])
        
        # Extract prediction
        label = result[0]['label'].upper()
        confidence = result[0]['score']
        
        # Convert to credibility score (0-100)
        if 'FAKE' in label or 'FALSE' in label:
            credibility_score = int((1 - confidence) * 100)
            category = "Likely Unreliable"
            color = "red"
            icon = "❌"
        else:
            credibility_score = int(confidence * 100)
            category = "Likely Reliable"
            color = "green"
            icon = "✅"
        
        if 40 <= credibility_score < 70:
            category = "Questionable - Verify"
            color = "orange"
            icon = "⚠️"
        
        pattern_score, indicators = analyze_patterns(text)
        
        return {
            'score': credibility_score,
            'category': category,
            'color': color,
            'icon': icon,
            'confidence': confidence,
            'ai_label': label,
            'indicators': indicators
        }
        
    except Exception as e:
        st.error(f"Analysis Error: {str(e)}")
        return None

# Generate Recommendations
def generate_recommendations(score, indicators):
    recommendations = []
    
    if score < 40:
        recommendations.append("🚨 **High Risk**: This content shows multiple red flags.")
        recommendations.append("📰 Cross-check with established news organizations.")
        recommendations.append("🔗 Search for the original source of information.")
    elif score < 70:
        recommendations.append("⚠️ **Moderate Risk**: This content requires verification.")
        recommendations.append("🔎 Verify from multiple credible sources.")
        recommendations.append("📅 Check dates, names, and specific claims.")
    else:
        recommendations.append("✅ **Lower Risk**: Shows signs of reliability.")
        recommendations.append("👍 Still recommended to verify from primary sources.")
        recommendations.append("📚 Check the reputation of the source.")
    
    return recommendations

# Educational Tips
def generate_tips():
    return [
        "🎯 **Check the Source**: Is it a reputable organization?",
        "📅 **Verify Dates**: Old stories are often recycled.",
        "🔗 **Look for Citations**: Reliable articles cite sources.",
        "📸 **Reverse Image Search**: Check for manipulated images.",
        "❓ **Question Bias**: Does it present multiple viewpoints?",
        "👥 **Consult Fact-checkers**: Use Snopes, FactCheck.org",
        "🔍 **Check About Page**: Legitimate sites have clear info.",
        "⚠️ **Be Skeptical**: If too good/bad to be true, verify it!"
    ]

# Main Input Section
st.markdown("### 📝 Analyze News Article or Social Media Content")

# Analysis mode selector - simplified
if st.session_state.model_loaded:
    analysis_mode = st.radio(
        "Analysis Mode:",
        ["🤖 AI-Powered Analysis", "⚡ Quick Pattern Check"],
        horizontal=True,
        label_visibility="collapsed"
    )
else:
    st.info("ℹ️ Running in Quick Pattern Check mode")
    analysis_mode = "⚡ Quick Pattern Check"

user_input = st.text_area(
    "Paste the text you want to analyze:",
    height=180,
    placeholder="Paste news article, headline, or social media post here...\n\nExample: 'BREAKING: Scientists discover miracle cure that doctors don't want you to know about!!!'"
)

# Example texts
col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    if st.button("📰 Try Fake News Example", use_container_width=True):
        st.session_state.example_text = "SHOCKING!!! Scientists discover miracle weight loss pill that doctors don't want you to know about! Click here to share this secret before it's taken down! 100% guaranteed results!!!"
        st.rerun()
with col2:
    if st.button("📄 Try Real News Example", use_container_width=True):
        st.session_state.example_text = "According to a study published in the Journal of Medicine, researchers at Harvard University found that regular exercise may reduce the risk of heart disease. However, experts caution that more research is needed to confirm these findings."
        st.rerun()

if 'example_text' in st.session_state:
    user_input = st.session_state.example_text
    del st.session_state.example_text

# Analyze Button
st.markdown("")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    analyze_button = st.button("🔍 Analyze Content", use_container_width=True, type="primary")

# Analysis Results
if analyze_button and user_input.strip():
    with st.spinner("🔄 Analyzing content..."):
        
        # Choose analysis method
        if "AI-Powered" in analysis_mode and st.session_state.model_loaded:
            result = analyze_with_ai(user_input)
            
            if result:
                score = result['score']
                category = result['category']
                color = result['color']
                icon = result['icon']
                confidence = result['confidence']
                ai_label = result['ai_label']
                indicators = result['indicators']
            else:
                score, indicators = analyze_patterns(user_input)
                if score >= 70:
                    category, color, icon = "Likely Reliable", "green", "✅"
                elif score >= 40:
                    category, color, icon = "Questionable - Verify", "orange", "⚠️"
                else:
                    category, color, icon = "Likely Unreliable", "red", "❌"
                confidence = None
                ai_label = None
        else:
            score, indicators = analyze_patterns(user_input)
            if score >= 70:
                category, color, icon = "Likely Reliable", "green", "✅"
            elif score >= 40:
                category, color, icon = "Questionable - Verify", "orange", "⚠️"
            else:
                category, color, icon = "Likely Unreliable", "red", "❌"
            confidence = None
            ai_label = None
        
        recommendations = generate_recommendations(score, indicators)
        tips = generate_tips()
        
        # Store in history
        st.session_state.analysis_history.append({
            'timestamp': datetime.now(),
            'score': score,
            'category': category,
            'text_preview': user_input[:100] + "..." if len(user_input) > 100 else user_input
        })
        
        # Display Results
        st.markdown("---")
        st.markdown("## 📊 Analysis Results")
        
        # Credibility Score Display
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(f"""
            <div style='text-align: center; padding: 40px; background: linear-gradient(135deg, rgba(255,255,255,0.1), rgba(255,255,255,0.05)); 
                        border-radius: 20px; border: 3px solid {color}; box-shadow: 0 8px 32px rgba(0,0,0,0.1); margin: 20px 0;'>
                <div style='font-size: 90px; margin: 0;'>{icon}</div>
                <h2 style='color: {color}; margin: 20px 0; font-weight: bold; font-size: 28px;'>{category}</h2>
                <div style='font-size: 80px; margin: 20px 0; font-weight: bold;'>{score}<span style='font-size: 40px; color: #888;'>/100</span></div>
                <p style='color: #888; font-size: 16px; margin: 5px 0;'>Credibility Score</p>
                {f"<p style='color: #888; font-size: 14px; margin-top: 15px;'>AI Confidence: {confidence*100:.1f}%</p>" if confidence else ""}
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("")
        
        # Indicators
        st.markdown("### 📋 Content Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### ✅ Positive Indicators")
            if indicators['positive']:
                for indicator in indicators['positive']:
                    st.markdown(f"- {indicator}")
            else:
                st.info("No positive indicators found")
        
        with col2:
            st.markdown("#### ❌ Suspicious Indicators")
            if indicators['negative']:
                for indicator in indicators['negative']:
                    st.markdown(f"- {indicator}")
            else:
                st.success("No suspicious indicators found")
        
        st.markdown("---")
        
        # Recommendations
        st.markdown("### 💡 Recommendations")
        for rec in recommendations:
            st.markdown(rec)
        
        st.markdown("---")
        
        # Educational Tips
        st.markdown("### 📚 How to Spot Fake News")
        cols = st.columns(2)
        for idx, tip in enumerate(tips):
            with cols[idx % 2]:
                st.markdown(tip)

elif analyze_button:
    st.warning("⚠️ Please enter some text to analyze.")

# Sidebar - Minimal and Clean
with st.sidebar:
    st.markdown("## 📊 Analysis History")
    
    if st.session_state.analysis_history:
        scores = [item['score'] for item in st.session_state.analysis_history[-10:]]
        
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(range(len(scores)), scores, marker='o', color='#667eea', linewidth=2, markersize=8)
        ax.fill_between(range(len(scores)), scores, alpha=0.2, color='#667eea')
        ax.set_ylabel('Credibility Score', fontsize=10)
        ax.set_xlabel('Analysis #', fontsize=10)
        ax.set_title('Credibility Trend', fontsize=12, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.axhline(y=70, color='g', linestyle='--', alpha=0.5, linewidth=2)
        ax.axhline(y=40, color='orange', linestyle='--', alpha=0.5, linewidth=2)
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("### Recent Analyses")
        for item in reversed(st.session_state.analysis_history[-5:]):
            with st.expander(f"{item['score']}/100 - {item['category']}"):
                st.write(f"**Time**: {item['timestamp'].strftime('%H:%M:%S')}")
                st.write(f"**Text**: {item['text_preview']}")
        
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.analysis_history = []
            st.rerun()
    else:
        st.info("No analyses yet. Try analyzing some content!")
    
    st.markdown("---")
    
    # Minimal About Section
    st.markdown("## ℹ️ About")
    st.markdown("""
    **Fact-or-Fiction** helps students identify potential misinformation using AI technology.
    
    **Quick Tips:**
    - Always verify from multiple sources
    - Check the publication date
    - Look for author credentials
    - Question sensational claims
    """)
    
    st.markdown("---")
    
    # Fact-Checking Resources
    st.markdown("## 🔗 Fact-Check Resources")
    st.markdown("""
    - [Snopes](https://www.snopes.com)
    - [FactCheck.org](https://www.factcheck.org)
    - [PolitiFact](https://www.politifact.com)
    - [Alt News](https://www.altnews.in)
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; padding: 20px;'>
    <p style='font-size: 14px;'><strong>Fact-or-Fiction</strong> | AI-Powered News Credibility Analyzer</p>
    <p style='font-size: 12px;'>Educational tool for promoting media literacy and critical thinking</p>
    <p style='font-size: 11px; margin-top: 10px;'>⚠️ This tool provides guidance only. Always verify from trusted sources.</p>
</div>
""", unsafe_allow_html=True)