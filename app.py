import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime
import re
import requests
import json

# Page Configuration
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="wide"
)

# Initialize session state
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'previous_credibility' not in st.session_state:
    st.session_state.previous_credibility = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# API Configuration (Read from secrets or environment)
API_KEY = st.secrets.get("ANTHROPIC_API_KEY", "")

# Title and Description
st.title("🔍 Fake News Detector for Students")
st.markdown("---")
st.write("**AI-Powered Analysis of News Articles and Social Media Content**")
st.write("This tool uses advanced AI to analyze credibility, detect misinformation patterns, and provide detailed explanations.")

# Basic Credibility Analysis Function (Fallback without API)
def analyze_credibility_basic(text):
    """
    Basic analysis without API - keyword matching
    """
    text_lower = text.lower()
    credibility_score = 50
    
    indicators = {
        'positive': [],
        'negative': []
    }
    
    # POSITIVE INDICATORS
    positive_patterns = {
        r'according to|research shows|study found|study shows|experts say|data suggests|evidence indicates': 
            ('Uses authoritative sources', 10),
        r'university|institute|journal|professor|dr\.|ph\.d|researcher': 
            ('References academic/expert sources', 8),
        r'\d{4}|\d{1,2}/\d{1,2}/\d{2,4}': 
            ('Contains specific dates', 5),
        r'however|although|despite|while|on the other hand': 
            ('Shows balanced perspective', 7),
        r'reported by|published in|cited in|source:': 
            ('Mentions sources', 6),
        r'statistics show|data from|survey|poll': 
            ('Uses data/statistics', 8)
    }
    
    # NEGATIVE INDICATORS
    negative_patterns = {
        r'shocking|unbelievable|you won\'t believe|miracle|secret revealed': 
            ('Sensational language', -15),
        r'!!!+|!!': 
            ('Excessive punctuation', -10),
        r'click here|share now|must read|viral': 
            ('Clickbait phrases', -12),
        r'conspiracy|cover-up|wake up|they don\'t want you to know': 
            ('Conspiracy language', -15),
        r'100%|absolutely|definitely|always|never|guaranteed': 
            ('Absolute claims', -10),
        r'breaking:|urgent:|alert:': 
            ('Urgency manipulation', -8)
    }
    
    for pattern, (reason, weight) in positive_patterns.items():
        if re.search(pattern, text_lower):
            credibility_score += weight
            indicators['positive'].append(reason)
    
    for pattern, (reason, weight) in negative_patterns.items():
        if re.search(pattern, text_lower):
            credibility_score += weight
            indicators['negative'].append(reason)
    
    # Additional checks
    if len(re.findall(r'\b[A-Z]{3,}\b', text)) > 3:
        credibility_score -= 12
        indicators['negative'].append('Excessive capitalization')
    
    credibility_score = max(0, min(100, credibility_score))
    
    if credibility_score >= 70:
        category = "Likely Reliable"
        color = "green"
        icon = "✅"
    elif credibility_score >= 40:
        category = "Questionable - Verify"
        color = "orange"
        icon = "⚠️"
    else:
        category = "Likely Unreliable"
        color = "red"
        icon = "❌"
    
    return credibility_score, category, color, icon, indicators

# AI-Powered Analysis Function using API
def analyze_with_ai(text):
    """
    Advanced analysis using Claude API
    """
    if not API_KEY:
        st.warning("⚠️ API key not configured. Using basic analysis mode.")
        return None
    
    try:
        prompt = f"""You are a fake news detection expert. Analyze the following text for credibility and misinformation patterns.

Text to analyze:
\"\"\"{text}\"\"\"

Provide your analysis in the following JSON format:
{{
    "credibility_score": <0-100>,
    "category": "<Likely Reliable|Questionable - Verify|Likely Unreliable>",
    "positive_indicators": ["list of positive credibility indicators found"],
    "negative_indicators": ["list of suspicious or misleading patterns found"],
    "detailed_explanation": "<comprehensive explanation of your analysis>",
    "key_concerns": ["list of main concerns if any"],
    "recommendations": ["specific actionable recommendations for the reader"]
}}

Be thorough and explain your reasoning."""

        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "Content-Type": "application/json",
                "x-api-key": API_KEY,
                "anthropic-version": "2023-06-01"
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 2000,
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            content = result['content'][0]['text']
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                analysis = json.loads(json_match.group())
                return analysis
            else:
                st.error("Could not parse AI response")
                return None
        else:
            st.error(f"API Error: {response.status_code}")
            return None
            
    except Exception as e:
        st.error(f"Error connecting to AI: {str(e)}")
        return None

# Generate Recommendations
def generate_recommendations(score, indicators):
    recommendations = []
    
    if score < 40:
        recommendations.append("🔍 **High Risk**: Multiple red flags detected. Verify before sharing.")
        recommendations.append("📰 Cross-check with established news organizations.")
    elif score < 70:
        recommendations.append("⚠️ **Moderate Risk**: Requires verification.")
        recommendations.append("🔎 Check multiple credible sources.")
    else:
        recommendations.append("✅ **Lower Risk**: Shows signs of reliability.")
        recommendations.append("👍 Still verify from primary sources.")
    
    if len(indicators.get('negative', [])) > 5:
        recommendations.append("⚡ **Alert**: Multiple suspicious patterns detected.")
    
    return recommendations

# Generate Educational Tips
def generate_tips():
    return [
        "🎯 **Check the Source**: Is it from a reputable organization?",
        "📅 **Verify Dates**: Old stories are often recycled as current news.",
        "🔗 **Look for Citations**: Reliable articles cite their sources.",
        "📸 **Reverse Image Search**: Check if images are manipulated.",
        "❓ **Question Bias**: Does it present multiple viewpoints?",
        "👥 **Consult Fact-checkers**: Sites like Snopes, FactCheck.org"
    ]

# Main Input Section
st.markdown("### 📝 Enter News Article or Social Media Content")

# Analysis mode selector
analysis_mode = st.radio(
    "Select Analysis Mode:",
    ["🤖 AI-Powered Analysis (Advanced)", "⚡ Basic Pattern Matching (Fast)"],
    help="AI mode provides detailed explanation but requires API key"
)

user_input = st.text_area(
    "Paste the text you want to analyze:",
    height=200,
    placeholder="Paste news article, social media post, or any text content here..."
)

# Analyze Button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    analyze_button = st.button("🔍 Analyze Credibility", use_container_width=True)

# Analysis Results
if analyze_button and user_input.strip():
    with st.spinner("Analyzing content..."):
        
        # Choose analysis method
        if "AI-Powered" in analysis_mode and API_KEY:
            ai_result = analyze_with_ai(user_input)
            
            if ai_result:
                score = ai_result['credibility_score']
                category = ai_result['category']
                
                # Determine color and icon
                if score >= 70:
                    color, icon = "green", "✅"
                elif score >= 40:
                    color, icon = "orange", "⚠️"
                else:
                    color, icon = "red", "❌"
                
                indicators = {
                    'positive': ai_result.get('positive_indicators', []),
                    'negative': ai_result.get('negative_indicators', [])
                }
                
                explanation = ai_result.get('detailed_explanation', '')
                concerns = ai_result.get('key_concerns', [])
                ai_recommendations = ai_result.get('recommendations', [])
                
            else:
                # Fallback to basic
                score, category, color, icon, indicators = analyze_credibility_basic(user_input)
                explanation = None
                concerns = []
                ai_recommendations = []
        else:
            score, category, color, icon, indicators = analyze_credibility_basic(user_input)
            explanation = None
            concerns = []
            ai_recommendations = []
        
        recommendations = generate_recommendations(score, indicators)
        tips = generate_tips()
        
        # Store in history
        st.session_state.previous_credibility = score
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
            <div style='text-align: center; padding: 30px; background-color: rgba(255,255,255,0.05); 
                        border-radius: 15px; border: 2px solid {color}'>
                <h1 style='font-size: 72px; margin: 0;'>{icon}</h1>
                <h2 style='color: {color}; margin: 10px 0;'>{category}</h2>
                <h1 style='font-size: 64px; margin: 10px 0;'>{score}/100</h1>
                <p style='color: gray;'>Credibility Score</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("")
        
        # AI Explanation (if available)
        if explanation:
            st.markdown("### 🤖 AI Analysis Explanation")
            st.info(explanation)
            
            if concerns:
                st.markdown("### ⚠️ Key Concerns Identified")
                for concern in concerns:
                    st.markdown(f"- {concern}")
        
        # Indicators
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### ✅ Positive Indicators")
            if indicators['positive']:
                for indicator in indicators['positive']:
                    st.markdown(f"- {indicator}")
            else:
                st.info("No positive indicators found")
        
        with col2:
            st.markdown("### ❌ Suspicious Indicators")
            if indicators['negative']:
                for indicator in indicators['negative']:
                    st.markdown(f"- {indicator}")
            else:
                st.success("No suspicious indicators found")
        
        st.markdown("---")
        
        # Recommendations
        st.markdown("### 💡 Recommendations")
        if ai_recommendations:
            for rec in ai_recommendations:
                st.markdown(f"- {rec}")
        else:
            for rec in recommendations:
                st.markdown(rec)
        
        st.markdown("---")
        
        # Educational Tips
        st.markdown("### 📚 Tips for Spotting Fake News")
        for tip in tips:
            st.markdown(tip)

elif analyze_button:
    st.warning("⚠️ Please enter some text to analyze.")

# Sidebar
with st.sidebar:
    st.markdown("## 📖 About")
    st.info("""
    **Fake News Detector for Students**
    
    Analyzes content for:
    - Language patterns
    - Credibility indicators
    - Emotional manipulation
    - Source attribution
    - Sensationalism markers
    
    **Two Analysis Modes:**
    - 🤖 AI-Powered (requires API key)
    - ⚡ Basic Pattern Matching
    """)
    
    st.markdown("---")
    
    # API Status
    st.markdown("## 🔑 API Status")
    if API_KEY:
        st.success("✅ API Key Configured")
    else:
        st.warning("⚠️ No API Key - Using Basic Mode")
        with st.expander("How to add API Key"):
            st.markdown("""
            1. Create `.streamlit/secrets.toml`
            2. Add: `ANTHROPIC_API_KEY = "your-key-here"`
            3. Get key from: console.anthropic.com
            """)
    
    st.markdown("---")
    
    # Analysis History
    if st.session_state.analysis_history:
        st.markdown("## 📈 Analysis History")
        
        scores = [item['score'] for item in st.session_state.analysis_history[-10:]]
        
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(range(len(scores)), scores, marker='o', color='#1f77b4', linewidth=2)
        ax.set_ylabel('Credibility Score')
        ax.set_xlabel('Analysis Number')
        ax.set_title('Recent Analysis Trend')
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=70, color='g', linestyle='--', alpha=0.5, label='Reliable')
        ax.axhline(y=40, color='orange', linestyle='--', alpha=0.5, label='Questionable')
        ax.legend()
        st.pyplot(fig)
        
        st.markdown("### Recent Analyses")
        for item in reversed(st.session_state.analysis_history[-5:]):
            with st.expander(f"{item['category']} - {item['score']}/100"):
                st.write(f"**Time**: {item['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                st.write(f"**Preview**: {item['text_preview']}")
    
    st.markdown("---")
    
    # Verification Resources
    st.markdown("## 🔗 Fact-Checking Resources")
    st.markdown("""
    - [Snopes](https://www.snopes.com)
    - [FactCheck.org](https://www.factcheck.org)
    - [PolitiFact](https://www.politifact.com)
    - [Alt News (India)](https://www.altnews.in)
    - [Google Fact Check](https://toolbox.google.com/factcheck/)
    """)
    
    if st.button("🗑️ Clear History"):
        st.session_state.analysis_history = []
        st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <p><strong>Fake News Detector for Students</strong></p>
    <p>Educational tool for promoting media literacy and critical thinking</p>
    <p style='font-size: 12px;'>⚠️ Always verify from multiple trusted sources</p>
</div>
""", unsafe_allow_html=True)