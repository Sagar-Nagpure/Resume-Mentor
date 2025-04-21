import streamlit as st
import pdfplumber
import os
import joblib
import cohere
import re
import pandas as pd
import json
import plotly.graph_objects as go

# --- GET API KEY FROM STREAMLIT SECRETS ---
cohere_api_key = st.secrets["COHERE_API_KEY"]

# --- INIT COHERE CLIENT ---
co = cohere.Client(cohere_api_key)

# --- LOAD CLASSIFIER MODEL ---
try:
    classifier = joblib.load("resume_classifier.pkl")
except Exception as e:
    st.error(f"❌ Could not load classification model: {e}")
    st.stop()

# --- STREAMLIT CONFIG ---
st.set_page_config(page_title="Resume Mentor", layout="centered")
st.title("📄 Resume Mentor")
st.write("Upload your resume to get a predicted job domain, AI suggestions, ATS score, and extracted key skills.")

# --- SKILL EXTRACTOR ---
def extract_skills(resume_text):
    skills = [
        "Python", "Java", "JavaScript", "C", "C++", "C#", "Ruby", "PHP", "Swift", "Go", "R", "MATLAB", "SQL", "HTML", "CSS",
        "TypeScript", "Perl", "Scala", "Kotlin", "Lua", "Dart", "React", "Angular", "Vue", "Node.js", "Flask", "Django",
        "TensorFlow", "Keras", "PyTorch", "Machine Learning", "Deep Learning", "Data Analysis", "Communication", "Teamwork"
    ]
    extracted = [s for s in skills if re.search(r'\b' + re.escape(s) + r'\b', resume_text, re.IGNORECASE)]
    return list(set(extracted))

# --- SESSION STATE SETUP ---
if "second_resume_visible" not in st.session_state:
    st.session_state.second_resume_visible = False
if "second_resume_text" not in st.session_state:
    st.session_state.second_resume_text = ""
if "ats_output" not in st.session_state:
    st.session_state.ats_output = None
if "skills_output" not in st.session_state:
    st.session_state.skills_output = None
if "suggestions_output" not in st.session_state:
    st.session_state.suggestions_output = None
if "domain_output" not in st.session_state:
    st.session_state.domain_output = None
if "comparison_output" not in st.session_state:
    st.session_state.comparison_output = None

# --- RESUME UPLOADER ---
uploaded_file = st.file_uploader("📤 Upload your resume (PDF only)", type="pdf")

if uploaded_file:
    with st.spinner("🔍 Extracting text from resume..."):
        resume_text = ""
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                resume_text += page.extract_text() or ""

    if not resume_text.strip():
        st.error("❌ Could not extract text from PDF.")
        st.stop()

    st.subheader("📃 Extracted Resume Text")
    st.text_area("Resume Content", resume_text, height=300)

    st.markdown("### 🔍 Choose an Action")

    col1, col2 = st.columns(2)
    with col1:
        ats_btn = st.button("📊 Get ATS Score")
    with col2:
        skills_btn = st.button("🔑 Show Key Skills")

    col3, col4 = st.columns(2)
    with col3:
        suggestions_btn = st.button("💡 Get AI Suggestions")
    with col4:
        domain_btn = st.button("🧭 Predicted Job Domain")

    col5, _, _ = st.columns([2, 1, 1])
    with col5:
        compare_btn = st.button("⚖️ Compare with Another Resume")

    # --- BUTTON HANDLING LOGIC ---
    if ats_btn or skills_btn or suggestions_btn or domain_btn:
        st.session_state.second_resume_visible = False
        st.session_state.ats_output = None
        st.session_state.skills_output = None
        st.session_state.suggestions_output = None
        st.session_state.domain_output = None
        st.session_state.comparison_output = None

    if compare_btn:
        st.session_state.second_resume_visible = True
        st.session_state.ats_output = None
        st.session_state.skills_output = None
        st.session_state.suggestions_output = None
        st.session_state.domain_output = None

    # --- ATS SCORE ---
    if ats_btn:
        with st.spinner("📈 Analyzing ATS compatibility..."):
            ats_prompt = (
                "You're an ATS system. Score the following resume (out of 100), with breakdown (keywords, formatting, clarity, relevance), "
                "and give improvement suggestions:\n\n"
                f"{resume_text}"
            )
            try:
                response = co.generate(model="command", prompt=ats_prompt, max_tokens=500, temperature=0.5)
                ats_output = response.generations[0].text.strip()
                st.session_state.ats_output = ats_output
            except Exception as e:
                st.error(f"❌ ATS Error: {e}")

    # --- SHOW SKILLS ---
    if skills_btn:
        skills = extract_skills(resume_text)
        st.session_state.skills_output = ", ".join(skills) if skills else "No key skills detected."

    # --- AI SUGGESTIONS ---
    if suggestions_btn:
        with st.spinner("💬 Generating suggestions..."):
            suggestion_prompt = (
                "You are a resume expert. Give 5 clear suggestions to improve this resume (clarity, structure, relevance):\n\n"
                f"{resume_text}"
            )
            try:
                response = co.generate(model="command", prompt=suggestion_prompt, max_tokens=500, temperature=0.7)
                suggestions = response.generations[0].text.strip()
                st.session_state.suggestions_output = suggestions
            except Exception as e:
                st.error(f"❌ Suggestion Error: {e}")

    # --- PREDICT DOMAIN ---
    if domain_btn:
        with st.spinner("🧠 Predicting..."):
            try:
                probs = classifier.predict_proba([resume_text])[0]
                classes = classifier.classes_
                sorted_probs = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)
                st.session_state.domain_output = [(domain, prob*100) for domain, prob in sorted_probs[:5]]
            except Exception as e:
                st.warning(f"⚠️ Classification error: {e}")

    # --- RESUME COMPARISON ---
    if compare_btn:
        st.session_state.second_resume_visible = True

    # --- Resume Compare Mode UI ---
    if st.session_state.second_resume_visible:
        st.subheader("📥 Upload Second Resume for Comparison")
        second_file = st.file_uploader("Upload Second Resume (PDF)", type="pdf", key="second_resume_upload")

        if second_file:
            with st.spinner("📄 Extracting second resume..."):
                second_text = ""
                with pdfplumber.open(second_file) as pdf:
                    for page in pdf.pages:
                        second_text += page.extract_text() or ""

            if not second_text.strip():
                st.error("❌ Could not extract text from second resume.")
            else:
                st.session_state.second_resume_text = second_text

        if st.session_state.second_resume_text:
            col6, col7 = st.columns(2)
            with col6:
                text_compare_btn = st.button("📑 Textual Comparison")
            with col7:
                visual_compare_btn = st.button("📊 Compare via Visualization")

            # --- Textual Comparison Logic ---
            if text_compare_btn:
                with st.spinner("⚖️ Comparing resumes..."):
                    prompt = (
                        "You are an ATS. Compare Resume A and Resume B.\n"
                        "Provide a table with columns: Aspect, Resume A, Resume B.\n"
                        "Include: ATS Score, Keywords, Formatting, Clarity, Relevance.\n"
                        "Then, in 1 line, say which resume is better and why.\n\n"
                        f"Resume A:\n{resume_text}\n\nResume B:\n{st.session_state.second_resume_text}"
                    )
                    try:
                        response = co.generate(model="command", prompt=prompt, max_tokens=700, temperature=0.5)
                        output = response.generations[0].text.strip()
                        st.session_state.comparison_output = output
                    except Exception as e:
                        st.error(f"❌ Comparison Error: {e}")

            # --- Visual Comparison Logic ---
            if visual_compare_btn:
                with st.spinner("📊 Creating bar chart comparison..."):
                    prompt = (
                        "You are an ATS. Compare Resume A and Resume B.\n"
                        "Return a JSON object like this:\n"
                        "{ \"Aspect\": [\"ATS Score\", \"Keywords\", \"Formatting\", \"Clarity\", \"Relevance\"], "
                        "\"Resume A\": [85, 78, 70, 90, 88], \"Resume B\": [82, 80, 75, 85, 90], "
                        "\"Verdict\": \"Resume B is slightly better due to better relevance.\" }\n\n"
                        f"Resume A:\n{resume_text}\n\nResume B:\n{st.session_state.second_resume_text}"
                    )

                    try:
                        response = co.generate(model="command", prompt=prompt, max_tokens=700, temperature=0.5)
                        raw_output = response.generations[0].text.strip()

                        json_match = re.search(r'\{.*\}', raw_output, re.DOTALL)
                        if not json_match:
                            st.error("⚠️ Could not parse comparison JSON.")
                        else:
                            comparison = json.loads(json_match.group())
                            df = pd.DataFrame({
                                'Aspect': comparison['Aspect'],
                                'Resume A': comparison['Resume A'],
                                'Resume B': comparison['Resume B']
                            })

                            fig = go.Figure(data=[ 
                                go.Bar(name='Resume A', x=df['Aspect'], y=df['Resume A'],
                                       marker_color='blue', text=df['Resume A'], textposition='inside'),
                                go.Bar(name='Resume B', x=df['Aspect'], y=df['Resume B'],
                                       marker_color='orange', text=df['Resume B'], textposition='inside')
                            ])

                            fig.update_layout(
                                barmode='group',
                                title="📊 Resume Comparison by Aspect",
                                yaxis=dict(title='Score (out of 100)', range=[0, 100])
                            )

                            st.plotly_chart(fig)

                    except Exception as e:
                        st.error(f"❌ Visualization Error: {e}")

    # --- Display Outputs ---
    if st.session_state.ats_output:
        st.subheader("📈 ATS Score & Feedback")
        st.markdown(f"```markdown\n{st.session_state.ats_output}\n```")

    if st.session_state.skills_output:
        st.subheader("🔑 Extracted Skills")
        st.write(st.session_state.skills_output)

    if st.session_state.suggestions_output:
        st.subheader("✅ Suggestions to Improve Your Resume")
        st.markdown(f"```markdown\n{st.session_state.suggestions_output}\n```")

    if st.session_state.domain_output:
        st.subheader("🧭 Predicted Job Domains (Top 5)")
        for domain, prob in st.session_state.domain_output:
            st.write(f"**{domain}** - {prob:.2f}%")

    if st.session_state.comparison_output:
        st.subheader("📑 Resume Comparison (Text View)")
        st.markdown(st.session_state.comparison_output)

else:
    st.info("📋 Please upload your resume to get started.")
