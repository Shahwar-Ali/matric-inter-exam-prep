import streamlit as st
import json
import time
from pypdf import PdfReader
from google import genai
from google.genai import types
from google.genai.errors import ServerError, APIError

# --- Page Configuration ---
st.set_page_config(
    page_title="Pak Exam Prep - Matric & Inter",
    page_icon="📖",
    layout="wide"
)

MODEL_NAME = "gemini-3.5-flash-lite"

# --- JSON String Cleanup Helper ---
def clean_json_string(raw_text: str) -> str:
    """Safely strips markdown code blocks and whitespace from JSON strings."""
    text = raw_text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()

# --- Resilient API Call Helper ---
def safe_generate_content(client, model, contents, config, retries=3):
    """Retries on momentary Google server or network bottlenecks."""
    for attempt in range(retries):
        try:
            return client.models.generate_content(
                model=model,
                contents=contents,
                config=config
            )
        except (ServerError, APIError) as e:
            if attempt == retries - 1:
                raise e
            time.sleep(3)

# --- PDF Text Extraction ---
def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    extracted = []
    for page_idx, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            extracted.append(f"--- Page {page_idx + 1} ---\n{text}")
    return "\n".join(extracted)

# --- API Key Detection ---
api_key = st.secrets.get("GEMINI_API_KEY", "")

# --- Sidebar Controls ---
with st.sidebar:
    st.title("🇵🇰 Exam Preferences")
    if not api_key:
        api_key = st.text_input("Gemini API Key:", type="password")
    
    academic_level = st.selectbox(
        "Academic Level:",
        ["Matric (9th & 10th Class)", "Intermediate (11th & 12th / FSc / ICS / ICom)"]
    )
    
    board_system = st.selectbox(
        "Board Pattern:",
        ["Punjab Boards (BISE Lahore, Gujranwala, Rawalpindi, etc.)", 
         "Federal Board (FBISE - SLO / Conceptual)", 
         "Sindh Boards (BIEK, BSEK)", 
         "KPK / Balochistan Boards"]
    )
    
    explanation_lang = st.radio(
        "Explanation Language:",
        ["English", "Urdu (اردو)", "Roman Urdu (Hamaray alfaaz)"]
    )
    
    st.markdown("---")
    if st.button("🔄 Start New Chapter / Reset", use_container_width=True):
        st.session_state.clear()
        st.rerun()

# --- Initialize State ---
if "chapter_text" not in st.session_state:
    st.session_state.chapter_text = ""
if "quiz_data" not in st.session_state:
    st.session_state.quiz_data = None
if "evaluation_result" not in st.session_state:
    st.session_state.evaluation_result = None

st.title("📚 Matric & Inter Textbook Exam Prep Coach")
st.caption("Upload any textbook chapter PDF or paste notes to generate board-pattern questions and receive strict grading.")

# --- Step 1: Chapter Input ---
if not st.session_state.chapter_text:
    st.subheader("1️⃣ Provide Your Textbook Chapter")
    upload_tab, paste_tab = st.tabs(["📄 Upload Textbook Chapter (PDF)", "📝 Paste Text / Notes"])
    
    with upload_tab:
        uploaded_pdf = st.file_uploader("Upload Chapter PDF (e.g., Punjab Curriculum, Sindh Text, or National Book Foundation)", type=["pdf"])
        if uploaded_pdf and st.button("Process PDF Chapter", type="primary"):
            with st.spinner("Extracting text from textbook pages..."):
                extracted = extract_text_from_pdf(uploaded_pdf)
                if not extracted.strip():
                    st.error("No readable text found. Please ensure the PDF is not a low-res image scan.")
                else:
                    st.session_state.chapter_text = extracted[:35000]
                    st.success("Chapter text extracted successfully!")
                    st.rerun()
                    
    with paste_tab:
        pasted_text = st.text_area("Paste chapter text or topic notes directly:", height=250)
        if pasted_text and st.button("Use Pasted Text", type="primary"):
            st.session_state.chapter_text = pasted_text.strip()
            st.rerun()

# --- Step 2: Question Generation ---
if st.session_state.chapter_text and st.session_state.quiz_data is None:
    st.info("✅ Chapter loaded! Select your question distribution and generate the test.")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        num_mcqs = st.slider("Multiple Choice Questions (MCQs)", 2, 8, 4)
    with c2:
        num_short = st.slider("Short Conceptual Questions", 1, 5, 2)
    with c3:
        num_long = st.slider("Long / Comprehensive Questions", 0, 2, 1)

    if st.button("🎯 Generate Practice Exam", type="primary", use_container_width=True):
        if not api_key:
            st.error("Please supply a valid Gemini API Key in the sidebar or Secrets.")
        else:
            client = genai.Client(api_key=api_key)
            with st.spinner("Analyzing textbook content and building board-style questions..."):
                prompt = f"""
You are a senior paper-setter and examiner for Pakistani Board Examinations ({board_system}) for {academic_level}.
Generate practice questions STRICTLY derived from the textbook material provided below.

TEXTBOOK CONTENT:
\"\"\"
{st.session_state.chapter_text}
\"\"\"

Produce:
- Exactly {num_mcqs} Multiple Choice Questions (MCQs) with 4 options each (A, B, C, D).
- Exactly {num_short} Short Answer Questions (testing key definitions, formulas, or short reasoning).
- Exactly {num_long} Long / Essay Questions (testing derivations, mechanisms, or multi-step concepts).

Format the output as a valid JSON object matching this schema:
{{
  "mcqs": [
    {{
      "id": 1,
      "question": "string",
      "options": ["A) string", "B) string", "C) string", "D) string"],
      "correct_option": "A",
      "textbook_ref": "brief sentence from chapter confirming this answer"
    }}
  ],
  "short_questions": [
    {{
      "id": 1,
      "question": "string",
      "ideal_key_points": ["point 1", "point 2"]
    }}
  ],
  "long_questions": [
    {{
      "id": 1,
      "question": "string",
      "rubric": "what must be explained to get full marks"
    }}
  ]
}}

Output ONLY the JSON object. Do not wrap in markdown or backticks.
"""
                try:
                    response = safe_generate_content(
                        client=client,
                        model=MODEL_NAME,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            temperature=0.3,
                            response_mime_type="application/json"
                        )
                    )
                    clean_res = clean_json_string(response.text)
                    st.session_state.quiz_data = json.loads(clean_res)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error parsing questions from textbook: {str(e)}")

# --- Step 3: Interactive Exam Sheet ---
if st.session_state.quiz_data and st.session_state.evaluation_result is None:
    quiz = st.session_state.quiz_data
    st.subheader("📝 Answer Your Exam Paper")
    st.caption("Answer directly against the textbook reference. When done, submit for checking.")
    
    user_answers = {"mcqs": {}, "short": {}, "long": {}}
    
    # 1. MCQs Section
    if quiz.get("mcqs"):
        st.markdown("### Part 1: Multiple Choice Questions (Section A)")
        for mcq in quiz["mcqs"]:
            user_answers["mcqs"][str(mcq["id"])] = st.radio(
                f"**Q{mcq['id']}:** {mcq['question']}",
                mcq["options"],
                index=None,
                key=f"mcq_{mcq['id']}"
            )
        st.divider()

    # 2. Short Questions Section
    if quiz.get("short_questions"):
        st.markdown("### Part 2: Short Answer Questions (Section B)")
        for sq in quiz["short_questions"]:
            user_answers["short"][str(sq["id"])] = st.text_area(
                f"**Q{sq['id']}:** {sq['question']}",
                placeholder="Write your concise answer based on the chapter...",
                key=f"sq_{sq['id']}"
            )
        st.divider()

    # 3. Long Questions Section
    if quiz.get("long_questions"):
        st.markdown("### Part 3: Long / Detailed Questions (Section C)")
        for lq in quiz["long_questions"]:
            user_answers["long"][str(lq["id"])] = st.text_area(
                f"**Q{lq['id']}:** {lq['question']}",
                placeholder="Provide detailed explanation, headings, and points...",
                height=150,
                key=f"lq_{lq['id']}"
            )

    if st.button("📤 Submit Paper for Evaluation", type="primary", use_container_width=True):
        client = genai.Client(api_key=api_key)
        with st.spinner("Checking answers strictly against textbook facts and grading in your preferred language..."):
            eval_prompt = f"""
You are an expert Pakistani Board examiner for {academic_level} ({board_system}).
Grade the student's submitted answers STRICTLY against the original chapter text provided below.

TEXTBOOK CONTENT:
\"\"\"
{st.session_state.chapter_text}
\"\"\"

QUESTIONS AND EXAM PAPER:
{json.dumps(quiz)}

STUDENT'S SUBMITTED ANSWERS:
{json.dumps(user_answers)}

Language for explanations and feedback: {explanation_lang}.
(If "Urdu (اردو)", write the explanation in fluent Urdu script. If "Roman Urdu", write in conversational Roman Urdu like "Aap ka jawab sahi hai kyunkay textbook page par wazeh likha hai...". If "English", use clear academic English.)

Grading Rules:
1. Verify MCQs against the correct options.
2. For Short & Long answers, verify if the candidate included the mandatory textbook definitions, formulas, or principles.
3. Deduct marks for factual inaccuracies or omissions according to board exam standards.

Output strictly valid JSON with this schema:
{{
  "total_marks_obtained": 0,
  "total_maximum_marks": 0,
  "percentage": 0,
  "general_verdict": "string in chosen language",
  "mcq_evaluations": [
    {{
      "id": 1,
      "is_correct": true,
      "student_choice": "string",
      "correct_choice": "string",
      "explanation": "string in chosen language"
    }}
  ],
  "subjective_evaluations": [
    {{
      "question": "string",
      "marks_awarded": 0,
      "max_marks": 0,
      "feedback": "string in chosen language",
      "textbook_reference": "exact excerpt or fact from the textbook chapter",
      "model_answer": "ideal answer in chosen language to score full marks"
    }}
  ]
}}
"""
            try:
                eval_res = safe_generate_content(
                    client=client,
                    model=MODEL_NAME,
                    contents=eval_prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.2,
                        response_mime_type="application/json"
                    )
                )
                clean_eval = clean_json_string(eval_res.text)
                st.session_state.evaluation_result = json.loads(clean_eval)
                st.rerun()
            except Exception as e:
                st.error(f"Error evaluating paper: {str(e)}")

# --- Step 4: Evaluation Results Screen ---
if st.session_state.evaluation_result is not None:
    res = st.session_state.evaluation_result
    st.success("🏁 Paper Graded Successfully!")
    
    # Score Metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Marks Obtained", f"{res.get('total_marks_obtained', 0)} / {res.get('total_maximum_marks', 0)}")
    m2.metric("Percentage", f"{res.get('percentage', 0)}%")
    m3.metric("Language", explanation_lang)
    
    st.info(f"**Examiner's Remarks:** {res.get('general_verdict', '')}")
    
    # MCQ Breakdown
    if res.get("mcq_evaluations"):
        st.markdown("### 🔘 MCQ Results")
        for item in res["mcq_evaluations"]:
            status = "✅ Correct" if item.get("is_correct") else "❌ Incorrect"
            with st.expander(f"MCQ #{item.get('id')} — {status}"):
                st.write(f"**Your Answer:** {item.get('student_choice', 'Unanswered')}")
                st.write(f"**Correct Answer:** {item.get('correct_choice', '')}")
                st.markdown(f"**Explanation:** {item.get('explanation', '')}")
                
    # Subjective Question Breakdown
    if res.get("subjective_evaluations"):
        st.markdown("### 📝 Short & Long Question Feedback")
        for subj in res["subjective_evaluations"]:
            with st.expander(f"Question: {subj.get('question')} (Marks: {subj.get('marks_awarded')}/{subj.get('max_marks')})"):
                st.markdown(f"**Examiner Feedback:** {subj.get('feedback')}")
                st.info(f"**📖 Textbook Reference Fact:** {subj.get('textbook_reference')}")
                st.markdown(f"**🌟 Model Board-Standard Answer:**\n{subj.get('model_answer')}")

    if st.button("⬅️ Practice Another Chapter"):
        st.session_state.clear()
        st.rerun()
