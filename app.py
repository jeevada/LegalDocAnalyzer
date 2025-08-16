# Import the necessary libraries
import os
import secrets
from flask import Flask, render_template, request, jsonify, session
from dotenv import load_dotenv
import PyPDF2
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
import uuid

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'txt'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Define the State structure
class DocumentState:
    def __init__(self):
        self.original_document = ""
        self.document_summary = ""
        self.history = []

# Setup the LLM
genai.configure(api_key=os.getenv("GOOGLE_API_KEY")) # type: ignore

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",  # or "gemini-1.5-pro" for stronger reasoning
    temperature=0.2
)

# Core Prompts
ANALYSIS_PROMPT_TEMPLATE = """
You are LexiMind, a next-generation AI legal analyst. Your mission is to transform a complex legal document into a structured, clear set of insights.

You MUST structure your entire response using the following special tags. Do NOT include any text outside of these tags.

1.  Start the plain-language summary with `###LEXIMIND_SUMMARY_START###` and end it with `###LEXIMIND_SUMMARY_END###`. This section should provide a concise, bullet-point overview of the document in everyday English(no legal jargon).
2.  Start the list of critical(important) clauses, obligations, or unusual terms that may need attention with `###LEXIMIND_CLAUSES_START###` and end it with `###LEXIMIND_CLAUSES_END###`. In this section, boldly point out important terms like penalties, auto-renewals, fees and restrictive clauses, etc. Use markdown for bolding. Include confidence levels.
3.  Start the list of risks and red flags with `###LEXIMIND_FLAGS_START###` and end it with `###LEXIMIND_FLAGS_END###`. In this section, identify risky clauses(risk to be aware of). Each one MUST start with the "🚩 **RED FLAG:**" prefix. Explain the risk clearly. Include confidence levels. If there are no red flags, simply write "No significant red flags were detected." inside the tags.

Example Output Structure:
###LEXIMIND_SUMMARY_START###
This is a summary of the document...
- First key point...
- Second key point...
###LEXIMIND_SUMMARY_END###
###LEXIMIND_CLAUSES_START###
- **Termination Clause:** The contract renews automatically... (Confidence: High)
- **Payment Terms:** Payments are due within 15 days... (Confidence: High)
###LEXIMIND_CLAUSES_END###
###LEXIMIND_FLAGS_START###
🚩 **RED FLAG:** The indemnity clause is one-sided... (Confidence: Medium | Legal Advice: Recommended)
###LEXIMIND_FLAGS_END###

Begin your analysis now for the following document:
---
{document}
---
"""

QA_PROMPT_TEMPLATE = """
You are LexiMind, currently in Interactive Q&A mode. 
Your SOLE and ONLY source of information is the legal document provided below.
DO NOT use any external knowledge.

**User's Question:** {question}

**Legal Document Text:**
---
{document}
---

**Instructions:**
1.  Answer the user's question based **exclusively** on the provided legal document.
2.  If the document contains the answer, provide a clear, direct, and actionable response in plain English. Quote or reference the specific clause number if possible.
3.  If the document does **not** contain the answer or is ambiguous, state that clearly. For example, "The document does not specify the rules for..."
4.  After your answer, provide a confidence level and recommend if consulting a lawyer is advisable, just as you did in the initial analysis.
5.  Keep the tone friendly and helpful.
"""

def allowed_file(filename):
    """Check if the file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(file_stream):
    """Extract text from PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(file_stream)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        raise Exception(f"Error reading PDF: {str(e)}")

def extract_text_from_txt(file_stream):
    """Extract text from TXT file"""
    try:
        text = file_stream.read().decode('utf-8')
        return text.strip()
    except Exception as e:
        raise Exception(f"Error reading text file: {str(e)}")

def get_document_state():
    """Get or create document state for the current session"""
    if 'document_state' not in session:
        session['document_state'] = {
            'original_filepath': None,
            'document_summary': '',
            'history': []
        }
    return session['document_state']



# MODIFICATION 2: Add a new parsing function
def parse_structured_analysis(raw_analysis):
    """Parses the AI's raw output with delimiters into a dictionary."""
    try:
        summary = raw_analysis.split("###LEXIMIND_SUMMARY_START###")[1].split("###LEXIMIND_SUMMARY_END###")[0].strip()
        clauses = raw_analysis.split("###LEXIMIND_CLAUSES_START###")[1].split("###LEXIMIND_CLAUSES_END###")[0].strip()
        flags = raw_analysis.split("###LEXIMIND_FLAGS_START###")[1].split("###LEXIMIND_FLAGS_END###")[0].strip()
        
        return {
            "summary": summary,
            "clauses": clauses,
            "flags": flags
        }
    except IndexError:
        # Fallback if the AI fails to follow the structure perfectly
        return {
            "summary": "The AI failed to generate a structured analysis. Please try again.",
            "clauses": "",
            "flags": ""
        }


def analyze_document(document_text):
    """Analyze the legal document"""
    try:
        prompt = ANALYSIS_PROMPT_TEMPLATE.format(document=document_text)
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content
    except Exception as e:
        return f"Error analyzing document: {str(e)}"

def answer_question(question, document_text):
    """Answer a specific question about the document"""
    try:
        prompt = QA_PROMPT_TEMPLATE.format(question=question, document=document_text)
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content
    except Exception as e:
        return f"Error answering question: {str(e)}"

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and analysis"""
    try:
        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check if file is allowed
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not supported. Please upload PDF or TXT files only.'}), 400
        
        # Extract text based on file type
        file_extension = file.filename.rsplit('.', 1)[1].lower() # type: ignore
        
        if file_extension == 'pdf':
            document_text = extract_text_from_pdf(file.stream)
        elif file_extension == 'txt':
            document_text = extract_text_from_txt(file.stream)
        else:
            return jsonify({'error': 'Unsupported file type'}), 400
        
        if not document_text or len(document_text.strip()) < 50:
            return jsonify({'error': 'Document appears to be empty or too short. Please check your file.'}), 400
        
        # Store document in session
        state = get_document_state()
        
        
        # Clean up old file if it exists
        if state.get('document_filepath') and os.path.exists(state['document_filepath']):
            os.remove(state['document_filepath'])

        # Save document text to a temporary file instead of session
        filename = f"{uuid.uuid4()}.txt"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(document_text)
        
        state['document_filepath'] = filepath
        
        # Analyze the document
        raw_analysis = analyze_document(document_text)
        structured_analysis = parse_structured_analysis(raw_analysis)
        state['document_summary'] = structured_analysis # Now stores a dictionary
        
        # Clear previous history
        state['history'] = []
        
        # Update session
        session['document_state'] = state
        session.modified = True
        
        return jsonify({
            'success': True,
            'analysis': structured_analysis,
            'filename': file.filename
        })
        
    except Exception as e:
        return jsonify({'error': f'Error processing file: {str(e)}'}), 500

@app.route('/analyze_text', methods=['POST'])
def analyze_text():
    """Analyze pasted text document"""
    try:
        data = request.get_json()
        document_text = data.get('document_text', '').strip()
        
        if not document_text:
            return jsonify({'error': 'No document text provided'}), 400
        
        if len(document_text) < 50:
            return jsonify({'error': 'Document text is too short. Please provide a more substantial document.'}), 400
        
        # Store document in session
        state = get_document_state()


        # Clean up old file if it exists
        if state.get('document_filepath') and os.path.exists(state['document_filepath']):
            os.remove(state['document_filepath'])

        # Save document text to a temporary file instead of session
        filename = f"{uuid.uuid4()}.txt"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(document_text)
            
        state['document_filepath'] = filepath
        
        # Analyze the document
        raw_analysis = analyze_document(document_text)
        structured_analysis = parse_structured_analysis(raw_analysis)
        state['document_summary'] = structured_analysis # Now stores a dictionary
        
        # Clear previous history
        state['history'] = []
        
        # Update session
        session['document_state'] = state
        session.modified = True
        
        return jsonify({
            'success': True,
            'analysis': structured_analysis # Send the dictionary
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/ask', methods=['POST'])
def ask_question():
    """Handle Q&A about the document"""
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({'error': 'No question provided'}), 400
        
        state = get_document_state()
        
        filepath = state.get('document_filepath')
        if not filepath or not os.path.exists(filepath):
            return jsonify({'error': 'No document has been analyzed yet. Please upload or paste a document first.'}), 400
        
        # Read the document from the file
        with open(filepath, 'r', encoding='utf-8') as f:
            document_text = f.read()

        answer = answer_question(question, document_text)
        
        # Add to history
        if 'history' not in state:
            state['history'] = []
        
        state['history'].append({
            'question': question,
            'answer': answer
        })
        
        # Update session
        session['document_state'] = state
        session.modified = True
        
        return jsonify({
            'success': True,
            'answer': answer,
            'history': state['history']
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_status')
def get_status():
    """Get current status and history"""
    try:
        state = get_document_state()
        
        has_document = bool(state.get('document_filepath') and os.path.exists(state['document_filepath']))
        return jsonify({
            'has_document': has_document,
            'has_analysis': bool(state.get('document_summary')),
            'history': state.get('history', []),
            'analysis': state.get('document_summary', '')
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/clear_session', methods=['POST'])
def clear_session():
    """Clear the current session"""
    try:
        # Also delete the temp file associated with the session
        state = get_document_state()
        filepath = state.get('document_filepath')
        if filepath and os.path.exists(filepath):
            os.remove(filepath)
        
        session.pop('document_state', None)
        return jsonify({'success': True, 'message': 'Session cleared successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/sample_document')
def get_sample_document():
    """Get a sample document for testing"""
    sample_document = """
    FREELANCE SERVICES AGREEMENT

    This Agreement is made on June 1, 2024, between "The Client" (Big Corp Inc.) and "The Contractor" (Jane Doe).

    1. SERVICES. The Contractor agrees to perform web development services, including front-end coding and back-end integration, as detailed in Exhibit A.

    2. TERM. The engagement shall commence on June 10, 2024, and will automatically renew on a monthly basis unless terminated by either party with 30 days' written notice.

    3. COMPENSATION. The Client will pay the Contractor a fixed fee of $5,000 per month, payable within 15 days of receipt of a monthly invoice. A late fee of 5% per month will be applied to any overdue balances.

    4. OWNERSHIP OF WORK PRODUCT. The Contractor agrees that any and all work product created under this agreement shall be the sole and exclusive property of The Client. The Contractor automatically assigns all rights to The Client upon creation.

    5. CONFIDENTIALITY. The Contractor shall not disclose any proprietary information of The Client during or after the term of this agreement. This obligation survives the termination of this agreement indefinitely.

    6. TERMINATION. The Client may terminate this agreement for any reason with 14 days' written notice. The Contractor may only terminate if The Client fails to pay undisputed invoices for more than 60 days.

    7. INDEMNIFICATION. The Contractor agrees to indemnify and hold harmless The Client from any and all claims, damages, or liabilities arising from the Contractor's work, including claims of copyright infringement.

    8. GOVERNING LAW. This agreement shall be governed by the laws of the State of Delaware, without regard to its conflict of law provisions. Any disputes shall be resolved exclusively in the state courts of Delaware.
    """
    
    return jsonify({'sample_document': sample_document.strip()})

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server errors"""
    return jsonify({'error': 'Internal server error. Please try again.'}), 500

if __name__ == '__main__':
    # For PythonAnywhere deployment, use debug=False in production
    app.run(debug=True)