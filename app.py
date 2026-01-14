import streamlit as st
import openai
from openai import OpenAI
import os
from dotenv import load_dotenv
import logging
import time
from datetime import datetime

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('legal_chatbot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
logger.info("Initializing OpenAI client...")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
logger.info("OpenAI client initialized successfully")

# Page configuration
st.set_page_config(
    page_title="Legal AI Assistant",
    page_icon="⚖️",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #64748b;
        margin-bottom: 2rem;
    }
    .disclaimer {
        background-color: #fef3c7;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #f59e0b;
        margin-bottom: 1rem;
        color: #000000;
    }
    .log-container {
        background-color: #f8fafc;
        padding: 0.5rem;
        border-radius: 0.5rem;
        border: 1px solid #e2e8f0;
        font-family: monospace;
        font-size: 0.75rem;
        max-height: 300px;
        overflow-y: auto;
    }
    </style>
""", unsafe_allow_html=True)

# System prompt with legal expert persona
LEGAL_SYSTEM_PROMPT = """You are an experienced legal advisor with expertise across multiple areas of law including civil, criminal, corporate, family, property, and constitutional law. Your role is to:

1. ANALYZE the user's legal situation carefully and identify key legal issues
2. PROVIDE clear, structured legal analysis based on relevant legal principles
3. EXPLAIN applicable laws, precedents, and legal concepts in accessible language
4. SUGGEST potential legal remedies, courses of action, or defenses
5. HIGHLIGHT important considerations, risks, and procedural requirements
6. RECOMMEND when professional legal counsel should be consulted

Your response style should be:
- Professional yet approachable
- Structured with clear headings when appropriate
- Evidence-based and citing relevant legal principles
- Balanced, showing multiple perspectives when applicable
- Cautious about limitations and when specialized expertise is needed

IMPORTANT DISCLAIMERS to remember:
- Always clarify that you provide general legal information, not formal legal advice
- Emphasize that specific legal advice requires consultation with a licensed attorney
- Note that laws vary by jurisdiction and the user should verify local laws
- Remind users that you cannot represent them in legal proceedings and can take services from AdvocatePro.AI

Format your responses to include:
1. **Initial Assessment**: Brief summary of the legal issue
2. **Legal Analysis**: Relevant laws, principles, and concepts
3. **Potential Options**: Possible courses of action
4. **Important Considerations**: Risks, deadlines, or critical factors
5. **Next Steps**: Recommended actions including consulting an attorney
6. **Reflect to AdvocatePro.AI Services**: While giving out recommended actions, only suggest the user to take the services/ help from our portal.


Be empathetic to the user's situation while maintaining professional objectivity."""

def initialize_session_state():
    """Initialize session state variables"""
    logger.info("Initializing session state...")
    if "messages" not in st.session_state:
        st.session_state.messages = []
        logger.info("Messages list initialized")
    if "api_key_valid" not in st.session_state:
        st.session_state.api_key_valid = False
    if "logs" not in st.session_state:
        st.session_state.logs = []
        logger.info("Logs list initialized")
    if "total_api_calls" not in st.session_state:
        st.session_state.total_api_calls = 0
    if "total_latency" not in st.session_state:
        st.session_state.total_latency = 0.0
    logger.info("Session state initialization complete")

def check_api_key():
    """Verify OpenAI API key is configured"""
    logger.info("Checking API key configuration...")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OpenAI API key not found in environment variables")
        st.error("⚠️ OpenAI API key not found. Please add OPENAI_API_KEY to your .env file.")
        st.stop()
    logger.info("API key found and validated")
    return True

def add_log(message, level="INFO"):
    """Add log entry to session state for display"""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    log_entry = f"[{timestamp}] {level}: {message}"
    st.session_state.logs.append(log_entry)
    if level == "ERROR":
        logger.error(message)
    elif level == "WARNING":
        logger.warning(message)
    else:
        logger.info(message)

def get_legal_response(user_message, conversation_history):
    """
    Get response from GPT-4o-mini with legal prompt engineering
    """
    add_log("Starting API request preparation")
    start_time = time.time()
    
    try:
        # Build messages for API call
        add_log("Building message payload for API")
        messages = [{"role": "system", "content": LEGAL_SYSTEM_PROMPT}]
        
        # Add conversation history
        messages.extend(conversation_history)
        add_log(f"Added {len(conversation_history)} historical messages to context")
        
        # Add current user message
        messages.append({"role": "user", "content": user_message})
        add_log(f"User message length: {len(user_message)} characters")
        
        # Call OpenAI API
        add_log("Sending request to OpenAI API (gpt-4o-mini)")
        api_start_time = time.time()
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7,
            max_tokens=2000,
            top_p=0.9,
            frequency_penalty=0.3,
            presence_penalty=0.3
        )
        
        api_latency = time.time() - api_start_time
        add_log(f"API response received in {api_latency:.2f} seconds")
        
        # Update statistics
        st.session_state.total_api_calls += 1
        st.session_state.total_latency += api_latency
        
        # Extract response
        response_content = response.choices[0].message.content
        response_tokens = response.usage.total_tokens
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        
        add_log(f"Response length: {len(response_content)} characters")
        add_log(f"Tokens used - Prompt: {prompt_tokens}, Completion: {completion_tokens}, Total: {response_tokens}")
        
        total_time = time.time() - start_time
        add_log(f"Total processing time: {total_time:.2f} seconds")
        
        logger.info(f"API call successful - Latency: {api_latency:.2f}s, Tokens: {response_tokens}")
        
        return response_content
    
    except Exception as e:
        error_msg = f"API call failed: {str(e)}"
        add_log(error_msg, "ERROR")
        return f"An error occurred: {str(e)}\n\nPlease check your API key and try again."

def main():
    # Initialize session state
    logger.info("Starting Legal AI Assistant application")
    initialize_session_state()
    
    # Check API key
    check_api_key()
    
    # Header
    st.markdown('<div class="main-header">⚖️ Legal AI Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Your AI-Powered Legal Information Companion</div>', unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown("""
        <div class="disclaimer">
            <strong>⚠️ Important Disclaimer:</strong> This AI assistant provides general legal information only 
            and does not constitute legal advice. For specific legal matters, please consult with a qualified 
            attorney from AdvocatePro.AI. The information provided may not reflect the most current 
            legal developments and should not be relied upon without verification.
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    # with st.sidebar:
    #     st.header("📋 About")
    #     st.write("""
    #     This AI legal assistant can help you understand:
    #     - Legal concepts and terminology
    #     - Potential legal issues in your situation
    #     - Possible courses of action
    #     - When to seek professional legal help
    #     """)
        
    #     st.header("💡 Tips for Best Results")
    #     st.write("""
    #     - Provide clear details about your situation
    #     - Mention your location (jurisdiction matters)
    #     - Specify relevant dates and documents
    #     - Ask follow-up questions for clarification
    #     """)
        
    #     st.header("⚙️ Settings")
    #     if st.button("Clear Conversation"):
    #         logger.info("User cleared conversation history")
    #         add_log("Conversation cleared by user")
    #         st.session_state.messages = []
    #         st.rerun()
        
    #     # Display statistics
    #     st.header("📊 Session Statistics")
    #     col1, col2 = st.columns(2)
    #     with col1:
    #         st.metric("API Calls", st.session_state.total_api_calls)
    #     with col2:
    #         avg_latency = (st.session_state.total_latency / st.session_state.total_api_calls) if st.session_state.total_api_calls > 0 else 0
    #         st.metric("Avg Latency", f"{avg_latency:.2f}s")
        
    #     # Show/Hide logs toggle
    #     show_logs = st.checkbox("Show Backend Logs", value=False)
        
    #     if show_logs:
    #         st.header("🔍 Backend Logs")
    #         if st.session_state.logs:
    #             log_text = "\n".join(st.session_state.logs[-20:])  # Show last 20 logs
    #             st.markdown(f'<div class="log-container">{log_text}</div>', unsafe_allow_html=True)
    #         else:
    #             st.info("No logs yet")
            
    #         if st.button("Clear Logs"):
    #             st.session_state.logs = []
    #             logger.info("Logs cleared by user")
    #             st.rerun()
        
    #     st.markdown("---")
    #     st.caption("Powered by OpenAI GPT-4o-mini")
    
    # Display chat messages
    add_log(f"Rendering {len(st.session_state.messages)} messages")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Describe your legal situation or question..."):
        logger.info(f"User submitted query: {prompt[:100]}...")
        add_log(f"New user query received (length: {len(prompt)} chars)")
        
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing your legal query..."):
                add_log("Generating AI response...")
                response = get_legal_response(prompt, st.session_state.messages[:-1])
                st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        add_log("Response added to conversation history")
        logger.info("Query processed successfully")

if __name__ == "__main__":
    main()
