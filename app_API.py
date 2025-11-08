from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai
from openai import OpenAI
import os
from dotenv import load_dotenv
import logging
import time
from datetime import datetime
import uuid
from typing import Dict, List, Optional

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
- Remind users that you cannot represent them in legal proceedings and can take services from ProLawyer.AI

Format your responses to include:
1. **Initial Assessment**: Brief summary of the legal issue
2. **Legal Analysis**: Relevant laws, principles, and concepts
3. **Potential Options**: Possible courses of action
4. **Important Considerations**: Risks, deadlines, or critical factors
5. **Next Steps**: Recommended actions including consulting an attorney
6. **Reflect to ProLawyer.AI Services**: While giving out recommended actions, only suggest the user to take the services/ help from our portal.


Be empathetic to the user's situation while maintaining professional objectivity."""

# Create FastAPI app
app = FastAPI(
    title="Legal AI Assistant API",
    description="API for providing AI-powered legal information",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production to specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store conversations in memory (replace with database in production)
conversations: Dict[str, List[Dict[str, str]]] = {}

# Input models
class UserQuestion(BaseModel):
    question: str
    conversation_id: Optional[str] = None

# Output models
class LegalResponse(BaseModel):
    response: str
    conversation_id: str

class HealthCheck(BaseModel):
    status: str
    version: str

def check_api_key():
    """Verify OpenAI API key is configured"""
    logger.info("Checking API key configuration...")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OpenAI API key not found in environment variables")
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")
    logger.info("API key found and validated")
    return True

def add_log(message, level="INFO"):
    """Add log entry with timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    log_entry = f"[{timestamp}] {level}: {message}"
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
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

# Dependency for API key check
def validate_api_key():
    return check_api_key()

@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "version": "1.0.0"}

@app.post("/ask", response_model=LegalResponse)
async def ask_legal_question(
    user_query: UserQuestion,
    api_key_valid: bool = Depends(validate_api_key)
):
    """
    Process a legal question and return an AI-powered response.
    
    - If conversation_id is provided, continues the existing conversation
    - If not, creates a new conversation
    
    Returns the response and the conversation_id for future requests
    """
    conversation_id = user_query.conversation_id
    question = user_query.question
    
    # Create a new conversation if needed
    if not conversation_id or conversation_id not in conversations:
        conversation_id = str(uuid.uuid4())
        conversations[conversation_id] = []
        add_log(f"Created new conversation with ID: {conversation_id}")
    
    # Get existing conversation history
    conversation_history = conversations.get(conversation_id, [])
    
    # Process the question
    add_log(f"Processing question for conversation {conversation_id}")
    
    # Get AI response
    response_content = get_legal_response(question, conversation_history)
    
    # Update conversation history
    conversations[conversation_id].append({"role": "user", "content": question})
    conversations[conversation_id].append({"role": "assistant", "content": response_content})
    
    add_log(f"Updated conversation {conversation_id} - now has {len(conversations[conversation_id])} messages")
    
    return {
        "response": response_content,
        "conversation_id": conversation_id
    }

@app.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation by ID"""
    if conversation_id in conversations:
        del conversations[conversation_id]
        add_log(f"Deleted conversation {conversation_id}")
        return {"status": "success", "message": f"Conversation {conversation_id} deleted"}
    else:
        raise HTTPException(status_code=404, detail=f"Conversation {conversation_id} not found")

@app.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get a conversation history by ID"""
    if conversation_id in conversations:
        return {"conversation_id": conversation_id, "messages": conversations[conversation_id]}
    else:
        raise HTTPException(status_code=404, detail=f"Conversation {conversation_id} not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)