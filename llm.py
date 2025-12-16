
"""
llm.py - LLM Chatbot with Fallback (Gemini → Groq → Local)
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

llm_models_cache = {}

def local_chatbot_initializer(model_name="TinyLlama/TinyLLama-1.1B-Chat-v1.0"):
    """
    Initialize a local chatbot model using Hugging Face Transformers.
    Args:
        model_name (str): The name of the model to load.
    Returns:
        pipeline: The loaded chatbot pipeline.
    """
    from transformers import pipeline
    from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace

    if model_name in llm_models_cache:
        return llm_models_cache[model_name]
    
    chatbot_pipeline = pipeline(
        "text-generation",
        model=model_name,
        device_map="auto",
        max_new_tokens=512,
        temperature=0.7,
    )

    llm = HuggingFacePipeline(pipeline=chatbot_pipeline)
    chat_model = ChatHuggingFace(llm=llm
                                 )
    llm_models_cache[model_name] = chat_model
    return chat_model


def api_gemini_initializer(model_name="gemini-1.5-flash"):
    """
    Initialize the Gemini API client.
    Args:
        api_key (str): The API key for authentication.
    Returns:
        GeminiClient: The initialized Gemini API client.
    """
    from langchain_google_genai import ChatGoogleGenerativeAI

    if model_name in llm_models_cache:
        return llm_models_cache[model_name]
    
    chat_model = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.7,
        max_tokens=512
    )

    llm_models_cache[model_name] = chat_model
    return chat_model

def api_groq_initializer(model_name="llama-3.3-70b-versatile"):
    """
    Initialize the Groq API client.
    Args:
        api_key (str): The API key for authentication.
    Returns:
        GroqClient: The initialized Groq API client.
    """
    from langchain_groq import ChatGroq

    if model_name in llm_models_cache:
        return llm_models_cache[model_name]
    
    chat_model = ChatGroq(
        model=model_name,
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.7,
        max_tokens=512
    )

    llm_models_cache[model_name] = chat_model
    return chat_model


def llm_initializer_with_fallback():
    """
    Initialize a chatbot model with fallback options.
    Tries: Gemini → Groq → Local TinyLlama
    Returns:
        chat_model: The initialized chatbot model.
    """
    # Try Gemini (fast, free tier available)
    if os.getenv("GOOGLE_API_KEY"):
        try:
            chat_model = api_gemini_initializer()
            print("✅ Using Gemini API")
            return chat_model
        except Exception as e:
            print(f"⚠️ Gemini failed: {e}")

    # Try Groq (very fast, free tier available)
    if os.getenv("GROQ_API_KEY"):
        try:
            chat_model = api_groq_initializer()
            print("✅ Using Groq API")
            return chat_model
        except Exception as e:
            print(f"⚠️ Groq failed: {e}")
    
    # Try local model (slow, but works offline)
    try:
        chat_model = local_chatbot_initializer()
        print("✅ Using Local TinyLlama")
        return chat_model
    except Exception as e:
        print(f"❌ Local model failed: {e}")

    return None


def chat_with_llm(user_message, chat_model=None):
    """
    Send a message to the chatbot and get a response.
    Args:
        user_message (str): The user's message.
        chat_model: The initialized chat model.
    Returns:
        str: The chatbot's response.
    """
    if chat_model is None:
        chat_model = llm_initializer_with_fallback()
    
    if chat_model is None:
        return "❌ No LLM available. Please check API keys or install local model."
    
    try:
        response = chat_model.invoke(user_message)
        return response.content
    except Exception as e:
        return f"❌ Error: {e}"


"""
#######################################################################################################
if __name__ == "__main__":
    import gradio as gr

    # Initialize model once at startup
    print("🔄 Initializing LLM...")
    llm_model = llm_initializer_with_fallback()

    def chatbot_interface(message, history):
        ""Gradio chatbot interface with conversation history""
        response = chat_with_llm(message, llm_model)
        return response

    # Gradio UI
    demo = gr.ChatInterface(
        fn=chatbot_interface,
        title="NeeverStopTalking LLM Chatbot",
        description="Ask me anything!",
        examples=[
            "Hello! How are you?",
            "What is Python?",
            "Explain machine learning in simple terms",
            "Merhaba, nasılsın?"
        ],
        theme="soft",
        retry_btn=None,
        undo_btn="🔙 Undo",
        clear_btn="🗑️ Clear",
    )

    demo.launch(server_name="0.0.0.0", server_port=7862, inbrowser=True)

"""