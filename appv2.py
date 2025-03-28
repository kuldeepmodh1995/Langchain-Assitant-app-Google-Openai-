
import streamlit as st
from google.generativeai import GenerativeModel, configure
from google.api_core.exceptions import InvalidArgument
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def initialize_session_state():
    """Initialize all session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "conversation_ended" not in st.session_state:
        st.session_state.conversation_ended = False
    if "api_keys_provided" not in st.session_state:
        st.session_state.api_keys_provided = False

def display_chat_messages():
    """Display chat messages from history"""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def get_gemini_response(user_input, api_key):
    """Get response from Gemini model"""
    try:
        configure(api_key=api_key)
        model = GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(user_input)
        return response.text
    except InvalidArgument as e:
        st.error("Invalid Google API Key. Please check your key and try again.")
        st.session_state.api_keys_provided = False
        return None
    except Exception as e:
        st.error(f"Error getting response from Gemini: {str(e)}")
        return None

def generate_summary_and_sentiment(openai_key):
    """Generate summary and sentiment analysis using OpenAI"""
    if not st.session_state.messages:
        return "No conversation to summarize", "Neutral"
    
    conversation = "\n".join(
        f"{msg['role']}: {msg['content']}" 
        for msg in st.session_state.messages
    )
    
    summary_prompt = ChatPromptTemplate.from_template(
        """Summarize this conversation in under 150 words. 
        Also provide one-word sentiment (Positive/Negative/Neutral).
        
        Conversation:
        {conversation}
        
        Summary:"""
    )
    
    try:
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            api_key=openai_key
        )
        summary_chain = summary_prompt | llm | StrOutputParser()
        result = summary_chain.invoke({"conversation": conversation})
        
        if "Sentiment:" in result:
            summary, sentiment = result.split("Sentiment:")
            sentiment = sentiment.strip()
        else:
            summary = result
            sentiment = "Neutral"
        
        return summary.strip(), sentiment.strip()
    except Exception as e:
        st.error(f"Error generating summary: {str(e)}")
        return None, None

def api_key_input_form():
    """Form for collecting API keys with enhanced validation"""
    with st.form("api_key_form"):
        st.subheader("🔑 Enter API Keys")
        col1, col2 = st.columns(2)
        
        with col1:
            google_key = st.text_input(
                "Google AI Studio API Key:",
                type="password",
                help="Get your key from https://aistudio.google.com/app/apikey"
            )
        with col2:
            openai_key = st.text_input(
                "OpenAI API Key:",
                type="password",
                help="Get your key from https://platform.openai.com/api-keys"
            )
        
        submitted = st.form_submit_button("Start Chatting →", use_container_width=True)
        if submitted:
            if google_key.strip() and openai_key.strip():
                # Simple pattern validation
                if not google_key.startswith("AIza"):
                    st.error("Google API key should start with 'AIza'")
                    return
                if not openai_key.startswith("sk-"):
                    st.error("OpenAI API key should start with 'sk-'")
                    return
                
                # Test keys quickly
                with st.spinner("Verifying API keys..."):
                    try:
                        # Quick test of Google key
                        configure(api_key=google_key)
                        test_model = GenerativeModel('gemini-2.0-flash')
                        test_model.generate_content("Test", safety_settings={'HARM_CATEGORY_HARASSMENT':'block_none'})
                        
                        # Store keys in session state
                        st.session_state.google_api_key = google_key.strip()
                        st.session_state.openai_api_key = openai_key.strip()
                        st.session_state.api_keys_provided = True
                        st.rerun()
                    except Exception as e:
                        st.error(f"API key verification failed: {str(e)}")
            else:
                st.warning("Please provide both API keys")

def main():
    st.set_page_config(
        page_title="Gemini Chatbot",
        page_icon="🤖",
        layout="centered"
    )
    
    initialize_session_state()
    
    # Custom CSS for better appearance
    st.markdown("""
    <style>
        .stTextInput input {
            font-family: monospace;
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
        }
        .stAlert {
            border-radius: 10px;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # API Key Input (only if not already provided)
    if not st.session_state.api_keys_provided:
        st.title("🤖 Gemini Chatbot")
        api_key_input_form()
        st.markdown("""
        <div style="margin-top: 20px; padding: 15px; background: #807d7d; border-radius: 10px; ">
            <h4>ℹ️ About API Keys</h4>
            <ul>
                <li>Your keys are used only during this session</li>
                <li>Keys are never stored or logged anywhere</li>
                <li>You'll need to re-enter keys if you refresh the page</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Main Chat Interface
    st.title("🤖 Gemini Chatbot")
    display_chat_messages()
    
    # Chat input (only if conversation not ended)
    if not st.session_state.conversation_ended:
        if prompt := st.chat_input("Type your message..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.spinner("Generating response..."):
                response = get_gemini_response(prompt, st.session_state.google_api_key)
                if response:
                    st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()
    
    # End conversation button
    if not st.session_state.conversation_ended and st.button("End Conversation"):
        st.session_state.conversation_ended = True
        st.rerun()
    
    # Show summary if conversation ended
    if st.session_state.conversation_ended:
        st.success("Conversation ended. Generating summary...")
        with st.spinner("Creating summary..."):
            summary, sentiment = generate_summary_and_sentiment(st.session_state.openai_api_key)
            if summary and sentiment:
                st.subheader("📝 Conversation Summary")
                st.write(summary)
                st.subheader("😊 Sentiment Analysis")
                st.write(f"{'😊' if 'Positive' in sentiment else '😞' if 'Negative' in sentiment else '😐'} {sentiment}")
            
            if st.button("Start New Conversation"):
                st.session_state.messages = []
                st.session_state.conversation_ended = False
                st.rerun()

if __name__ == "__main__":
    main()