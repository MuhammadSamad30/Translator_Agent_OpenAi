import streamlit as st
import asyncio
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig

gemini_api_key = st.secrets.get["GEMINI_API_KEY"]

# import streamlit as st
# import asyncio
# from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig
# from dotenv import load_dotenv
# import os

# load_dotenv()
# gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set in .env file.")

external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

translator = Agent(
    name='Translator Agent',
    instructions="""
You are a translator agent created by Muhammad Samad.

Your job is to translate the given text from one language to another. Automatically detect the source language and translate it into the requested target language.

Keep the meaning, tone, and context accurate. Do not change names, dates, or important information. Respond only with the translated text unless the input is about your creator.

🔹 If the user asks anything like:
- "Who made you?"
- "Who is your creator?"
- "Who developed you?"
- "تمہیں کس نے بنایا؟"
- "تمہارا بنانے والا کون ہے؟"

Then respond with:

**English**: "I was created by Muhammad Samad."  
**Urdu**: "مجھے محمد سمعاد نے بنایا ہے۔"

For all other inputs, perform your translation task as usual.
"""
)


async def run_translator_agent(user_input):
    return await Runner.run(translator, input=user_input, run_config=config)

st.set_page_config(page_title="Translator Agent by Muhammad Samad",
                   page_icon="🌍", layout="centered")

st.markdown("""
    <style>
    .title {
        font-size: 2.5em;
        font-weight: bold;
        text-align: center;
        margin-bottom: 0.2em;
    }
    .subtitle {
        font-size: 1.1em;
        text-align: center;
        color: #888;
        margin-bottom: 2em;
    }
    .footer {
        margin-top: 2em;
        text-align: center;
        font-size: 0.9em;
        color: #999;
    }
    </style>
    <div class="title">🌐 Translator Agent</div>
    <div class="subtitle">Powered by Gemini API<br>Created by <strong>Muhammad Samad</strong></div>
""", unsafe_allow_html=True)

with st.container():
    st.markdown("### 📝 Enter your text below")
    user_input = st.text_area(
        label="",
        placeholder="Example: Translate to Urdu: I am learning Python programming.",
        height=150
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        translate_clicked = st.button("🌍 Translate", use_container_width=True)

    if translate_clicked:
        if user_input.strip() == "":
            st.warning("⚠️ Please enter some text to translate.")
        else:
            with st.spinner("Translating..."):
                response = asyncio.run(run_translator_agent(user_input))
                st.success("✅ Translation Complete:")
                st.markdown(
                    f"<div style='padding: 15px; border-radius: 10px; font-size: 1.1em;'>{response.final_output}</div>", unsafe_allow_html=True)

st.markdown("""
    <div class="footer">
        &copy; 2025 Muhammad Samad. All rights reserved.
    </div>
""", unsafe_allow_html=True)
