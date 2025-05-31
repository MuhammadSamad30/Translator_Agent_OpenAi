import streamlit as st
import asyncio
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig
from dotenv import load_dotenv
import os

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

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
You are a translator agent. Your job is to translate the given text from one language to another.

Always detect the source language automatically and translate it into the target language as requested.

Keep the meaning, tone, and context accurate while translating.

Do not change names, dates, or important information.

Respond only with the translated text, nothing else.
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
