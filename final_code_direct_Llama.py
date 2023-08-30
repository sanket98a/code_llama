from langchain.llms import HuggingFacePipeline, LlamaCpp,CTransformers
# from langchain.callbacks.base import BaseCallbackHandler
import streamlit as st
from streamlit.components.v1 import html
import streamlit.components.v1 as components
from streamlit_chat import message
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks import StreamlitCallbackHandler
# st_callback = StreamlitCallbackHandler(st.container())
import textwrap
import torch
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Home", page_icon=None, layout="centered",
                   initial_sidebar_state="auto", menu_items=None)


## logo
with st.sidebar:
    st.markdown("""<div style='text-align: left; margin-top:-80px;margin-left:-40px;'>
    <img src="https://affine.ai/wp-content/uploads/2023/05/Affine-Logo.svg" alt="logo" width="300" height="60">
    </div>""", unsafe_allow_html=True)


st.markdown("""
    <div style='text-align: center; margin-top:-70px; margin-bottom: 5px;margin-left: -50px;'>
    <h2 style='font-size: 40px; font-family: Courier New, monospace;
                    letter-spacing: 2px; text-decoration: none;'>
    <img src="https://acis.affineanalytics.co.in/assets/images/logo_small.png" alt="logo" width="70" height="60">
    <span style='background: linear-gradient(45deg, #ed4965, #c05aaf);
                            -webkit-background-clip: text;
                            -webkit-text-fill-color: transparent;
                            text-shadow: none;'>
                    IntelliCodeEx
    </span>
    <span style='font-size: 40%;'>
    <sup style='position: relative; top: 5px; color: #ed4965;'>by Affine</sup>
    </span>
    </h2>
    </div>
    """, unsafe_allow_html=True)

history=[]

if torch.cuda.is_available():
    device_type = "cuda:0"
else:
    device_type = "cpu"
Instruction_prompt = """Get the output in bullet points. Avoid the repetetion."""

with st.sidebar:
    language = st.selectbox("Select Language :-",["C#","Javascript",".NET"])
    # model_name=st.selectbox("Select Model :-",["Llama-2-7B-Chat",'CodeLlama-7B-Instruct'])
    temperature=st.slider("Temperature :-",0.0,1.0,0.7)
    #top_p=st.slider("top_p :-",0.0,1.0,0.95)
    #top_k=st.slider("top_k :- ",0,100,50)
    INSTRUCTION_PROMPT=st.text_area("System Prompt :-",f"{Instruction_prompt}",height=200)

# Default Sys Prompt
# DEFAULT_SYSTEM_PROMPT = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
DEFAULT_SYSTEM_PROMPT=f"""You are {language} coding assistant. Assist the user by explaining.
if you don't know say, 'I don't Know the answer."""

model_id="TheBloke/Llama-2-7B-Chat-GGML"
model_basename="llama-2-7b-chat.ggmlv3.q4_0.bin"

# # Load the selected model
# if model_name=="Llama-2-7B-Chat":
#     print("Llama 7B model Loading")
#     model_id="TheBloke/Llama-2-7B-Chat-GGML"
#     model_basename="llama-2-7b-chat.ggmlv3.q4_0.bin"
# else:
#     print("CodeLlama-7B-Instruct-GGML model Loading")
#     model_id="TheBloke/CodeLlama-7B-Instruct-GGML"
#     model_basename="codellama-7b-instruct.ggmlv3.Q2_K.bin"

# prompt special tokens
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

# create the custom prompt
def get_prompt(
    message: str, system_prompt: str
) -> str:
    texts = [f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n"]
    # # for user_input, response in chat_history:
    # #     texts.append(f"{user_input.strip()} [/INST] {response.strip()} </s><s> [INST] ")
    texts.append(f"user provided code to explain:{message.strip()}\n please explain the above code, following the instructions given by user:\n {INSTRUCTION_PROMPT}[/INST]")
    # prompt=f"""[INST] please explain the user provided code in natural languge. Please wrap your code answer using ```. user provided code:{message}[/INST]"""

    return "".join(texts)

## Load the Local Llama 2 model
def llama_model(model_id, model_basename, max_new_tokens=None, temperature=0.7):
    model_path = hf_hub_download(repo_id=model_id, filename=model_basename)
    #model_path = r"D:\code generation ai merc demo\llama-2-7b-chat.ggmlv3.q4_0.bin"

    kwargs = {
        "model_path": model_path,
        "n_ctx": 2014,
        "max_tokens": 2014,
        }
    if device_type.lower() == "mps":
        kwargs["n_gpu_layers"] = 20
    if device_type.lower() == "cuda:0":
        kwargs["n_gpu_layers"] = 15
        kwargs["n_batch"] = 40
        kwargs["temperature"] =temperature
    print("GGML Model Loaded Succesfully.")
    return LlamaCpp(**kwargs)
   
print("Model Loading start")
model=llama_model(model_id=model_id,model_basename=model_basename,temperature=temperature)
print("Load Model Successfully.")

# print(f"{model_name} Model Loading start")
# model=llama_model(model_path=model_path,temperature=temperature)
# print(f"{model_name}Load Model Successfully.")

# if 'prompts' not in st.session_state:
#     st.session_state.prompts = []
# if 'responses' not in st.session_state:
#     st.session_state.responses = []

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            final_prompt=get_prompt(prompt,DEFAULT_SYSTEM_PROMPT)
            print(final_prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = model.predict(final_prompt)
            # for response in model.generate([final_prompt]):
            #     full_response += response
            #     message_placeholder.markdown(response + "▌")
            message_placeholder.markdown(full_response)
            print(full_response)
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )