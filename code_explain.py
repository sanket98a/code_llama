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

st.title("Affine-LocalGPT")

history=[]
if torch.cuda.is_available():
    device_type = "cuda:0"
else:
    device_type = "cpu"


# Default Sys Prompt
# DEFAULT_SYSTEM_PROMPT = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
DEFAULT_SYSTEM_PROMPT="""You are python coding assistant.Assist the user by explaining.
if you don't know say, 'I don't Know the answer.'"""

with st.sidebar:
    model_name=st.selectbox("Select Model :-",['Llama 7B','Llama 13B'])
    temperature=st.slider("Temperature :-",0.0,1.0,0.1)
    top_p=st.slider("top_p :-",0.0,1.0,0.95)
    top_k=st.slider("top_k :- ",0,100,50)
    DEFAULT_SYSTEM_PROMPT=st.text_area("System Prompt :-",f"{DEFAULT_SYSTEM_PROMPT}",height=400)

# Load the selected model
if model_name=="Llama 7B":
    print("Llama 7B model Loading")
    model_path='llama-2-7b-chat.ggmlv3.q4_0.bin'
else:
    print("Llama 13B model Loading")
    model_path="llama-2-13b-chat.ggmlv3.q2_K.bin"

# prompt special tokens
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"


# create the custom prompt
def get_prompt(
    message: str, chat_history: list[tuple[str, str]], system_prompt: str
) -> str:
    texts = [f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n"]
    for user_input, response in chat_history:
        texts.append(f"{user_input.strip()} [/INST] {response.strip()} </s><s> [INST] ")
    texts.append(f"{message.strip()} [/INST]")
    return "".join(texts)


## Load the Local Llama 2 model
def llama_model(model_id=None,model_basename=None,max_new_tokens=None,temperature=None):
    model_path = hf_hub_download(repo_id=model_id, filename=model_basename)
  
    kwargs = {
        "model_path": model_path,
        "n_ctx": 1024,
        "max_tokens": 1024,
        }
    if device_type.lower() == "mps":
        kwargs["n_gpu_layers"] = 25
    if device_type.lower() == "cuda:0":
        kwargs["n_gpu_layers"] = 25
        kwargs["n_batch"] = 60
        kwargs["temperature"] =0.7
    print("GGML Model Loaded Succesfully.")
    return LlamaCpp(**kwargs)
   


model_path="TheBloke/CodeLlama-7B-Instruct-GGML"
model_basename="codellama-7b-instruct.ggmlv3.Q2_K.bin"
print(f"{model_name} Model Loading start")
model=llama_model(model_path=model_path,model_basename=model_basename,temperature=temperature)
print(f"{model_name}Load Model Successfully.")

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
            final_prompt=get_prompt(prompt,history,DEFAULT_SYSTEM_PROMPT)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for response in model.predict(final_prompt):
                full_response += response
                message_placeholder.markdown(response + "▌")
            message_placeholder.markdown(full_response)
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )