import os
import tempfile
from config import OPENAI_API_KEY
import openai
import streamlit as st
from PyPDF2 import PdfReader
from langchain.document_loaders import Docx2txtLoader
from langchain.text_splitter import CharacterTextSplitter

# API KEYS
OPENAI_API_KEY = OPENAI_API_KEY  # Access the API key from environment variable in config.py

st.title('Welcome to Bahamas AI')
st.subheader('Your new AI assistant')

# variable to input to ChatGPT API eventually
text_gpt = []

# Handle uploaded files
uploaded_files = st.file_uploader("Upload a video meeting transcription", type=("docx", "pdf"),
                                  accept_multiple_files=True)

if uploaded_files:
    # Create a temporary directory to store the uploaded files
    temp_dir = tempfile.TemporaryDirectory()

    # Save each uploaded file to the temporary directory
    docx_file_paths = []
    docx_file_names = []
    pdf_file_paths = []
    for uploaded_file in uploaded_files:
        if uploaded_file.name.endswith(".docx"):
            file_path = os.path.join(temp_dir.name, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())
            docx_file_paths.append(file_path)
            docx_file_names.append(uploaded_file.name)
        elif uploaded_file.name.endswith(".pdf"):
            pdf_reader = PdfReader(uploaded_file)
            raw_text = ''
            for i, page in enumerate(pdf_reader.pages):
                content = page.extract_text()
                if content:
                    raw_text += content
                else:
                    break
            # We need to split the text using Character Text Split such that it should not increase token size
            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=800,
                chunk_overlap=200,
                length_function=len,
            )
            texts = text_splitter.split_text(raw_text)
            text_final = '\n'.join(texts)
            text_gpt.append(
                "PDF input number " + str(i) + " :" + uploaded_file.name + ": " + text_final)

    # Now you can access the file paths and use Docx2txtLoader to retrieve the text
    for idx, file_path in enumerate(docx_file_paths):
        st.write(f"Content of {file_path}:")
        loader = Docx2txtLoader(file_path)
        data = loader.load()
        text = str(data[0])
        # Extracting the page content
        start_index = text.find("page_content=") + len("page_content=")
        end_index = text.find(" metadata=")
        page_content = text[start_index:end_index].strip()

        # Storing the page content
        text_gpt.append(
            "Video meeting transcription number " + str(idx) + " :" + docx_file_names[idx] + ": " + page_content)
        st.write(text_gpt)


# Handle multiple text inputs


# Create a function to manage the text fields using Streamlit Session State
# Create a function to manage the text fields using Streamlit Session State
# Create a function to manage the text fields using Streamlit Session State
def manage_text_fields():
    # Initialize the session state variable if it doesn't exist
    if "text_fields" not in st.session_state:
        st.session_state.text_fields = []

    # Initialize the session state variable for dropdown values if it doesn't exist
    if "dropdown_values" not in st.session_state:
        st.session_state.dropdown_values = []

    # Create the "Add Text Field" button
    if st.button("Add Text Field"):
        st.session_state.text_fields.append("")
        st.session_state.dropdown_values.append("---")  # Default value

    # Display all the text fields and dropdowns in the session state lists
    for i, (text_field, dropdown_value) in enumerate(
            zip(st.session_state.text_fields, st.session_state.dropdown_values)):
        st.write(f"Text Field {i + 1}:")
        text_input_key = f"text_input_{i}"
        dropdown_key = f"dropdown_{i}"
        st.text_area("Add your text below :point_down:", text_field, key=text_input_key)
        st.selectbox("Please indicate input type :warning: ",
                     options=["PDF Summary", "Email", "Additional Information"], index=0,
                     key=dropdown_key)

        # Update the dropdown value in the session state when changed by the user
        st.session_state.dropdown_values[i] = st.session_state[dropdown_key]
    st.expander("Show all text fields", expanded=True)


# Call the function to manage the text fields and dropdowns
manage_text_fields()

## FIRST STEP

# Integrate with ChatGPT API

system_content = "You are a professional assistant for a digital development company called 'The Amazingfull " \
                 "Circus', with one job: summarize the following input and " \
                 "highlight the " \
                 "most important points under bullet points but still make full sentences, as if you were " \
                 "using the input to build a project proposal contract. You will get " \
                 "text inputs. If information is repeated in multiple " \
                 "text inputs, try not repeat it to much, or rephrase it. Each " \
                 "text input is going to indicate beforehand what kind of input it is: a PDF summary, " \
                 "a general conversation, a video meeting transcription or an email. Take that into account " \
                 "when " \
                 "analysing the text inputs. If you get different spellings, always choose the spelling that " \
                 "is provided in an email, a pdf summary or a general text before the ones coming from a " \
                 "a video meeting transcription. \n\nYour response should be organized " \
                 "" \
                 "as follows: \n\n\1. Project " \
                 "description\n\n 2. Goal\n\n3. Objectives \n\n You can add some " \
                 "analysis regarding missing information " \
                 "in each section. Watch out for proper nouns and acronyms. Document names should be " \
                 "a good source for company names. Don't trust video meeting transcriptions for proper nouns " \
                 "orthography \n\n"
# Open AI LLMS
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

chat = ChatOpenAI(openai_api_key=OPENAI_API_KEY,
                  temperature=0.9)

template = "You are a professional assistant for a digital development company called 'The Amazingfull " \
                 "Circus', with one job: summarize the following input and " \
                 "highlight the " \
                 "most important points under bullet points but still make full sentences, as if you were " \
                 "using the input to build a project proposal contract: {input}. If information is repeated in " \
                 " try not repeat it to much, or rephrase it. Each " \
                 "text input indicates beforehand what kind of input it is: a PDF summary, " \
                 "a general conversation, a video meeting transcription or an email. Take that into account " \
                 "when " \
                 "analysing the text inputs. If you get different spellings, always choose the spelling that " \
                 "is provided in an email, a pdf summary or a general text before the ones coming from a " \
                 "a video meeting transcription. \n\nYour response should be organized " \
                 "" \
                 "as follows: {output} \n\n You can add some " \
                 "analysis regarding missing information " \
                 "in each section. Watch out for proper nouns and acronyms. Document names should be " \
                 "a good source for company names. Don't trust video meeting transcriptions for proper nouns " \
                 "orthography \n\n"

system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template = "\n\n".join(text_gpt)
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

chain = LLMChain(llm=chat, prompt=chat_prompt)


if st.button("Run", key="run"):
    st.write(chain.run(input=system_content, output="\n\n\1. Project description\n\n 2. Goal\n\n3. Objectives"))
