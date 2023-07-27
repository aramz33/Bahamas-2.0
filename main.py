import os
import tempfile
import streamlit as st
from PyPDF2 import PdfReader
from langchain.document_loaders import Docx2txtLoader
from langchain.text_splitter import CharacterTextSplitter
import jinja2
import pdfkit

st.set_page_config(page_title="Bahamas AI", page_icon=":robot_face:", layout="wide",
                   initial_sidebar_state="collapsed")
# API KEYS


OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

if "final" not in st.session_state:
    st.session_state.final = False

if "button_2" not in st.session_state:
    st.session_state.button_2 = False

if "button_3" not in st.session_state:
    st.session_state.button_3 = False

if "button_4" not in st.session_state:
    st.session_state.button_4 = False

if "rerun" not in st.session_state:
    st.session_state.rerun = False

# Title
st.title(':palm_tree: WELCOME TO BAHAMAS AI :palm_tree:')
st.header('Your new tropical AI assistant :robot_face: :tropical_drink:')

# Sidebar
with st.sidebar:
    st.header("Settings")
    st.slider("Temperature", min_value=0.0, max_value=1.4, value=0.3, step=0.1, key="temperature")
    st.caption(
        'The lower the temperature, the more predictable the text is. Lower temperatures are more "boring" while '
        'higher temperatures are more surprising. 0.7 is the default on ChatGPT.')

# variable to input to ChatGPT API eventually
if "text_gpt" not in st.session_state:
    st.session_state.text_gpt = []


# Handle uploaded files
def handle_uploaded_files(uploaded_files):
    if uploaded_files:
        # Create a temporary directory to store the uploaded files
        temp_dir = tempfile.TemporaryDirectory()

        # Save each uploaded file to the temporary directory
        docx_file_paths = []
        docx_file_names = []
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
                for j, page in enumerate(pdf_reader.pages):
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
                st.session_state.text_gpt.append(
                    "PDF input number " + str(j) + " :" + uploaded_file.name + ": " + text_final)

        # Now you can access the file paths and use Docx2txtLoader to retrieve the text
        for idx, file_path in enumerate(docx_file_paths):
            loader = Docx2txtLoader(file_path)
            data = loader.load()
            text = str(data[0])
            # Extracting the page content
            start_index = text.find("page_content=") + len("page_content=")
            end_index = text.find(" metadata=")
            page_content = text[start_index:end_index].strip()

            # Storing the page content
            st.session_state.text_gpt.append(
                "Video meeting transcription number " + str(idx) + " :" + docx_file_names[idx] + ": " + page_content)
            st.write(st.session_state.text_gpt)


# Integrate with ChatGPT API
# Open AI LLMS
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

if 'chain1_1' not in st.session_state:
    st.session_state['chain1_1'] = None
if 'chain1_2' not in st.session_state:
    st.session_state['chain1_2'] = None
if 'chain1_3' not in st.session_state:
    st.session_state['chain1_3'] = None

if 'chain2_1' not in st.session_state:
    st.session_state['chain2_1'] = None
if 'chain2_2' not in st.session_state:
    st.session_state['chain2_2'] = None
if 'chain2_3' not in st.session_state:
    st.session_state['chain2_3'] = None

if 'chain3_1' not in st.session_state:
    st.session_state['chain3_1'] = None
if 'chain3_2' not in st.session_state:
    st.session_state['chain3_2'] = None
if 'chain3_3' not in st.session_state:
    st.session_state['chain3_3'] = None

if 'comments' not in st.session_state:
    st.session_state['comments'] = []

if 'comments_input' not in st.session_state:
    st.session_state['comments_input'] = ""

st.session_state.system_content = "You are a professional assistant for a digital development company called 'The " \
                                  "Amazingfull " \
                                  "Circus'. You have two jobs: summarize the following input and " \
                                  "provide the important information for project proposals. Act like you would be " \
                                  "using the " \
                                  "following input to build a project proposal contract: {input}. \n\n " \
                                  "\n\nYour response should be organized in 3 parts " \
                                  "as follows: {output} \n\n. WRITE ONLY ABOUT THESE PARTS, NOT MORE. Act " \
                                  "professional. If information is repeated " \
                                  "don't repeat it too much, or rephrase it. Each " \
                                  "text input indicates beforehand what kind of input it is: a PDF summary, " \
                                  "additional information inputed by the user or an email. Take that into " \
                                  "account when " \
                                  "analysing the text inputs.  Watch out for proper nouns and acronyms. Document names " \
                                  "should be " \
                                  "a good source for company names. Be direct and concise. USE PROFESSIONAL LANGUAGE. " \
                                  "WRITE EVERYTHING IN BULLET POINTS. \n\n"


def generate_final(comment):
    chat = ChatOpenAI(openai_api_key=OPENAI_API_KEY,
                      temperature=st.session_state.temperature)

    if comment in st.session_state.comments:
        st.session_state.comments_input = "\n\n".join(st.session_state.comments)
    else:
        st.session_state.comments.append(comment)
        st.session_state.comments_input = "\n\n".join(st.session_state.comments)

    final_system_content = "Put the following information together into a project proposal " \
                                            "contract: {input}. \n\n Your response will be used to write a pdf " \
                                            "document on python. \n\n Here are some user comments to keep in mind" + st.session_state.comments_input
    final_system_message_prompt = SystemMessagePromptTemplate.from_template(
        final_system_content
    )
    final_user_template = "\n\n".join(
        [st.session_state.step_1_choice, st.session_state.step_2_choice, st.session_state.step_4_choice])
    final_input_prompt = HumanMessagePromptTemplate.from_template(
        final_user_template
    )

    chat_prompt = ChatPromptTemplate.from_messages(
        [final_system_message_prompt, final_input_prompt]
    )

    st.session_state.parent_chain = LLMChain(llm=chat, prompt=chat_prompt, output_key="Final result")
    # Get the output from the chat model
    final_output = st.session_state.parent_chain.run(input=final_input_prompt)
    st.write(final_output)

    return final_output


def generate_summary(comment):
    chat = ChatOpenAI(openai_api_key=OPENAI_API_KEY,
                      temperature=0.8)

    if comment in st.session_state.comments:
        st.session_state.comments_input = "\n\n".join(st.session_state.comments)
    else:
        st.session_state.comments.append(comment)
        st.session_state.comments_input = "\n\n".join(st.session_state.comments)

    final_system_content = "Summarize the following information in a paragraph of less than 70 words : {input}. Do not right in bullet points." \
                            "\n\n Here are some very important user comments to keep in mind :" + st.session_state.comments_input


    final_system_message_prompt = SystemMessagePromptTemplate.from_template(
        final_system_content
    )

    st.write(final_system_message_prompt)

    final_user_template = "\n\n".join(
        [st.session_state.step_1_choice, st.session_state.step_2_choice, st.session_state.step_4_choice])
    final_input_prompt = HumanMessagePromptTemplate.from_template(
        final_user_template
    )

    chat_prompt = ChatPromptTemplate.from_messages(
        [final_system_message_prompt, final_input_prompt]
    )

    st.session_state.summary_chain = LLMChain(llm=chat, prompt=chat_prompt, output_key="Summary")

    st.write(st.session_state.summary_chain.run(input=final_input_prompt))


def run_generation_1():
    chat = ChatOpenAI(openai_api_key=OPENAI_API_KEY,
                      temperature=st.session_state.temperature)
    system_message_prompt = SystemMessagePromptTemplate.from_template(
        st.session_state.system_content
    )
    first_user_template = "\n\n".join(st.session_state.text_gpt)
    first_input_prompt = HumanMessagePromptTemplate.from_template(
        first_user_template
    )

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, first_input_prompt]
    )

    st.session_state.chain1 = LLMChain(llm=chat, prompt=chat_prompt, output_key="Step 1 result")

    with st.spinner(' :robot_face: Initializing...'):
        st.session_state['chain1_1'] = st.session_state.chain1.run(input=first_input_prompt,
                                                                   output="\n\n - Project description \n\n - Objectives \n\n - Scope")
        st.session_state['chain1_2'] = st.session_state.chain1.run(input=first_input_prompt,
                                                                   output="\n\n - Project description \n\n - Objectives \n\n - Scope")
        st.session_state['chain1_3'] = st.session_state.chain1.run(input=first_input_prompt,
                                                                   output="\n\n - Project description \n\n - Objectives \n\n - Scope")


def run_generation_2():
    chat = ChatOpenAI(openai_api_key=OPENAI_API_KEY,
                      temperature=st.session_state.temperature)
    system_message_prompt = SystemMessagePromptTemplate.from_template(
        st.session_state.system_content
    )
    first_user_template = "\n\n".join(st.session_state.text_gpt)
    first_input_prompt = HumanMessagePromptTemplate.from_template(
        first_user_template
    )

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, first_input_prompt]
    )
    st.session_state.chain2 = LLMChain(llm=chat, prompt=chat_prompt, output_key="Step 2 result")

    with st.spinner(' :robot_face: Calculating...'):
        st.session_state['chain2_1'] = st.session_state.chain2.run(input=first_input_prompt,
                                                                   output="\n\n - Deliverables \n\n - Timeline \n\n - Budget")
        st.session_state['chain2_2'] = st.session_state.chain2.run(input=first_input_prompt,
                                                                   output="\n\n - Deliverables \n\n - Timeline \n\n - Budget")
        st.session_state['chain2_3'] = st.session_state.chain2.run(input=first_input_prompt,
                                                                   output="\n\n - Deliverables \n\n - Timeline \n\n - Budget")


def run_generation_3():
    chat = ChatOpenAI(openai_api_key=OPENAI_API_KEY,
                      temperature=st.session_state.temperature)
    system_message_prompt = SystemMessagePromptTemplate.from_template(
        st.session_state.system_content
    )
    first_user_template = "\n\n".join(st.session_state.text_gpt)
    first_input_prompt = HumanMessagePromptTemplate.from_template(
        first_user_template
    )

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, first_input_prompt]
    )

    st.session_state.chain3 = LLMChain(llm=chat, prompt=chat_prompt, output_key="Step 3 result")

    with st.spinner(' :robot_face: Wait for it...'):
        st.session_state['chain3_1'] = st.session_state.chain3.run(input=first_input_prompt,
                                                                   output="\n\n - Goals \n\n - Methodolgy \n\n - Risks")
        st.session_state['chain3_2'] = st.session_state.chain3.run(input=first_input_prompt,
                                                                   output="\n\n - Goals \n\n - Methodolgy \n\n - Risks")
        st.session_state['chain3_3'] = st.session_state.chain3.run(input=first_input_prompt,
                                                                   output="\n\n - Goals \n\n - Methodolgy \n\n - Risks")


## INPUTS
with st.form("Inputs"):
    uploaded_files = st.file_uploader("Upload your files", type=("docx", "pdf"),
                                      accept_multiple_files=True)

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12, tab13, tab14, tab15, tab16, tab17, tab18, tab19, tab20  = st.tabs(
        ["Text 1", "Text 2", "Text 3", "Text 4", "Text 5", "Text 6", "Text 7", "Text 8", "Text 9", "Text 10", "Text 11", "Text 12", "Text 13", "Text 14", "Text 15", "Text 16", "Text 17", "Text 18", "Text 19", "Text 20"])
    # Initialize the session state variable if it doesn't exist
    if "text_fields" not in st.session_state:
        st.session_state.text_fields = []

        # Initialize the session state variable for dropdown values if it doesn't exist
    if "dropdown_values" not in st.session_state:
        st.session_state.dropdown_values = []

    # Display all the text fields and dropdowns in the session state lists
    idx = 0
    for tab in tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12, tab13, tab14, tab15, tab16, tab17, tab18, tab19, tab20:
        with tab:
            idx += 1
            st.subheader("Text inputs")
            text_input_key = f"text_input_{idx}"
            dropdown_key = f"dropdown_{idx}"
            st.text_area("", key=text_input_key, placeholder="Add your text here", label_visibility="collapsed")
            st.selectbox("Please indicate input type :warning: ",
                         options=["", "PDF Summary", "Email", "Additional Information"],
                         key=dropdown_key, format_func=lambda x: "Select input type" if x == '' else x, label_visibility="collapsed")

    for idx in range(1, 21):
        st.session_state.dropdown_values.append(st.session_state[f"dropdown_{idx}"])
        st.session_state.text_fields.append(st.session_state[f"text_input_{idx}"])
        if st.session_state[f"dropdown_{idx}"] != "" and st.session_state[f"text_input_{idx}"] is not None:
            st.session_state.text_gpt.append(
                st.session_state[f"dropdown_{idx}"] + ": " + st.session_state[f"text_input_{idx}"])

    st.caption("Please click on 'Run' once you have uploaded all your files and added your text inputs")

    st.session_state.submit_input = st.form_submit_button('Run', use_container_width=True)
    if st.session_state.submit_input:
        with st.empty():
            with st.spinner('Running...'):
                handle_uploaded_files(uploaded_files)
                run_generation_1()
                st.experimental_rerun()
        st.write(":white_check_mark: Done! Proceed to Step 1 :white_check_mark:")

# Divider

st.divider()

###### FIRST STEP

with st.form("Project description - Objectives - Scope"):
    st.subheader("STEP 1 : Project description - Objectives - Scope")
    tab1, tab2, tab3 = st.tabs(["Text 1", "Text 2", "Text 3"])

    if st.session_state['chain1_1'] and st.session_state['chain1_2'] and st.session_state['chain1_3'] is not None:
        with tab1:
            st.write(st.session_state['chain1_1'])
        with tab2:
            st.write(st.session_state['chain1_2'])
        with tab3:
            st.write(st.session_state['chain1_3'])

    if "step_1_choice" not in st.session_state:
        st.session_state.step_1_choice = None

    st.session_state.step_1_choice = st.selectbox("Please choose your favorite output",
                                                  options=["Text 1", "Text 2", "Text 3"], key="step1_choice")

    st.write("You have chosen: " + st.session_state.step_1_choice)

    # store the text choice in a variable
    if st.session_state.step1_choice == "Text 1":
        st.session_state.step_1_choice = st.session_state['chain1_1']
    elif st.session_state.step1_choice == "Text 2":
        st.session_state.step_1_choice = st.session_state['chain1_2']
    elif st.session_state.step1_choice == "Text 3":
        st.session_state.step_1_choice = st.session_state['chain1_3']

    st.session_state.submit_step1 = st.form_submit_button('Confirm choice')

    if st.session_state.submit_step1:
        with st.empty():
            with st.spinner('Running...'):
                run_generation_2()
        st.write(" :white_check_mark: Done!  Proceed to step 2 ! :white_check_mark: ")

    # comment = st.text_input("Please add your comments here", max_chars=1000)
    # st.caption("Please click on the button below to return a better text")
    # rerun = st.button(label='Rerun')

st.divider()

###### SECOND STEP
with st.form("Deliverables - Timeline - Budget"):
    st.subheader("STEP 2: Deliverables - Timeline - Budget")
    tab1, tab2, tab3 = st.tabs(["Text 1", "Text 2", "Text 3"])

    if st.session_state['chain2_1'] and st.session_state['chain2_2'] and st.session_state['chain2_3'] is not None:
        with tab1:
            st.write(st.session_state['chain2_1'])
        with tab2:
            st.write(st.session_state['chain2_2'])
        with tab3:
            st.write(st.session_state['chain2_3'])

    if "step_2_choice" not in st.session_state:
        st.session_state.step_2_choice = None

    st.session_state.step_2_choice = st.selectbox("Please choose your favorite output",
                                                  options=["Text 1", "Text 2", "Text 3"], key="step2_choice")

    st.write("You have chosen: " + st.session_state.step_2_choice)

    # store the text choice in a variable
    if st.session_state.step2_choice == "Text 1":
        st.session_state.step_2_choice = st.session_state['chain2_1']
    elif st.session_state.step2_choice == "Text 2":
        st.session_state.step_2_choice = st.session_state['chain2_2']
    elif st.session_state.step2_choice == "Text 3":
        st.session_state.step_2_choice = st.session_state['chain2_3']

    st.session_state.submit_step2 = st.form_submit_button('Confirm choice', disabled=not st.session_state.submit_step1)

    if st.session_state.submit_step2:
        with st.empty():
            with st.spinner('Running...'):
                run_generation_3()
        st.write(" :white_check_mark: Done! Proceed to Step 3 ! :white_check_mark:")

st.divider()

###### THIRD STEP
with st.form("Goals - Methodology - Risks"):
    st.subheader("STEP 3: Goals - Methodology - Risks")
    tab1, tab2, tab3 = st.tabs(["Text 1", "Text 2", "Text 3"])

    if st.session_state['chain3_1'] and st.session_state['chain3_2'] and st.session_state['chain3_3'] is not None:
        with tab1:
            st.write(st.session_state['chain3_1'])
        with tab2:
            st.write(st.session_state['chain3_2'])
        with tab3:
            st.write(st.session_state['chain3_3'])

    if "step_4_choice" not in st.session_state:
        st.session_state.step_4_choice = None

    st.session_state.step_4_choice = st.selectbox("Please choose your favorite output",
                                                  options=["Text 1", "Text 2", "Text 3"], key="step4_choice")

    st.write("You have chosen: " + st.session_state.step_4_choice)

    # store the text choice in a variable
    if st.session_state.step4_choice == "Text 1":
        st.session_state.step_4_choice = st.session_state['chain3_1']
    elif st.session_state.step4_choice == "Text 2":
        st.session_state.step_4_choice = st.session_state['chain3_2']
    elif st.session_state.step4_choice == "Text 3":
        st.session_state.step_4_choice = st.session_state['chain3_3']

    st.session_state.submit_step3 = st.form_submit_button('Generate output', disabled=not st.session_state.submit_step2)

st.session_state.comment = st.text_area("Correct the generated output by inputting your comments below", placeholder="Add your comments here as if you were talking to ChatGPT \nDon't repeat the same comment twice \nDon't rewrite the same comment after rerunning" )

st.session_state.rerun = st.button(label='Rerun', use_container_width=True)

if st.session_state.submit_step3:
    with st.spinner('Generating contract...'):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Final result")
            if "comment" not in st.session_state:
                st.session_state.comment = ''
            generate_final(st.session_state.comment)
        with col2:
            st.subheader("Summary")
            generate_summary(st.session_state.comment)

if st.session_state.rerun:
    with st.spinner('Generating contract...'):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Final result")
            if "comment" not in st.session_state:
                st.session_state.comment = ''
            generate_final(st.session_state.comment)
        with col2:
            st.subheader("Summary")
            generate_summary(st.session_state.comment)

st.write(st.session_state.comments_input)
