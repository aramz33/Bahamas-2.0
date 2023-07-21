import os
import tempfile
import streamlit as st
from PyPDF2 import PdfReader
from langchain.document_loaders import Docx2txtLoader
from langchain.text_splitter import CharacterTextSplitter

st.set_page_config(page_title="Bahamas AI", page_icon=":robot_face:", layout="wide", initial_sidebar_state="auto")

# API KEYS
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

if "button_1" not in st.session_state:
    st.session_state.button_1 = False


def toggle1():
    if st.session_state.button_1:
        st.session_state.button_1 = False
    else:
        st.session_state.button_1 = True


if "button_2" not in st.session_state:
    st.session_state.button_2 = False


def toggle2():
    if st.session_state.button_2:
        st.session_state.button_2 = False
    else:
        st.session_state.button_2 = True


if "button_3" not in st.session_state:
    st.session_state.button_3 = False


def toggle3():
    if st.session_state.button_3:
        st.session_state.button_3 = False
    else:
        st.session_state.button_3 = True


if "button_4" not in st.session_state:
    st.session_state.button_4 = False


def toggle4():
    if st.session_state.button_4:
        st.session_state.button_4 = False
    else:
        st.session_state.button_4 = True


# Title
st.title('Welcome to Bahamas AI')
st.subheader('Your new AI assistant')

# Sidebar
with st.sidebar:
    st.header("Settings")
    st.slider("Temperature", min_value=0.0, max_value=1.4, value=0.6, step=0.1, key="temperature")
    st.caption(
        'The lower the temperature, the more predictable the text is. Lower temperatures are more "boring" while '
        'higher temperatures are more surprising. 0.7 is the default on ChatGPT.')

# variable to input to ChatGPT API eventually
text_gpt = []

with st.expander("Inputs", expanded=not st.session_state.button_1):
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

    st.session_state


    # Handle multiple text inputs

    # Create a function to manage the text fields using Streamlit Session State
    def manage_text_fields():
        col1, col2 = st.columns([0.7, 0.3])

        # Initialize the session state variable if it doesn't exist
        if "text_fields" not in st.session_state:
            st.session_state.text_fields = []

        # Initialize the session state variable for dropdown values if it doesn't exist
        if "dropdown_values" not in st.session_state:
            st.session_state.dropdown_values = []

        # Create the "Add Text Field" button
        if st.button("Add Text Field"):
            st.session_state.text_fields.append("")
            st.session_state.dropdown_values.append("")  # Default value

        # Display all the text fields and dropdowns in the session state lists
        for idx, (text_field, dropdown_value) in enumerate(
                zip(st.session_state.text_fields, st.session_state.dropdown_values)):
            with st.container():
                col1.subheader(f"Text Field {idx + 1}:")
                text_input_key = f"text_input_{idx}"
                dropdown_key = f"dropdown_{idx}"
                col1.text_area("Add your text below :point_down:", text_field, key=text_input_key)
                col2.selectbox("Please indicate input type :warning: ",
                               options=["PDF Summary", "Email", "Additional Information"],
                               key=dropdown_key)

                # Update the dropdown value in the session state when changed by the user
                st.session_state.dropdown_values[i] = st.session_state[dropdown_key]


    # Call the function to manage the text fields and dropdowns
    manage_text_fields()

    ## FIRST STEP

    # Integrate with ChatGPT API

    # Open AI LLMS
    from langchain import LLMChain
    from langchain.chat_models import ChatOpenAI
    from langchain.prompts.chat import (
        ChatPromptTemplate,
        SystemMessagePromptTemplate,
        HumanMessagePromptTemplate,
    )

    chat = ChatOpenAI(openai_api_key=OPENAI_API_KEY,
                      temperature=st.session_state.temperature)

    st.session_state.system_content = "You are a professional assistant for a digital development company called 'The " \
                                      "Amazingfull " \
                                      "Circus', with one job: summarize the following input and " \
                                      "provide the important information for project proposals. Act like you would be " \
                                      "using the" \
                                      "following input to build a project proposal contract: {input}. \n\n " \
                                      "If information is repeated " \
                                      " try not repeat it to much, or rephrase it. Each " \
                                      "text input indicates beforehand what kind of input it is: a PDF summary, " \
                                      "a general conversation, a video meeting transcription or an email. Take that into account when " \
                                      "analysing the text inputs. \n\nYour response should be organized " \
                                      "as follows: {output} \n\n Watch out for proper nouns and acronyms. Document names should be " \
                                      "a good source for company names. Be direct and concise. Don't trust video meeting transcriptions for proper nouns " \
                                      "orthography \n\n"

    system_message_prompt = SystemMessagePromptTemplate.from_template(
        st.session_state.system_content
    )
    first_user_template = "\n\n".join(text_gpt)
    first_input_prompt = HumanMessagePromptTemplate.from_template(
        first_user_template
    )

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, first_input_prompt]
    )

    st.session_state.chain1 = LLMChain(llm=chat, prompt=chat_prompt, verbose=True, output_key="Step 1 result")
    st.session_state.chain2 = LLMChain(llm=chat, prompt=chat_prompt, verbose=True, output_key="Step 2 result")
    st.session_state.chain3 = LLMChain(llm=chat, prompt=chat_prompt, verbose=True, output_key="Step 3 result")

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

    st.caption('Please click on the button below once you have uploaded all your files and added your text inputs')


    def run_generation_1():
        with st.spinner('Generating Step 1...'):
            st.session_state['chain1_1'] = st.session_state.chain1.run(input=first_input_prompt,
                                                                       output="\n\n 1. Project description \n\n 2. Objectives \n\n 3. Scope")
            st.session_state['chain1_2'] = st.session_state.chain1.run(input=first_input_prompt,
                                                                       output="\n\n 1. Project description \n\n 2. Objectives \n\n 3. Scope")
            st.session_state['chain1_3'] = st.session_state.chain1.run(input=first_input_prompt,
                                                                       output="\n\n 1. Project description \n\n 2. Objectives \n\n 3. Scope")


    def run_generation_2():
        with st.spinner('Generating Step 2...'):
            st.session_state['chain2_1'] = st.session_state.chain2.run(input=first_input_prompt,
                                                                       output="\n\n - Deliverables \n\n - Timeline \n\n - Budget")
            st.session_state['chain2_2'] = st.session_state.chain2.run(input=first_input_prompt,
                                                                       output="\n\n - Deliverables \n\n - Timeline \n\n - Budget")
            st.session_state['chain2_3'] = st.session_state.chain2.run(input=first_input_prompt,
                                                                       output="\n\n - Deliverables \n\n - Timeline \n\n - Budget")


    def run_generation_3():
        with st.spinner('Generating Step 3...'):
            st.session_state['chain3_1'] = st.session_state.chain3.run(input=first_input_prompt,
                                                                       output="\n\n - Goals \n\n - Methodolgy \n\n - Risks")
            st.session_state['chain3_2'] = st.session_state.chain3.run(input=first_input_prompt,
                                                                       output="\n\n - Goals \n\n - Methodolgy \n\n - Risks")
            st.session_state['chain3_3'] = st.session_state.chain3.run(input=first_input_prompt,
                                                                       output="\n\n - Goals \n\n - Methodolgy \n\n - Risks")


    if st.button(label='Run Generation', use_container_width=True):
        with st.empty():
            run_generation_1()
        st.write("Done! :sunglasses: Please click on the 'Next' button to continue")

# Next Button

st.button(label='Next', on_click=toggle1, disabled=st.session_state.button_1, type='primary', key="button1")

# Divider
st.divider()

###### Second step

with st.expander("Project description - Objectives - Scope", expanded=st.session_state.button_1):
    tab1, tab2, tab3 = st.tabs(["Text 1", "Text 2", "Text 3"])

    if st.session_state['chain1_1'] and st.session_state['chain1_2'] and st.session_state['chain1_3'] is not None:
        with tab1:
            st.write(st.session_state['chain1_1'])
        with tab2:
            st.write(st.session_state['chain1_2'])
        with tab3:
            st.write(st.session_state['chain1_3'])

    if "step_2_choice" not in st.session_state:
        st.session_state.step_2_choice = None

    st.session_state.step_2_choice = st.selectbox("Please choose your favorite generated text",
                                                  options=["Text 1", "Text 2", "Text 3"], key="step2_choice")

    with st.empty():
        run_generation_2()

    st.write("You have chosen: " + st.session_state.step_2_choice)

    # store the text choice in a variable
    if st.session_state.step_2_choice == "Text 1":
        st.session_state.step_2_choice = st.session_state['chain1_1']
        st.session_state['chain1_2']=None
        st.session_state['chain1_3']=None
    elif st.session_state.step_2_choice == "Text 2":
        st.session_state.step_2_choice = st.session_state['chain1_2']
        st.session_state['chain1_1']=None
        st.session_state['chain1_3']=None
    elif st.session_state.step_2_choice == "Text 3":
        st.session_state.step_2_choice = st.session_state['chain1_3']
        st.session_state['chain1_1']=None
        st.session_state['chain1_2']=None

    # comment = st.text_input("Please add your comments here", max_chars=1000)
    # st.caption("Please click on the button below to return a better text")
    # rerun = st.button(label='Rerun')

st.button(label='Next', on_click=toggle2, disabled=st.session_state.button_2, type='primary', key="button2")

st.divider()

with st.expander("Deliverables - Timeline - Budget", expanded=st.session_state.button_2):
    tab1, tab2, tab3 = st.tabs(["Text 1", "Text 2", "Text 3"])



    if st.session_state['chain2_1'] and st.session_state['chain2_2'] and st.session_state['chain2_3'] is not None:
        with tab1:
            st.write(st.session_state['chain2_1'])
        with tab2:
            st.write(st.session_state['chain2_2'])
        with tab3:
            st.write(st.session_state['chain2_3'])

    if "step_3_choice" not in st.session_state:
        st.session_state.step_3_choice = None

    with st.empty():
        run_generation_3()

    st.session_state.step_3_choice = st.selectbox("Please choose your favorite generated text",
                                                  options=["Text 1", "Text 2", "Text 3"], key="step3_choice")

    st.write("You have chosen: " + st.session_state.step_3_choice)


    # store the text choice in a variable
    if st.session_state.step_3_choice == "Text 1":
        st.session_state.step_3_choice = st.session_state['chain2_1']
        st.session_state['chain2_2']=None
        st.session_state['chain2_3']=None
    elif st.session_state.step_3_choice == "Text 2":
        st.session_state.step_3_choice = st.session_state['chain2_2']
        st.session_state['chain2_1']=None
        st.session_state['chain2_3']=None
    elif st.session_state.step_3_choice == "Text 3":
        st.session_state.step_3_choice = st.session_state['chain2_3']
        st.session_state['chain2_1']=None
        st.session_state['chain2_2']=None

st.button(label='Next', on_click=toggle3, disabled=st.session_state.button_3, type='primary', key="button3")


st.divider()

with st.expander("Goals - Methodology - Risks", expanded=st.session_state.button_3):
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

    st.session_state.step_4_choice = st.selectbox("Please choose your favorite generated text",
                                                  options=["Text 1", "Text 2", "Text 3"], key="step4_choice")

    st.write("You have chosen: " + st.session_state.step_4_choice)

    # store the text choice in a variable
    if st.session_state.step_4_choice == "Text 1":
        st.session_state.step_4_choice = st.session_state['chain3_1']
        st.session_state['chain3_2']=None
        st.session_state['chain3_3']=None
    elif st.session_state.step_4_choice == "Text 2":
        st.session_state.step_4_choice = st.session_state['chain3_2']
        st.session_state['chain3_1']=None
        st.session_state['chain3_3']=None
    elif st.session_state.step_4_choice == "Text 3":
        st.session_state.step_4_choice = st.session_state['chain3_3']
        st.session_state['chain3_1']=None
        st.session_state['chain3_2']=None

st.button(label='Next', on_click=toggle4, disabled=st.session_state.button_4, type='primary', key="button4")

st.divider()

with st.expander("Final result", expanded=st.session_state.button_4):
    if st.session_state.step_2_choice and st.session_state.step_3_choice and st.session_state.step_4_choice is not None:
        chat = ChatOpenAI(openai_api_key=OPENAI_API_KEY,
                          temperature=st.session_state.temperature)

        st.session_state.final_system_content = "Put together the following information into a project proposal contract: {input}. \n\n "
        final_system_message_prompt = SystemMessagePromptTemplate.from_template(
            st.session_state.final_system_content
        )
        final_user_template = "\n\n".join(
            [st.session_state.step_2_choice, st.session_state.step_3_choice, st.session_state.step_4_choice])
        final_input_prompt = HumanMessagePromptTemplate.from_template(
            final_user_template
        )

        chat_prompt = ChatPromptTemplate.from_messages(
            [final_system_message_prompt, final_input_prompt]
        )

        st.session_state.parent_chain = LLMChain(llm=chat, prompt=chat_prompt, verbose=True, output_key="Final result")

        st.write(st.session_state.parent_chain.run(input=final_input_prompt))
