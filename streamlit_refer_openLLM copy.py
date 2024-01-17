import streamlit as st
import time

from loguru import logger

from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

# 참조 데이터(PDF, Dox, PPT) 로딩을 위한 라이브러리
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import UnstructuredPowerPointLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

# 몇개까지의 대화를 메모리에 넣어줄지 설정
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS

# from streamlit_chat import message (메모리 구축을 위해 필요한 추가 라이브러리)
from langchain.callbacks import get_openai_callback
from langchain.memory import StreamlitChatMessageHistory

# 트랜스포머의 BitsandBytesConfig를 통해 양자화 매개변수 정의
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# API가 아닌 HuggingFace의 오픈소스 LLM을 가져다 쓰기 때문에 필요한 라이브러리 [아래 2가지]
from langchain.llms import HuggingFaceHub

# from langchain.llms import HuggingFacePipeline
# from langchain.chains import LLMChain
# from langchain.prompts import PromptTemplate

def main():
    st.set_page_config(
    page_title="DirChat",
    page_icon=":books:",
    )

    st.title("_Private Data :red[QA Chat]_ :books:")        #_: 이텔릭체, red[] : 빨간색

    # converstaion, chat_history, processComplete 변수 정의
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    # with 구문 : 구성 요소에 딸린 요소를 정하기 위한거
    with st.sidebar:
        uploaded_files =  st.file_uploader("Upload your file", type=['pdf','docx','pptx'], accept_multiple_files=True)
        process = st.button("Process")

    if process:     # 만약 Process 버튼이 눌리면        
        files_text = get_text(uploaded_files)
        logger.debug("get_text")
        text_chunks = get_text_chunks(files_text)
        logger.debug("get_text_chunks")
        vetorestore = get_vectorstore(text_chunks)
        logger.debug("get_vectorstore")
     
        st.session_state.conversation = get_conversation_chain(vetorestore)

        st.session_state.processComplete = True

    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant", 
                                        "content": "안녕하세요! 주어진 문서에 대해 궁금하신 것이 있으면 언제든 물어봐주세요!"}]

    # 메시지마다 with로 묶어줌 role에 따라서 markdown : 이미지(아이콘)과 메시지를 묶어서 추가
    # for : 메시지가 올라올때마다 작동
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 메모리에 저장하여 기억한뒤에 답변
    history = StreamlitChatMessageHistory(key="chat_messages")

    # Chat logic
    if query := st.chat_input("질문을 입력해주세요."):      # 사용자가 질문을 입력하면
        st.session_state.messages.append({"role": "user", "content": query})

        # 사용자 질문
        with st.chat_message("user"):
            st.markdown(query)

        # Assistant 답변
        with st.chat_message("assistant"):
            chain = st.session_state.conversation
            
            with st.spinner("Thinking..."):     # 로딩시 progress를 표시하는 UI                
                logger.debug("asked..")
                result = chain({"question": query})
                logger.debug("results:{}".format(result['answer']))
                # time.sleep(3)
                # with get_openai_callback() as cb:
                # st.session_state.chat_history = result['chat_history']
                logger.debug("get chat_history")
                response = result['answer']
                source_documents = result['source_documents']

                logger.debug("get answer, source_documents")

                st.markdown(response)
                with st.expander("참고 문서 확인"):
                    st.markdown(source_documents[0].metadata['source'], help = source_documents[0].page_content)    # help : ? 부분에 마우스를 올리면 text 띄움
                    st.markdown(source_documents[1].metadata['source'], help = source_documents[1].page_content)
                    st.markdown(source_documents[2].metadata['source'], help = source_documents[2].page_content)

                logger.debug("Show answer and set source_documents")

        # Add assistant message to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

def get_text(docs):
    doc_list = []
    
    for doc in docs:
        file_name = doc.name  # doc 객체의 이름을 파일 이름(임시 저장 경로)으로 사용
        with open(file_name, "wb") as file:  # 파일을 doc.name으로 저장
            file.write(doc.getvalue())
            logger.info(f"Uploaded {file_name}")

        if '.pdf' in doc.name:
            loader = PyPDFLoader(file_name)
            documents = loader.load_and_split()
        elif '.docx' in doc.name:
            loader = Docx2txtLoader(file_name)
            documents = loader.load_and_split()
        elif '.pptx' in doc.name:
            loader = UnstructuredPowerPointLoader(file_name)
            documents = loader.load_and_split()

        doc_list.extend(documents)
    return doc_list

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks = text_splitter.split_documents(text)
    return chunks

def get_vectorstore(text_chunks):
    model_name = "jhgan/ko-sbert-nli"   # 한국어 임베딩 모델 로딩
    encode_kwargs = {'normalize_embeddings': True}    # 임베딩을 통해 원하는 근거 자료를 찾는 retriver 역할을 하기 위해 정규화를 켜야함
    embeddings = HuggingFaceEmbeddings(
                                        model_name=model_name,
                                        encode_kwargs=encode_kwargs
                                        )  
    vectordb = FAISS.from_documents(text_chunks, embeddings)    
    return vectordb

def get_conversation_chain(vetorestore):
    # # hugging face에서 기학습 모델 로딩
    # model_id = "kyujinpy/Ko-PlatYi-6B"
    # logger.debug("Start HF-LLM tokenizer and model")
    # model = AutoModelForCausalLM.from_pretrained(model_id)
    # logger.debug("load HF-LLM model")
    # tokenizer = AutoTokenizer.from_pretrained(model_id)
    # logger.debug("Get HF-LLM tokenizer")    

    # # 모델을 통해 질문을 보내고 답변 받는 과정 (파이프라인)
    # text_generation_pipeline = pipeline(
    #     model=model,
    #     tokenizer=tokenizer,
    #     task="text-generation",
    #     temperature=0, # 답변 샘플링에 있어서 [0] 일관성 있는 대답(사실적 묘사), [1] 다양한 대답(창의적 답변)
    #     return_full_text=True,
    #     max_new_tokens=300,
    # )

    # # bank knowledge
    # prompt_template = """
    # ### [INST]
    # Instruction: Answer the question based on your knowledge.
    # Here is context to help:

    # {context}

    # ### QUESTION:
    # {question}

    # [/INST]
    # """

    # koplatyi_llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

    # # Create prompt from prompt template
    # prompt = PromptTemplate(
    #     input_variables=["context", "question"],
    #     template=prompt_template,
    # )

    # # Create llm chain    
    # llm_chain = LLMChain(llm=koplatyi_llm, prompt=prompt)
    # logger.debug("Set HF-LLM model")

    # Create llm chain    
    llm_chain = HuggingFaceHub(repo_id="kyujinpy/Ko-PlatYi-6B", model_kwargs={"temperature":0, "max_length":300})
    logger.debug("Load HF-LLM model")

    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')
    conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm_chain, 
            chain_type="stuff", 
            retriever=vetorestore.as_retriever(search_type='mmr', vervose=True), 
            memory=memory,  # chat_history 저장, 답변에 해당하는 부분만 히스토리에 닮도록 설정
            get_chat_history=lambda h: h,   # 들어온 그대로 히스토리에 넣도록 설정
            return_source_documents=True,   # LLM이 참고한 문서를 출력하도록 설정
            verbose=True
        )

    logger.debug("Set conversation_chain")
    
    return conversation_chain

# def get_conversation_chain(vetorestore, openai_api_key):
#     llm = ChatOpenAI(openai_api_key=openai_api_key, model_name='gpt-3.5-turbo', temperature=0)
#     conversation_chain = ConversationalRetrievalChain.from_llm(
#             llm=llm, 
#             chain_type="stuff", 
#             retriever=vetorestore.as_retriever(search_type = 'mmr', vervose = True), 
#             memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),  # chat_history 저장, 답변에 해당하는 부분만 히스토리에 닮도록 설정
#             get_chat_history=lambda h: h,   # 들어온 그대로 히스토리에 넣도록 설정
#             return_source_documents=True,   # LLM이 참고한 문서를 출력하도록 설정
#             verbose=True
#         )
#     return conversation_chain

if __name__ == '__main__':
    main()
