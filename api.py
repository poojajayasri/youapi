from flask import Flask, request, jsonify
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import find_dotenv, load_dotenv
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import textwrap
from langchain.document_loaders import YoutubeLoader

app = Flask(__name__)

# Initialize OpenAIEmbeddings
load_dotenv(find_dotenv())
embeddings = OpenAIEmbeddings()

def create_db_from_youtube_video_url(video_url):
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)

    db = FAISS.from_documents(docs, embeddings)
    return db


def get_response_from_query(db, query, k=4):
    """
    gpt-3.5-turbo can handle up to 4097 tokens. Setting the chunksize to 1000 and k to 4 maximizes
    the number of tokens to analyze.
    """

    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2)

    # Template to use for the system message prompt
    template = """
        You are a helpful assistant that that can provide timestamps for the most motivational part of the video {docs}
        the human specifies the no. of seconds of the video
        
        If you feel like you don't have enough information, try to provide an answer based on the closest possible possibility wrt {docs}
        
        Your answers should be verbose and detailed.
        """

    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    # Human question prompt
    human_template = "Answer the following question: {question}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain = LLMChain(llm=chat, prompt=chat_prompt)

    response = chain.run(question=query, docs=docs_page_content)
    response = response.replace("\n", "")
    return response, docs



@app.route('/api/query', methods=['POST'])
def query_handler():
    # Get the request data
    data = request.get_json()

    # Extract video URL and query from the request
    video_url = data.get('video_url')
    query = data.get('query')

    # Create the database from the YouTube video URL
    db = create_db_from_youtube_video_url(video_url)

    # Get the response and docs from the query
    response, docs = get_response_from_query(db, query)

    # Wrap the response in 50 characters per line
    wrapped_response = textwrap.fill(response, width=50)

    # Prepare the JSON response
    response_data = {
        'response': wrapped_response,
        'docs': [doc.page_content for doc in docs]
    }

    return jsonify(response_data)

if __name__ == '__main__':
    app.run()
