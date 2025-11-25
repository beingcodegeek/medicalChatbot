from flask import Flask, render_template, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
from src.prompt import system_prompt
import os

app = Flask(__name__)

# -----------------------------
# ENV & KEYS
# -----------------------------
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENROUTER_API_KEY"] = OPENROUTER_API_KEY

# -----------------------------
# EMBEDDINGS & PINECONE
# -----------------------------
embeddings = download_hugging_face_embeddings()

index_name = "medical-chatbot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings,
)

retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)

# -----------------------------
# LLM (OpenRouter)
# -----------------------------
chatModel = ChatOpenAI(
    model="gpt-4o",                      # or another OpenRouter model
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    max_tokens=1024,                     # avoid 402 errors
    default_headers={
        "HTTP-Referer": "http://localhost",
        "X-Title": "MedicalChatbot",
    },
)

# -----------------------------
# CONVERSATION MEMORY
# -----------------------------
memory = ConversationBufferMemory(
    return_messages=True,
)

# -----------------------------
# 1) History-aware retriever
#    (turns "how to cure it?" into
#     "how to cure acne?" using chat history)
# -----------------------------
contextualize_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Given the chat history and the latest user question, "
            "rewrite the question so it is a standalone query. "
            "Do NOT answer it.",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(
    llm=chatModel,
    retriever=retriever,
    prompt=contextualize_prompt,
)

# -----------------------------
# 2) QA chain that also sees chat history
# -----------------------------
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(chatModel, qa_prompt)

rag_chain = create_retrieval_chain(
    history_aware_retriever,
    question_answer_chain,
)

# -----------------------------
# FLASK ROUTES
# -----------------------------
@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]
    print("User:", msg)

    # current chat history for this session
    chat_history = memory.chat_memory.messages

    # pass both input + history into RAG chain
    response = rag_chain.invoke(
        {
            "input": msg,
            "chat_history": chat_history,
        }
    )

    answer = response["answer"]
    print("Bot:", answer)

    # update memory
    memory.chat_memory.add_user_message(msg)
    memory.chat_memory.add_ai_message(answer)

    return str(answer)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
