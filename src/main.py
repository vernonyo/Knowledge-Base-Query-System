import os
import json
import re
import datetime
import shutil
import faiss
from flask import Flask, render_template, request, jsonify, Response, send_file
from typing_extensions import List
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_core.messages.system import SystemMessage
from langchain_core.messages.ai import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.document_loaders import PyPDFLoader
from langgraph.graph import START, END, MessagesState, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field


cwd = os.path.dirname(os.path.realpath(__file__))

def load_config(file_path):
    with open(file_path, 'r') as file:
        config = json.load(file)
    return config

config = load_config(cwd+'/config.json')

#os.environ["LANGSMITH_TRACING"] = config["api"]["tracing"]
#if not os.environ.get("LANGSMITH_API_KEY"):
    #os.environ["LANGSMITH_API_KEY"] = config["api"]["langsmith"]
os.environ["OLLAMA_MAX_LOADED_MODELS"] = str(config["max models"])


def create_categories(dir_path, name):
    """
    Creates a folder for a new category
    """
    # Create a directory to store the PDF files
    if not os.path.exists(dir_path+"/data/"):
        os.mkdir(dir_path+"/data/")
    if not os.path.exists(dir_path+"/data/"+name+"/"):
        try:
            os.mkdir(dir_path+"/data/"+name+"/")
        except OSError:
            return "Invalid category name"
    else:
        return "Category already exists!"

    return "Category created! Please proceed to upload the content"


def delete_categories(dir_path, name, file):
    """
    Deletes a file OR a category folder including its contents
    """
    file_path = dir_path + "/data/" + name + "/" + file
    print(file_path)

    if not os.path.exists(file_path):
        return "Category doesn't exist!"
    else:
        print("Removing...")
        category = os.listdir(dir_path + "/data/" + name)

        #For deleting a single file
        try:
            loader = PyPDFLoader(file_path)
            for page in loader.lazy_load():
                print(page)
                title = page.metadata['title']
                break
            split = 0
            while True:
                try:
                    print(title+str(split))
                    vector_store.delete(ids=[title+str(split)])
                except ValueError as e:
                    print(e)
                    break
                split += 1
            vector_store.save_local(cwd+"/"+config["vectorstore"]["save"])
            os.remove(file_path)
            return "The file has been deleted"

        #For deleting a category
        except ValueError as e:
            print(f"Error occurred: {e}")

            for f in category:
                #Get name of the PDF
                loader = PyPDFLoader(dir_path + "/data/" + name + "/" + f)
                for page in loader.lazy_load():
                    title = page.metadata["title"]
                    print(page.metadata["title"])
                    break

                #Iterate through and delete all vectors with ID containing PDF title
                vector_id = 0
                while True:
                    try:
                        print("Deleting: " + title+str(vector_id))
                        vector_store.delete(ids=[title+str(vector_id)])
                    except ValueError as e:
                        print(e)
                        break
                    vector_id += 1

            vector_store.save_local(cwd+"/"+config["vectorstore"]["save"])

            #Delete category folder
            shutil.rmtree(file_path)

            return "The category and all its contents have been deleted successfully"


def update_categories(dir_path, name):
    """
    Updates the categories by loading, tokenising and embedding all its contents
    """
    print("Loading data...")
    file_path = dir_path + "/data/" + name
    if not os.path.exists(file_path):
        print("Category doesn't exist! Please make a new one if necessary")
        return "Category doesn't exist! Please make a new one if necessary"
    docs = load(file_path)
    message = embed(docs)
    print("Category updated!")
    return message


def upload_data(dir_path, files, category_name, image):
    """
    Saves the files uploaded by the admin
    """
    if category_name == "Select a category":
        return "Please select a category or create a new one if necessary"
    
    category_directory = dir_path + "/data/" + category_name + "/"

    if not os.path.exists(category_directory):
        return "Category doesn't exist! Please make a new one if necessary"
    
    for file in files:
        if image == "false":
            file.save(os.path.join(category_directory+file.filename))
        else:
            file.save(os.path.join(category_directory+file.filename[:-4]+"IMAGE")+file.filename[-4:])

    return 'File uploaded successfully'


def list_categories():
    """
    Returns the existing list of categories
    """
    print("Listing categories")
    if not os.path.exists("data/"):
        os.mkdir("data/")
        return []

    list_of_categories = os.listdir(cwd+"/data/")
    #print(list_of_categories)
    jsonify(list_of_categories)
    return list_of_categories


def list_category_files(category_name):
    """
    Returns the existing list of files in a category
    """
    print("Listing files")
    if not os.path.exists("data/"):
        os.mkdir("data/")
        return []

    print(cwd + "/data/" + category_name + "/")
    list_of_files = os.listdir(cwd + "/data/" + category_name + "/")
    print(list_of_files)
    print("finish")
    jsonify(list_of_files)
    return list_of_files

#Generative LLM
if config["generative model"]["platform"] == "ollama":
    llm = ChatOllama(
        model = config["generative model"]["model"],
        temperature = 0,
        keep_alive=config["sleep timer"],
        num_gpu=config["number of GPU"]
    )
else:
    llm = "" #New model code here

#Embeddings model
if config["embeddings"]["platform"] == "ollama":
    embeddings = OllamaEmbeddings(
        model=config["embeddings"]["model"],
    )
else:
    embeddings = "" # New embeddings code here

#Vector Store
index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))
#Try to load existing vector store
try:
    vector_store = FAISS.load_local(
        cwd+"/"+config["vectorstore"]["save"], embeddings, allow_dangerous_deserialization=True
    )
#Create new vector store
except Exception as e:
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

#Text splitter/tokeniser
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)


def load(dir_path:str):
    """
    Loads the PDFs and converts them to text
    """
    docs = []
    directory = os.listdir(dir_path+"/")
    for file in directory:
        loader = PyPDFLoader(dir_path + "/" + file)
        page_number = 0
        for page in loader.lazy_load():
            page_number += 1
            docs.append(page)
    return docs


def embed(docs):
    """
    Tokenises and embeds the data as vectors then stores them in the respective vector store
    """
    print("Embedding...")
    id_list = []
    all_splits = text_splitter.split_documents(docs)

    #for doc in range(0, len(docs)):
    #    print(type(docs[doc]))
    #    splits = text_splitter.split_documents(docs[doc])
    #    all_splits.append(splits)
    #    split_number = 0
    #    for split in splits:
    #        split_number += 1
    #        id_list.append(ids[doc]+split_number)
    #print(id_list)

    title = ""
    for split in all_splits:
        next_title = split.metadata["title"]
        if title == "":
            title = next_title
            vector_id = 0
        #print(title)
        if title == next_title:
            #print("Next chunk")
            #print(title + str(vector_id))
            id_list.append(title + str(vector_id))
        else:
            #print("New doc")
            vector_id = 0
            title = next_title
            #print(title + str(vector_id))
            id_list.append(title + str(vector_id))
        vector_id += 1
    print(id_list)

    # Index chunks
    try:
        _ = vector_store.add_documents(documents=all_splits, ids=id_list)
    except ValueError as e:
        print(e)
        return "ValueError!"

    vector_store.save_local(cwd+"/"+config["vectorstore"]["save"])

    return "Category updated!"


# Define state for application
class State(MessagesState):
    """
    State of RAG system

    original_question: Query sent by user

    question: Query sent to generation model

    context: Context retrieved

    times_graded
    """
    original_question: str
    question: str
    context: List[Document]
    times_graded: int
    response: str


# Define application steps
def retrieve(state: State):
    """
    Retrieve relevant context documents, based on similarity search between vectors
    """
    print("---RETRIEVE---")
    retrieved_docs = vector_store.similarity_search(state["question"])
    #print(retrieved_docs)
    #Truncate to the first 2 documents
    max_docs = config["generative model"]["max docs"]
    #print(retrieved_docs[0])
    return {"context": retrieved_docs[:max_docs-1]}


class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


# LLM tool function call
if config["tool model"]["platform"] == "ollama":
    tool_llm = ChatOllama(
        model = config["tool model"]["model"],
        temperature = 0,
        keep_alive="10m"
    )
else:
    tool_llm = "" # Tool_llm code goes here
structured_llm_grader = tool_llm.with_structured_output(GradeDocuments)

# Prompt
system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    Identify the keywords in the question. 
    If the document contains keyword(s) **or** semantic meaning related to the question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader


def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question using retrieval grader tool.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """
    #Check if max limit of number of loops has been reached. If yes, mark as "End" to end loop
    times_graded = state["times_graded"]
    times_graded += 1
    print(times_graded)
    if times_graded > config["generative model"]["max loops"]:
        rewrite = "End"
        return {"rewrite": rewrite, "times_graded": times_graded}

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["context"]

    # Score each doc
    filtered_docs = []
    rewrite = "No"
    for doc in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": doc.page_content, "times_graded": times_graded}
        )
        grade = score.binary_score
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(doc)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            rewrite = "Yes"
            continue

    return {"context": filtered_docs, "question": question, "rewrite": rewrite, "times_graded": times_graded}


def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    rewrite = state["rewrite"]

    if rewrite == "Yes":
        # All documents have been filtered check_relevance
        # We will re-write the question
        print(
            "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, REWRITE---"
        )
        return "transform_query"
    elif rewrite == "No":
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"
    else:
        return "end"


#Query rewriting tool to aid context retrieval
system = """You are a question re-writer that converts an input question to a better version that is optimized \n 
     for vectorstore retrieval. Look at the initial question and try to reason about the underlying semantic intent / meaning.
     Do not provide an explanation."""
re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "Here is the initial question: \n\n {question} \n "
            "Formulate an improved question:",
        ),
    ]
)
question_rewriter = re_write_prompt | tool_llm | StrOutputParser()


def transform_query(state):
    """
    Transform the query to produce a better question using question rewriter tool.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """

    print("---TRANSFORM QUERY---")
    question = state["question"]
    documents = state["context"]

    # Re-write question
    better_question = question_rewriter.invoke({"question": question})
    #print(better_question)
    return {"context": documents, "question": better_question}


def generate(state: State):
    """
    Generate response to query based on retrieved context and query.
    """
    print("---GENERATE OUTPUT---")
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])

    messages = SystemMessage(
        '''You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question.
        Use three sentences maximum and keep the answer concise and written in a professional tone.'''
#        If markdown formatting is used, change to html formatting.
        f'''Context: {docs_content}
        Answer:''')
    prompt = [messages] + state["messages"]
    print(prompt)
    response = llm.invoke(prompt)

    return {"response": response.content}

class GradeResponse(BaseModel):
    """Binary score for relevance check on response."""

    binary_score: str = Field(
        description="Response is relevant to the question, 'yes' or 'no'"
    )

structured_response_grader = tool_llm.with_structured_output(GradeResponse)

# Prompt
system = """You are a grader assessing the response to a user question. \n 
    Identify the keywords in the question. 
    If the response contains keyword(s) **or** semantic meaning related to the question, grade it as 'yes'."""
response_grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Response: \n\n {document} \n\n User question: {question}"),
    ]
)

response_grader = response_grade_prompt | structured_response_grader


def grade_response(state:State):
    """
    Grades the response of the LLM
    """
    print("---CHECK RESPONSE RELEVANCE TO QUESTION---")
    question = state["question"]
    response = state["response"]
    date = datetime.date.today().isoformat()
    dir_path = cwd[:-4]

    score = response_grader.invoke(
        {"question": question, "document": response}
    )
    grade = score.binary_score
    if grade == "yes":
        print("---GRADE: RESPONSE RELEVANT---")
        message = response
        messages = SystemMessage(
            '''Please repeat the following, verbatim, without quotations:''')
        prompt = [messages] + [response]
        print(prompt)
        message = llm.invoke(prompt)
    else:
        print("---GRADE: RESPONSE NOT RELEVANT---")
        print(response)
        to_save = "\n" + date + ":: " + question
        if question == "Hello world":
            message = llm.invoke(f"Please repeat the following statement only, without quotations: Hello")
            return
        try:
            with open(dir_path + '/Unanswered_queries/unanswered_queries.txt', "a", encoding="UTF-8") as f:
                f.write(to_save)
        except FileNotFoundError as e:
            print(e)
            to_save = date + ":: " + question
            with open(dir_path + '/Unanswered_queries/unanswered_queries.txt', "w", encoding="UTF-8") as f:
                f.write(to_save)
        prompt = config['message']['fail']
        message = llm.invoke(f"Please repeat the following statement only, without quotations: {prompt}")

    return {"messages": message}


def add_failed_query(state: State):
    print("SAVING UNANSWERED QUERY")
    date = datetime.date.today().isoformat()
    query = state["original_question"]
    dir_path = cwd[:-4]
    to_save = "\n" + date + ":: " + query
    if query == "Hello world":
        response = llm.invoke(f"Please repeat the following statement only, without quotations: Hello")
        return
    try:
        with open(dir_path + '/Unanswered_queries/unanswered_queries.txt', "a", encoding="UTF-8") as f:
            f.write(to_save)
    except FileNotFoundError as e:
        print(e)
        to_save = date + ":: " + query
        with open(dir_path + '/Unanswered_queries/unanswered_queries.txt', "w", encoding="UTF-8") as f:
            f.write(to_save)
    prompt = config['message']['fail']
    response = llm.invoke(f"Please repeat the following statement only, without quotations: {prompt}")
    return {"messages": response}


# Compile application
graph_builder = StateGraph(State)
graph_builder.add_node("retrieve", retrieve)
graph_builder.add_node("grade_documents", grade_documents)
graph_builder.add_node("transform_query", transform_query)  # transform_query
graph_builder.add_node("generate", generate)


if config["grade"] == "yes":
    graph_builder.add_node("failed", add_failed_query)
    graph_builder.add_edge(START, "retrieve")
    graph_builder.add_edge("retrieve", "grade_documents")
    graph_builder.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "transform_query": "transform_query",
            "generate": "generate",
            "end": "failed"
        },
    )
    graph_builder.add_edge("transform_query", "retrieve")
    graph_builder.add_edge("generate", END)
    graph_builder.add_edge("failed", END)
else:
    graph_builder.add_node("failed", grade_response)
    graph_builder.add_edge(START, "retrieve")
    graph_builder.add_edge("retrieve", "generate")
    graph_builder.add_edge("generate", "failed")
    graph_builder.add_edge("failed", END)

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

thread_number = 0

# For visualising the graph
#from IPython.display import Image, display
#from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles
#
#img = Image(
#    graph.get_graph().draw_mermaid_png(
#        draw_method=MermaidDrawMethod.API,
#    )
#)
#b = img.data
#with open(cwd+"Graph.png", "wb") as png:
#    png.write(b)

#Pre-loading the LLM
if config["preload"]=="yes":
    print("Pre-load...")
    graph.invoke({"question": "Hello world", 
                  "messages": [{"role": "user", "content": "Hello world"}], 
                  "original_question": "Hello world", 
                  "times_graded": 1}, 
                  config = {"configurable": {"thread_id": "default"}})

# Webpage and related functions
app = Flask(__name__)
# home route that returns below text when root url is accessed
@app.route("/")
def qa_page():
    global ran
    ran = 0
    return render_template('QA.html')

# Route for admin url
@app.route("/admin/")
def admin_page():
    return render_template('Admin.html')


@app.route('/run', methods=['GET', 'POST'])
def run():
    print("Begin")
    print(thread_number)
    thread_id = str(thread_number)
    thread_config = {"configurable": {"thread_id": thread_id}}
    query = request.get_json().get('message', '')
    print(query)
    print("Next")
    global ran
    ran = 1
    def generate_json(query):
        c=0
        brackets = 0
        try:
            for name, data in graph.stream(
                {
                    "question": query, 
                    "messages": [{"role": "user", "content": query}], 
                    "original_question": query, 
                    "times_graded": 0
                },
                config = thread_config, 
                stream_mode=["messages", "values"]
            ):
                #print(msg.content)
                #print(type(data))
                #print(data)
                if name == "values": #Context source retrieval
                    c=c+1
                    #print(name)
                    if c==2:
                        global context
                        try:
                            context_path = data['context'][0].metadata['source']
                        except IndexError:
                            yield "<h1> Warning: There is no content in the system! Please contact the system administrator for assistance!"
                            return
                        print(context_path)
                        context_link = re.sub(" ", "%20", context_path)
                        context_name = data['context'][0].metadata['title']
                        #context_element = "<a href="+context_link+f">{context_name}</a>"
                        context = context_path, context_name
                        #print(context_element)
                    continue
                else:
                    #Debug which step the graph is on, can be done via LangGraph's native methods
                    #if type(data) == tuple:
                    #    print(" "+data[1]['langgraph_node'])
                    #Truncates the output of DeepSeek R1 responses
                    if "deepseek-r1" in config["generative model"]["model"]:
                        if "think>" in data[0].content:
                            brackets += 1
                        if brackets == 2 or (data[1]['langgraph_node'] == "failed"):
                            yield data[0].content
                    #Sends response for other models
                    elif config["grade"] == "yes":
                        if (data[1]['langgraph_node'] == "failed") or (data[1]['langgraph_node'] == "generate"):
                            yield data[0].content
                        print(data[0].content, end="")
                        #yield data[0].content
                    else:
                        if (data[1]['langgraph_node'] == "failed"):
                            yield data[0].content
                        print(data[0].content, end="")
        except Exception as e:
            print(e)
            return "Error! Please inform the system administrator for assistance!"
        print("\nEND")
        return context

    return Response(
        generate_json(query), mimetype="text/event-stream"
    )


@app.route('/source', methods=['GET'])
def source():
    print("Returning source")
    #global context
    #return "Source: " + context
    try:
        global context
        #global context_name
        global ran
        if ran == 1:
            context_path = context[0].replace('\\','/')
            #edited_path = re.search('(.+/src)(/data/.+)', context_path)
            #full_path = os.path.join(app.root_path, edited_path.group(2))
            return send_file(context_path)
        else:
            return "Please send a query before attempting to cite source."
    except NameError:
        return "<h1> Warning: There is no content in the system! Please contact the system administrator for assistance!"


@app.route('/sourcetitle', methods=['GET'])
def source_title():
    print("Returning source")
    try:
        global context
        global ran
        if ran == 1:
            return "Source: " + context[1]
        else:
            return "<h3> Please send a query before attempting to cite source."
    except NameError:
        return "<h1> Warning: There is no content in the system! Please contact the system administrator for assistance!"


@app.route('/new', methods=['GET'])
def new_thread():
    print("Refreshing...")
    global thread_number
    thread_number = thread_number + 1
    thread_id = str(thread_number)
    print(thread_id)
    thread_config = {"configurable": {"thread_id": thread_id}}
    graph.update_state(thread_config, {"messages": []})
    return []


@app.route('/create', methods=['POST'])
def create_cat():
    print("Create")
    name = request.get_json().get('name', '')
    message = create_categories(cwd, name)
    return Response(
        message, mimetype="text/event-stream"
    )

@app.route('/list', methods=['GET'])
def list_cat():
    #print("List")
    list_of_cat = list_categories()
    print(list_of_cat)
    return jsonify(list_of_cat)

@app.route('/files', methods=['POST'])
def list_files():
    print("List files")
    name = request.get_json().get('name', '')
    list_of_files = list_category_files(name)
    print(list_of_files)
    return jsonify(list_of_files)

@app.route('/delete', methods=['POST'])
def delete_cat():
    print(request.get_json())
    name = request.get_json().get('name', '')
    file = request.get_json().get('f', '')
    print(file)
    if file == "All":
        file = ""
    print("Remove: " + file + " from " + name)
    message = delete_categories(cwd, name, file)
    return Response(
        message, mimetype="text/event-stream"
    )


@app.route('/upload', methods=['POST'])
def upload():
    name = request.form['name']
    image = request.form['image']
    print(image)
    print(name)
    files = request.files.getlist('file')
    for file in files:
        if not(file.filename.rsplit('.', 1)[1].lower() == "pdf"):
            print("Not PDF file")
            return "Please submit only PDF files"
    print(files)
    print('File')

    if 'file' not in request.files:
        return 'Please select a file'

    file = files[0]
    if file.filename == '':
        return 'No selected file'

    message = upload_data(cwd, files, name, image)
    update_categories(cwd, name)
    return message

# DEPRECATED

#@app.route('/update', methods=['POST'])
#def update():
#    name = request.get_json().get('name', '')
#    print("Update: " + name)
#    update_categories(cwd, name)
#    return Response(
#        "Category Updated!", mimetype="text/event-stream"
#    )


@app.route('/email', methods=['POST'])
def save_email():
    """
    Saving email assigned by admin to config file
    """

    email = request.get_json().get('email', '')
    match = re.match(r"^[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)*$", email)
    if match:
        print("Saving")
    else:
        return "Please enter a valid email address"
    
    print("Email: " + email)
    with open(cwd+'/config.json', 'r') as f:
        config = json.load(f)
        f.close()

    # Edit the data
    config['email'] = email

    # Write it back to the file
    with open(cwd+'/config.json', 'w') as f:
        config_save = json.dumps(config, indent=4)
        f.write(config_save)
        f.close()

    return Response(
        "Email saved!", mimetype="text/event-stream"
    )


if __name__ == '__main__':

    app.run()
