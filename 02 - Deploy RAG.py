# Databricks notebook source
# MAGIC %run ./utils/00-init  $reset_all_data=false

# COMMAND ----------

# DBTITLE 1,Test PAT Token
index_name=f"{catalog}.{db}.databricks_documentation_vs_index"
host = "https://" + spark.conf.get("spark.databricks.workspaceUrl")

test_demo_permissions(host, secret_scope="dbsecret", secret_key="rag_sp_token", vs_endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME, index_name=index_name, embedding_endpoint_name="databricks-bge-large-en")

# COMMAND ----------

# url used to send the request to your model from the serverless endpoint
host = "https://" + spark.conf.get("spark.databricks.workspaceUrl")
os.environ['DATABRICKS_TOKEN'] = dbutils.secrets.get("dbsecret", "rag_sp_token")

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
from langchain.vectorstores import DatabricksVectorSearch
from langchain.embeddings import DatabricksEmbeddings

# Test embedding Langchain model
#NOTE: your question embedding model must match the one used in the chunk in the previous model 
embedding_model = DatabricksEmbeddings(endpoint="databricks-bge-large-en")
print(f"Test embeddings: {embedding_model.embed_query('What is Apache Spark?')[:20]}...")

def get_retriever(persist_dir: str = None):
    os.environ["DATABRICKS_HOST"] = host
    #Get the vector search index
    vsc = VectorSearchClient(workspace_url=host, personal_access_token=os.environ["DATABRICKS_TOKEN"])
    vs_index = vsc.get_index(
        endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME,
        index_name=index_name
    )

    # Create the retriever
    vectorstore = DatabricksVectorSearch(
        vs_index, text_column="content", embedding=embedding_model
    )
    return vectorstore.as_retriever()


# test our retriever
vectorstore = get_retriever()
similar_documents = vectorstore.get_relevant_documents("How do I track my Databricks Billing?")
print(f"Relevant documents: {similar_documents[0]}")

# COMMAND ----------

# Test Databricks Foundation LLM model
from langchain.chat_models import ChatDatabricks
chat_model = ChatDatabricks(endpoint="databricks-llama-2-70b-chat", max_tokens = 200)
print(f"Test chat model: {chat_model.predict('What is Apache Spark')}")

# COMMAND ----------

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatDatabricks

TEMPLATE = """You are an assistant for Databricks users. You are answering python, coding, SQL, data engineering, spark, data science, DW and platform, API or infrastructure administration question related to Databricks. If the question is not related to one of these topics, kindly decline to answer. If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep the answer as concise as possible.
Use the following pieces of context to answer the question at the end:
{context}
Question: {question}
Answer:
"""
prompt = PromptTemplate(template=TEMPLATE, input_variables=["context", "question"])

chain = RetrievalQA.from_chain_type(
    llm=chat_model,
    chain_type="stuff",
    retriever=get_retriever(),
    chain_type_kwargs={"prompt": prompt}
)

# COMMAND ----------

# langchain.debug = True #uncomment to see the chain details and the full prompt being sent
question = {"query": "How can I track billing usage on my workspaces?"}
answer = chain.run(question)
print(answer)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #Saving model to Unity Catalog

# COMMAND ----------

from mlflow.models import infer_signature
import mlflow
import langchain

mlflow.set_registry_uri("databricks-uc")
model_name = f"{catalog}.{db}.dbdemos_chatbot_model"

with mlflow.start_run(run_name="dbdemos_chatbot_rag") as run:
    signature = infer_signature(question, answer)
    model_info = mlflow.langchain.log_model(
        chain,
        loader_fn=get_retriever,  # Load the retriever with DATABRICKS_TOKEN env as secret (for authentication).
        artifact_path="chain",
        registered_model_name=model_name,
        pip_requirements=[
            "mlflow==" + mlflow.__version__,
            "langchain==" + langchain.__version__,
            "databricks-vectorsearch",
        ],
        input_example=question,
        signature=signature
    )

# COMMAND ----------

# Create or update serving endpoint
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedModelInput

serving_endpoint_name = "main.rag_chatbot.dbdemos_chatbot_model"
latest_model_version = get_latest_model_version(model_name)

# COMMAND ----------

print(latest_model_version)
print(serving_endpoint_name)

# COMMAND ----------

# DBTITLE 1,Load Registered Model
import mlflow.pyfunc

model_version_uri = "models:/{model_name}/1".format(model_name=serving_endpoint_name)

print("Loading registered model version from URI: '{model_uri}'".format(model_uri=latest_model_version))
model_version_1 = mlflow.pyfunc.load_model(model_version_uri)

model_champion_uri = "models:/{model_name}@Champion".format(model_name=serving_endpoint_name)

print("Loading registered model version from URI: '{model_uri}'".format(model_uri=model_champion_uri))
champion_model = mlflow.pyfunc.load_model(model_champion_uri)

# COMMAND ----------

question = "What is Databricks?"

answer = champion_model.predict(data=[{"query": question}])
print(answer[0])

# COMMAND ----------

display_gradio_app("databricks-demos-chatbot")