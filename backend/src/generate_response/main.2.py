import os, json
import boto3
from aws_lambda_powertools import Logger
from langchain.llms.bedrock import Bedrock
from langchain.memory.chat_message_histories import DynamoDBChatMessageHistory
from langchain.memory import ConversationBufferMemory
# from langchain.embeddings import BedrockEmbeddings
from langchain_community.embeddings import BedrockEmbeddings
# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import BedrockChat
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import ConversationChain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate


MEMORY_TABLE = os.environ["MEMORY_TABLE"]
BUCKET = os.environ["BUCKET"]


s3 = boto3.client("s3")
logger = Logger()

def get_bedrock_client(region):
    bedrock_client = boto3.client("bedrock-runtime", region_name=region)
    return bedrock_client
    
def create_langchain_vector_embedding_using_bedrock(bedrock_client, bedrock_embedding_model_id):
    bedrock_embeddings_client = BedrockEmbeddings(
        client=bedrock_client,
        model_id=bedrock_embedding_model_id)
    return bedrock_embeddings_client
    
def create_bedrock_llm(bedrock_client, model_version_id):
    bedrock_llm = BedrockChat(
        model_id=model_version_id, 
        client=bedrock_client,
        model_kwargs={'temperature': 0}
        )
    return bedrock_llm


@logger.inject_lambda_context(log_event=True)
def lambda_handler(event, context):
    logger.info("*********** 22")
    logger.info(event)
    event_body = json.loads(event["body"])
    logger.info("*********** 25")
    logger.info("*********** 26",event_body)
    file_name = event_body["fileName"]
    logger.info("*********** 28")
    logger.info(file_name)
    human_input = event_body["prompt"]
    logger.info("*********** 31")
    logger.info(human_input)
    conversation_id = event["pathParameters"]["conversationid"]
    logger.info("*********** 33")
    logger.info(conversation_id)

    user = event["requestContext"]["authorizer"]["claims"]["sub"]
    
    logger.info("*********** 40")
    logger.info(user)
    
    logger.info("*********** 45")
    logger.info(BUCKET)
    logger.info("*********** 47")
    logger.info(f"{user}/{file_name}/index.faiss")
    
    s3.download_file(BUCKET, f"{user}/{file_name}/index.faiss", "/tmp/index.faiss")
    s3.download_file(BUCKET, f"{user}/{file_name}/index.pkl", "/tmp/index.pkl")
    
    bedrock_model_id = "amazon.titan-embed-text-v1"
    bedrock_embedding_model_id = "amazon.titan-text-express-v1"
    region = "us-east-1"
    
    # Creating all clients for chain
    bedrock_client = get_bedrock_client(region)
    bedrock_llm = create_bedrock_llm(bedrock_client, bedrock_model_id)
    bedrock_embeddings_client = create_langchain_vector_embedding_using_bedrock(bedrock_client, bedrock_embedding_model_id)
    
    logger.info("*********** 59")
    logger.info(bedrock_client)

    faiss_index = FAISS.load_local("/tmp", bedrock_embeddings_client, allow_dangerous_deserialization=True)
    
    logger.info("*********** 71")
    logger.info(bedrock_embeddings_client)
    logger.info("*********** 73")
    logger.info(bedrock_llm)
    logger.info("*********** 75")
    logger.info(faiss_index)

    message_history = DynamoDBChatMessageHistory(
        table_name=MEMORY_TABLE, session_id=conversation_id
    )
    logger.info("*********** 81")
    logger.info(message_history)

    # memory = ConversationBufferMemory(
    #     memory_key="chat_history",
    #     chat_memory=message_history,
    #     input_key="question",
    #     output_key="answer",
    #     return_messages=True,
    # )
    # logger.info("*********** 91.12345")
    # logger.info(memory)
    
    # chain = ConversationalRetrievalChain.from_llm(
    #     llm=llm,
    #     memory=memory,
    #     chain_type="stuff",
    #     retriever=faiss_index.as_retriever(),
    #     return_source_documents=True,
    #     get_chat_history=lambda h : h,
    #     verbose=False)

    # response = chain({"question": "How are you today?"})
    # logger.info("*********** 92")
    # logger.info(response)

    # response = chain({"question": "Can you help me understand ESG?"})
    # logger.info("*********** 93")
    # logger.info(response)

    # res = chain({"question": "What is ARC's potential hurt approach?"})
    # logger.info("*********** 94")
    # logger.info(res)

# ORIGINAL CODE
    # qa = ConversationalRetrievalChain.from_llm(
    #     llm=llm,
    #     retriever=faiss_index.as_retriever(),
    #     memory=memory,
    #     return_source_documents=True,
    # )
    # logger.info("*********** 100")
    # logger.info(qa)

    # res = qa({"question": human_input})

    # logger.info("*********** 105")
    # logger.info(res)
    
    logger.info("*********** 82")
    
    # LangChain prompt template
    prompt = ChatPromptTemplate.from_template("""System: The following is a friendly conversation between a knowledgeable helpful assistant and a customer.
    The assistant is talkative and provides lots of specific details from it's context.

    Current conversation:
    {context}

    User: {input}
    Bot:""")
    
    logger.info("*********** 83")
    docs_chain = create_stuff_documents_chain(bedrock_llm, prompt)
    logger.info("*********** 84")
    retrieval_chain = create_retrieval_chain(
        retriever=faiss_index.as_retriever(),
        combine_docs_chain = docs_chain
    )
    logger.info("*********** 85")
    
    logger.info(f"Invoking the chain with similarity using FAISS, Bedrock FM {bedrock_model_id}, and Bedrock embeddings with {bedrock_embedding_model_id}")
    logger.info("*********** 86")
    # response = retrieval_chain.invoke({"context":message_history,"input": human_input})
    response = retrieval_chain.invoke(
        {"context":message_history,"input": "what is the name of the medicine?"},
        config={
            "configurable": {"session_id": "abc123"}
        },  # constructs a key "abc123" in `store`.
    )

    logger.info("*********** 87")
    logger.info("These are the similar documents from FAISS based on the provided query:")
    source_documents = response.get('context')
    logger.info("*********** 88")
    for d in source_documents:
        print("")
        logger.info(f"Text: {d.page_content}")
    
    logger.info("*********** 89")
    logger.info(f"The answer from Bedrock {bedrock_model_id} is: {response.get('answer')}")
    logger.info("*********** 90")
    logger.info(response)


    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "*",
        },
        "body": json.dumps(response["answer"]),
    }
