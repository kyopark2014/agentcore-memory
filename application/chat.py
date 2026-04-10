import traceback
import boto3
import os
import json
import re
import uuid
import base64
import info 
import utils
import langgraph_agent
import mcp_config
import csv
import PyPDF2
from langchain_core.documents import Document
from urllib import parse

from io import BytesIO
from PIL import Image
from langchain_aws import ChatBedrock
from botocore.config import Config
from botocore.exceptions import ClientError
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, AIMessageChunk
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

import logging
import sys

logging.basicConfig(
    level=logging.INFO,  # Default to INFO level
    format='%(filename)s:%(lineno)d | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger("chat")

workingDir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(workingDir, "config.json")

# Simple memory class to replace ConversationBufferWindowMemory
class SimpleMemory:
    def __init__(self, k=5):
        self.k = k
        self.chat_memory = SimpleChatMemory()
    
    def load_memory_variables(self, inputs):
        return {"chat_history": self.chat_memory.messages[-self.k:] if len(self.chat_memory.messages) > self.k else self.chat_memory.messages}

class SimpleChatMemory:
    def __init__(self):
        self.messages = []
    
    def add_user_message(self, message):
        self.messages.append(HumanMessage(content=message))
    
    def add_ai_message(self, message):
        self.messages.append(AIMessage(content=message))
    
    def clear(self):
        self.messages = []

reasoning_mode = 'Disable'
debug_messages = []  # List to store debug messages

config = utils.load_config()
print(f"config: {config}")

projectName = config.get("projectName", "es")
bedrock_region = config.get("region", "ap-northeast-2")

accountId = config.get("accountId")
knowledge_base_name = config.get("knowledge_base_name")
s3_bucket = config.get("s3_bucket")
s3_prefix = "docs"
s3_image_prefix = "images"

path = config.get('sharing_url', '')
doc_prefix = "docs/"

MSG_LENGTH = 100    

model_name = "Claude 4 Sonnet"
model_type = "claude"
models = info.get_model_info(model_name)
number_of_models = len(models)
model_id = models[0]["model_id"]
debug_mode = "Enable"

reasoning_mode = 'Disable'
user_id = "mcp"
multi_region = 'Disable'

def update(modelName, debugMode, reasoningMode, memoryMode):    
    global model_name, model_id, model_type, debug_mode, reasoning_mode, enable_memory
    global models, user_id

    if model_name != modelName:
        model_name = modelName
        logger.info(f"model_name: {model_name}")
        
        models = info.get_model_info(model_name)
        model_id = models[0]["model_id"]
        model_type = models[0]["model_type"]
                                
    if debug_mode != debugMode:
        debug_mode = debugMode        
        logger.info(f"debug_mode: {debug_mode}")

    if reasoning_mode != reasoningMode:
        reasoning_mode = reasoningMode
        logger.info(f"reasoning_mode: {reasoning_mode}")    

    if enable_memory != memoryMode:
        enable_memory = memoryMode
        logger.info(f"enable_memory: {enable_memory}")

map_chain = dict() 
checkpointers = dict() 
memorystores = dict() 

memory_chain = None
checkpointer = MemorySaver()
memorystore = InMemoryStore()

def initiate():
    global memory_chain, checkpointer, memorystore, checkpointers, memorystores

    if user_id in map_chain:  
        logger.info(f"memory exist. reuse it!")
        memory_chain = map_chain[user_id]

        checkpointer = checkpointers[user_id]
        memorystore = memorystores[user_id]
    else: 
        logger.info(f"memory not exist. create new memory!")
        memory_chain = SimpleMemory(k=5)
        map_chain[user_id] = memory_chain

        checkpointer = MemorySaver()
        memorystore = InMemoryStore()

        checkpointers[user_id] = checkpointer
        memorystores[user_id] = memorystore

def clear_chat_history():
    global memory_chain
    # Initialize memory_chain if it doesn't exist
    if memory_chain is None:
        initiate()
    
    if memory_chain and hasattr(memory_chain, 'chat_memory'):
        memory_chain.chat_memory.clear()
    else:
        memory_chain = SimpleMemory(k=5)
    map_chain[user_id] = memory_chain

def save_chat_history(text, msg):
    global memory_chain
    # Initialize memory_chain if it doesn't exist
    if memory_chain is None:
        initiate()
    
    if memory_chain and hasattr(memory_chain, 'chat_memory'):
        memory_chain.chat_memory.add_user_message(text)
        if len(msg) > MSG_LENGTH:
            memory_chain.chat_memory.add_ai_message(msg[:MSG_LENGTH])                          
        else:
            memory_chain.chat_memory.add_ai_message(msg) 

selected_chat = 0
def get_max_output_tokens(model_id: str = "") -> int:
    """Return the max output tokens based on the model ID."""
    if "claude-4" in model_id or "claude-sonnet-4" in model_id or "claude-opus-4" in model_id or "claude-haiku-4" in model_id:
        return 16384
    return 8192

def get_chat(extended_thinking):
    global selected_chat, model_type

    logger.info(f"models: {models}")
    logger.info(f"selected_chat: {selected_chat}")
    
    profile = models[selected_chat]
    # print('profile: ', profile)
        
    bedrock_region =  profile['bedrock_region']
    modelId = profile['model_id']
    model_type = profile['model_type']
    if model_type == 'claude':
        maxOutputTokens = get_max_output_tokens(modelId)
    else:
        maxOutputTokens = 5120  # 5k
    number_of_models = len(models)

    logger.info(f"LLM: {selected_chat}, bedrock_region: {bedrock_region}, modelId: {modelId}, model_type: {model_type}")

    if profile['model_type'] == 'nova':
        STOP_SEQUENCE = '"\n\n<thinking>", "\n<thinking>", " <thinking>"'
    elif profile['model_type'] == 'claude':
        STOP_SEQUENCE = "\n\nHuman:" 
    elif profile['model_type'] == 'openai':
        STOP_SEQUENCE = "" 
                          
    # bedrock   
    boto3_bedrock = boto3.client(
        service_name='bedrock-runtime',
        region_name=bedrock_region,
        config=Config(
            retries = {
                'max_attempts': 30
            },
            read_timeout=300
        )
    )

    if profile['model_type'] != 'openai' and extended_thinking=='Enable':
        maxReasoningOutputTokens=64000
        logger.info(f"extended_thinking: {extended_thinking}")
        thinking_budget = min(maxOutputTokens, maxReasoningOutputTokens-1000)

        parameters = {
            "max_tokens":maxReasoningOutputTokens,
            "temperature":1,            
            "thinking": {
                "type": "enabled",
                "budget_tokens": thinking_budget
            },
            "stop_sequences": [STOP_SEQUENCE]
        }
    elif profile['model_type'] != 'openai' and extended_thinking=='Disable':
        parameters = {
            "max_tokens":maxOutputTokens,     
            "temperature":0.1,
            "top_k":250,
            "stop_sequences": [STOP_SEQUENCE]
        }
    elif profile['model_type'] == 'openai':
        parameters = {
            "max_tokens":maxOutputTokens,     
            "temperature":0.1
        }

    chat = ChatBedrock(   # new chat model
        model_id=modelId,
        client=boto3_bedrock, 
        model_kwargs=parameters,
        region_name=bedrock_region
    )
    
    # Disable streaming for OpenAI models
    if profile['model_type'] == 'openai':
        chat.streaming = False
    
    if multi_region=='Enable':
        selected_chat = selected_chat + 1
        if selected_chat == number_of_models:
            selected_chat = 0
    else:
        selected_chat = 0

    return chat

def print_doc(i, doc):
    if len(doc.page_content)>=100:
        text = doc.page_content[:100]
    else:
        text = doc.page_content
            
    logger.info(f"{i}: {text}, metadata:{doc.metadata}")

def isKorean(text):
    # check korean
    pattern_hangul = re.compile('[\u3131-\u3163\uac00-\ud7a3]+')
    word_kor = pattern_hangul.search(str(text))
    # print('word_kor: ', word_kor)

    if word_kor and word_kor != 'None':
        # logger.info(f"Korean: {word_kor}")
        return True
    else:
        # logger.info(f"Not Korean:: {word_kor}")
        return False
    
def traslation(chat, text, input_language, output_language):
    system = (
        "You are a helpful assistant that translates {input_language} to {output_language} in <article> tags." 
        "Put it in <result> tags."
    )
    human = "<article>{text}</article>"
    
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    # print('prompt: ', prompt)
    
    chain = prompt | chat    
    try: 
        result = chain.invoke(
            {
                "input_language": input_language,
                "output_language": output_language,
                "text": text,
            }
        )
        
        msg = result.content
        # print('translated text: ', msg)
    except Exception:
        err_msg = traceback.format_exc()
        logger.info(f"error message: {err_msg}")     
        raise Exception ("Not able to request to LLM")

    return msg[msg.find('<result>')+8:len(msg)-9] # remove <result> tag

def show_extended_thinking(st, result):
    # logger.info(f"result: {result}")
    if "thinking" in result.response_metadata:
        if "text" in result.response_metadata["thinking"]:
            thinking = result.response_metadata["thinking"]["text"]
            st.info(thinking)

####################### LangChain #######################
# General Conversation
#########################################################
def general_conversation(query):
    global memory_chain
    initiate()  # Initialize memory_chain
    llm = get_chat(extended_thinking=reasoning_mode)

    system = (
        "당신의 이름은 서연이고, 질문에 대해 친절하게 답변하는 사려깊은 인공지능 도우미입니다."
        "상황에 맞는 구체적인 세부 정보를 충분히 제공합니다." 
        "모르는 질문을 받으면 솔직히 모른다고 말합니다."
    )
    
    human = "Question: {input}"
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system), 
        MessagesPlaceholder(variable_name="history"), 
        ("human", human)
    ])
                
    if memory_chain and hasattr(memory_chain, 'load_memory_variables'):
        history = memory_chain.load_memory_variables({})["chat_history"]
        # Ensure history starts with a HumanMessage (Bedrock Converse API requirement)
        if history and isinstance(history[0], AIMessage):
            history = history[1:]
    else:
        history = []

    chain = prompt | llm | StrOutputParser()
    try: 
        stream = chain.stream(
            {
                "history": history,
                "input": query,
            }
        )  
        logger.info(f"stream: {stream}")
            
    except Exception:
        err_msg = traceback.format_exc()
        logger.info(f"error message: {err_msg}")      
        raise Exception ("Not able to request to LLM: "+err_msg)
        
    return stream

def get_summary(docs):    
    llm = get_chat(extended_thinking=reasoning_mode)

    text = ""
    for doc in docs:
        text = text + doc
    
    if isKorean(text)==True:
        system = (
            "다음의 <article> tag안의 문장을 요약해서 500자 이내로 설명하세오."
        )
    else: 
        system = (
            "Here is pieces of article, contained in <article> tags. Write a concise summary within 500 characters."
        )
    
    human = "<article>{text}</article>"
    
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    # print('prompt: ', prompt)
    
    chain = prompt | llm    
    try: 
        result = chain.invoke(
            {
                "text": text
            }
        )
        
        summary = result.content
        logger.info(f"esult of summarization: {summary}")
    except Exception:
        err_msg = traceback.format_exc()
        logger.info(f"error message: {err_msg}") 
        raise Exception ("Not able to request to LLM")
    
    return summary

def summary_image(img_base64, instruction):      
    llm = get_chat(extended_thinking=reasoning_mode)

    if instruction:
        logger.info(f"instruction: {instruction}")
        query = f"{instruction}. <result> tag를 붙여주세요. 한국어로 답변하세요."
        
    else:
        query = "이미지가 의미하는 내용을 풀어서 자세히 알려주세요. markdown 포맷으로 답변을 작성합니다."
    
    messages = [
        HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_base64}", 
                    },
                },
                {
                    "type": "text", "text": query
                },
            ]
        )
    ]
    
    for attempt in range(5):
        logger.info(f"attempt: {attempt}")
        try: 
            result = llm.invoke(messages)
            
            extracted_text = result.content
            # print('summary from an image: ', extracted_text)
            break
        except Exception:
            err_msg = traceback.format_exc()
            logger.info(f"error message: {err_msg}")                    
            raise Exception ("Not able to request to LLM")
        
    return extracted_text

def extract_text(img_base64):    
    multimodal = get_chat(extended_thinking=reasoning_mode)
    query = "텍스트를 추출해서 markdown 포맷으로 변환하세요. <result> tag를 붙여주세요."
    
    extracted_text = ""
    messages = [
        HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_base64}", 
                    },
                },
                {
                    "type": "text", "text": query
                },
            ]
        )
    ]
    
    for attempt in range(5):
        logger.info(f"attempt: {attempt}")
        try: 
            result = multimodal.invoke(messages)
            
            extracted_text = result.content
            # print('result of text extraction from an image: ', extracted_text)
            break
        except Exception:
            err_msg = traceback.format_exc()
            logger.info(f"error message: {err_msg}")                    
            # raise Exception ("Not able to request to LLM")
    
    logger.info(f"Extracted_text: {extracted_text}")
    if len(extracted_text)<10:
        extracted_text = "텍스트를 추출하지 못하였습니다."    

    return extracted_text

fileId = uuid.uuid4().hex
# print('fileId: ', fileId)

####################### LangChain #######################
# Image Summarization
#########################################################
def summarize_image(image_content, prompt, st):
    img = Image.open(BytesIO(image_content))
    
    width, height = img.size 
    logger.info(f"width: {width}, height: {height}, size: {width*height}")
    
    # Image resizing and size verification
    isResized = False
    max_size = 5 * 1024 * 1024  # 5MB in bytes
    
    # Initial resizing (based on pixel count)
    while(width*height > 2000000):  # Limit to approximately 2M pixels
        width = int(width/2)
        height = int(height/2)
        isResized = True
        logger.info(f"width: {width}, height: {height}, size: {width*height}")
    
    if isResized:
        img = img.resize((width, height))
    
    # Base64 size verification and additional resizing
    max_attempts = 5
    for attempt in range(max_attempts):
        buffer = BytesIO()
        img.save(buffer, format="PNG", optimize=True)
        img_bytes = buffer.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")
        
        # Base64 size verification (actual transmission size)
        base64_size = len(img_base64.encode('utf-8'))
        logger.info(f"attempt {attempt + 1}: base64_size = {base64_size} bytes")
        
        if base64_size <= max_size:
            break
        else:
            # Resize smaller if still too large
            width = int(width * 0.8)
            height = int(height * 0.8)
            img = img.resize((width, height))
            logger.info(f"resizing to {width}x{height} due to size limit")
    
    if base64_size > max_size:
        logger.warning(f"Image still too large after {max_attempts} attempts: {base64_size} bytes")
        raise Exception(f"이미지 크기가 너무 큽니다. 5MB 이하의 이미지를 사용해주세요.")

    # extract text from the image
    if debug_mode=="Enable":
        status = "이미지에서 텍스트를 추출합니다."
        logger.info(f"status: {status}")
        st.info(status)

    text = extract_text(img_base64)
    logger.info(f"extracted text: {text}")

    if text.find('<result>') != -1:
        extracted_text = text[text.find('<result>')+8:text.find('</result>')] # remove <result> tag
        # print('extracted_text: ', extracted_text)
    else:
        extracted_text = text
    
    if debug_mode=="Enable":
        status = f"### 추출된 텍스트\n\n{extracted_text}"
        logger.info(f"status: {status}")
        st.info(status)
    
    if debug_mode=="Enable":
        status = "이미지의 내용을 분석합니다."
        logger.info(f"status: {status}")
        st.info(status)

    image_summary = summary_image(img_base64, prompt)
    
    if text.find('<result>') != -1:
        image_summary = image_summary[image_summary.find('<result>')+8:image_summary.find('</result>')]
    logger.info(f"image summary: {image_summary}")
            
    # if len(extracted_text) > 10:
    #     contents = f"## Image analysis\n\n{image_summary}\n\n## Extracted text\n\n{extracted_text}"
    # else:
    #     contents = f"## Image analysis\n\n{image_summary}"
    contents = f"## 이미지 분석\n\n{image_summary}"
    logger.info(f"image contents: {contents}")

    return contents

streaming_index = None
index = 0
def add_notification(containers, message):
    global index

    if index == streaming_index:
        index += 1

    if containers is not None:
        containers['notification'][index].info(message)
    index += 1

def update_streaming_result(containers, message, type):
    global streaming_index
    streaming_index = index

    if containers is not None:
        if type == "markdown":
            containers['notification'][streaming_index].markdown(message)
        elif type == "info":
            containers['notification'][streaming_index].info(message)
def update_tool_notification(containers, tool_index, message):
    if containers is not None:
        containers['notification'][tool_index].info(message)

tool_info_list = dict()
tool_input_list = dict()
tool_name_list = dict()

sharing_url = config["sharing_url"] if "sharing_url" in config else None
s3_prefix = "docs"
capture_prefix = "captures"

def get_tool_info(tool_name, tool_content):
    tool_references = []    
    urls = []
    content = ""
    
    # OpenSearch
    if tool_name == "SearchIndexTool": 
        if ":" in tool_content:
            extracted_json_data = tool_content.split(":", 1)[1].strip()
            try:
                json_data = json.loads(extracted_json_data)
                # logger.info(f"extracted_json_data: {extracted_json_data[:200]}")
            except json.JSONDecodeError:
                logger.info("JSON parsing error")
                json_data = {}
        else:
            json_data = {}
        
        if "hits" in json_data:
            hits = json_data["hits"]["hits"]
            if hits:
                logger.info(f"hits[0]: {hits[0]}")

            for hit in hits:
                text = hit["_source"]["text"]
                metadata = hit["_source"]["metadata"]
                
                content += f"{text}\n\n"

                filename = metadata["name"].split("/")[-1]
                # logger.info(f"filename: {filename}")
                
                content_part = text.replace("\n", "")
                tool_references.append({
                    "url": metadata["url"], 
                    "title": filename,
                    "content": content_part[:100] + "..." if len(content_part) > 100 else content_part
                })
                
        logger.info(f"content: {content}")
        
    # aws document
    elif tool_name == "search_documentation":
        try:
            # Handle tool_content when it is a list (e.g. [{'type': 'text', 'text': '...'}])
            if isinstance(tool_content, list):
                # Extract text field from the first list item
                if len(tool_content) > 0 and isinstance(tool_content[0], dict) and 'text' in tool_content[0]:
                    tool_content = tool_content[0]['text']
                else:
                    logger.info(f"Unexpected list format: {tool_content}")
                    return content, urls, tool_references
            
            # Parse JSON when tool_content is a string
            if isinstance(tool_content, str):
                json_data = json.loads(tool_content)
            elif isinstance(tool_content, dict):
                json_data = tool_content
            else:
                logger.info(f"Unexpected tool_content type: {type(tool_content)}")
                return content, urls, tool_references
            
            # Extract results from search_results array
            search_results = json_data.get('search_results', [])
            if not search_results:
                # If no search_results, json_data may be the array itself
                if isinstance(json_data, list):
                    search_results = json_data
                else:
                    logger.info(f"No search_results found in JSON data")
                    return content, urls, tool_references
            
            for item in search_results:
                logger.info(f"item: {item}")
                
                if isinstance(item, str):
                    try:
                        item = json.loads(item)
                    except json.JSONDecodeError:
                        logger.info(f"Failed to parse item as JSON: {item}")
                        continue
                
                if isinstance(item, dict) and 'url' in item and 'title' in item:
                    url = item['url']
                    title = item['title']
                    content_text = item.get('context', '')[:100] + "..." if len(item.get('context', '')) > 100 else item.get('context', '')
                    tool_references.append({
                        "url": url,
                        "title": title,
                        "content": content_text
                    })
                else:
                    logger.info(f"Invalid item format: {item}")
                    
        except json.JSONDecodeError as e:
            logger.info(f"JSON parsing error: {e}, tool_content: {tool_content}")
            pass
        except Exception as e:
            logger.info(f"Unexpected error in search_documentation: {e}, tool_content type: {type(tool_content)}")
            pass

        logger.info(f"content: {content}")
        logger.info(f"tool_references: {tool_references}")
            
    # aws-knowledge
    elif tool_name == "aws___read_documentation":
        logger.info(f"#### {tool_name} ####")
        if isinstance(tool_content, dict):
            json_data = tool_content
        elif isinstance(tool_content, list):
            json_data = tool_content
        else:
            json_data = json.loads(tool_content)
        
        logger.info(f"json_data: {json_data}")
        payload = json_data["response"]["payload"]
        if "content" in payload:
            payload_content = payload["content"]
            if "result" in payload_content:
                result = payload_content["result"]
                logger.info(f"result: {result}")
                if isinstance(result, str) and "AWS Documentation from" in result:
                    logger.info(f"Processing AWS Documentation format: {result}")
                    try:
                        # Extract URL from "AWS Documentation from https://..."
                        url_start = result.find("https://")
                        if url_start != -1:
                            # Find the colon after the URL (not inside the URL)
                            url_end = result.find(":", url_start)
                            if url_end != -1:
                                # Check if the colon is part of the URL or the separator
                                url_part = result[url_start:url_end]
                                # If the colon is immediately after the URL, use it as separator
                                if result[url_end:url_end+2] == ":\n":
                                    url = url_part
                                    content_start = url_end + 2  # Skip the colon and newline
                                else:
                                    # Try to find the actual URL end by looking for space or newline
                                    space_pos = result.find(" ", url_start)
                                    newline_pos = result.find("\n", url_start)
                                    if space_pos != -1 and newline_pos != -1:
                                        url_end = min(space_pos, newline_pos)
                                    elif space_pos != -1:
                                        url_end = space_pos
                                    elif newline_pos != -1:
                                        url_end = newline_pos
                                    else:
                                        url_end = len(result)
                                    
                                    url = result[url_start:url_end]
                                    content_start = url_end + 1
                                
                                # Remove trailing colon from URL if present
                                if url.endswith(":"):
                                    url = url[:-1]
                                
                                # Extract content after the URL
                                if content_start < len(result):
                                    content_text = result[content_start:].strip()
                                    # Truncate content for display
                                    display_content = content_text[:100] + "..." if len(content_text) > 100 else content_text
                                    display_content = display_content.replace("\n", "")
                                    
                                    tool_references.append({
                                        "url": url,
                                        "title": "AWS Documentation",
                                        "content": display_content
                                    })
                                    content += content_text + "\n\n"
                                    logger.info(f"Extracted URL: {url}")
                                    logger.info(f"Extracted content length: {len(content_text)}")
                    except Exception as e:
                        logger.error(f"Error parsing AWS Documentation format: {e}")
        logger.info(f"content: {content}")
        logger.info(f"tool_references: {tool_references}")

    elif tool_name in ("memory_search", "memory_get"):
        pass

    else:        
        try:
            if isinstance(tool_content, dict):
                json_data = tool_content
            elif isinstance(tool_content, list):
                json_data = tool_content
            else:
                json_data = json.loads(tool_content)
            
            logger.info(f"json_data: {json_data}")
            if isinstance(json_data, dict) and "path" in json_data:  # path
                path = json_data["path"]
                if isinstance(path, list):
                    for url in path:
                        urls.append(url)
                else:
                    urls.append(path)
            elif isinstance(json_data, list):  # Parse JSON from text field when json_data is a list
                for item in json_data:
                    if isinstance(item, dict) and "text" in item:
                        try:
                            text_json = json.loads(item["text"])
                            if isinstance(text_json, dict) and "path" in text_json:
                                path = text_json["path"]
                                if isinstance(path, list):
                                    for url in path:
                                        urls.append(url)
                                else:
                                    urls.append(path)
                        except (json.JSONDecodeError, TypeError):
                            pass            


            if isinstance(json_data, dict):
                for item in json_data:
                    logger.info(f"item: {item}")
                    if "reference" in item and "contents" in item:
                        url = item["reference"]["url"]
                        title = item["reference"]["title"]
                        content_text = item["contents"][:100] + "..." if len(item["contents"]) > 100 else item["contents"]
                        tool_references.append({
                            "url": url,
                            "title": title,
                            "content": content_text
                        })
            elif isinstance(json_data, list):
                logger.info(f"json_data is a list: {json_data}")
                for item in json_data:
                    if isinstance(item, dict) and "text" in item:
                        try:
                            # Parse JSON string inside text field
                            text_json = json.loads(item["text"])
                            if isinstance(text_json, list):
                                # Parsed JSON is a list
                                for ref_item in text_json:
                                    if isinstance(ref_item, dict) and "reference" in ref_item and "contents" in ref_item:
                                        url = ref_item["reference"]["url"]
                                        title = ref_item["reference"]["title"]
                                        content_text = ref_item["contents"][:100] + "..." if len(ref_item["contents"]) > 100 else ref_item["contents"]
                                        tool_references.append({
                                            "url": url,
                                            "title": title,
                                            "content": content_text
                                        })
                            elif isinstance(text_json, dict) and "reference" in text_json and "contents" in text_json:
                                # Parsed JSON is a dict
                                url = text_json["reference"]["url"]
                                title = text_json["reference"]["title"]
                                content_text = text_json["contents"][:100] + "..." if len(text_json["contents"]) > 100 else text_json["contents"]
                                tool_references.append({
                                    "url": url,
                                    "title": title,
                                    "content": content_text
                                })
                        except (json.JSONDecodeError, TypeError) as e:
                            logger.warning(f"Failed to parse text JSON: {e}")
                            pass
                    elif isinstance(item, dict) and "reference" in item and "contents" in item:
                        # List item has reference directly
                        url = item["reference"]["url"]
                        title = item["reference"]["title"]
                        content_text = item["contents"][:100] + "..." if len(item["contents"]) > 100 else item["contents"]
                        tool_references.append({
                            "url": url,
                            "title": title,
                            "content": content_text
                        })
                
            logger.info(f"tool_references: {tool_references}")

        except json.JSONDecodeError:
            pass

    return content, urls, tool_references

#########################################################
# Memory
#########################################################
import agentcore_memory

agent_type = 'langgraph'
enable_memory = 'Enable'
user_id = agent_type # for testing

memory_id = actor_id = session_id = None
def initiate_memory():
    global memory_id, actor_id, session_id

    # Ensure user_id is valid
    effective_user_id = user_id if user_id and user_id.strip() else agent_type
    if not effective_user_id or not effective_user_id.strip():
        effective_user_id = "default"  # Fallback to default if agent_type is also invalid
    logger.info(f"Using user_id: {effective_user_id}")

    # initate memory variables    
    memory_id, actor_id, session_id, namespace = agentcore_memory.load_memory_variables(effective_user_id)
    logger.info(f"memory_id: {memory_id}, actor_id: {actor_id}, session_id: {session_id}, namespace: {namespace}")

    if memory_id is None:
        # retrieve memory id
        memory_id = agentcore_memory.retrieve_memory_id()
        logger.info(f"memory_id: {memory_id}")        
        
        # create memory if not exists
        if memory_id is None:
            logger.info(f"Memory will be created...")
            memory_id = agentcore_memory.create_memory(namespace, effective_user_id)
            logger.info(f"Memory was created... {memory_id}")
        
        # create strategy if not exists
        agentcore_memory.create_strategy_if_not_exists(memory_id=memory_id, namespace=namespace, strategy_name=effective_user_id)

        # save memory variables
        agentcore_memory.update_memory_variables(
            user_id=effective_user_id, 
            memory_id=memory_id, 
            actor_id=actor_id, 
            session_id=session_id, 
            namespace=namespace)
    
enable_short_term_memory = "Disable"

def save_to_memory(query, result):
    if memory_id is None and enable_memory=="Enable":
        initiate_memory()    
    agentcore_memory.save_conversation_to_memory(memory_id, actor_id, session_id, query, result) 