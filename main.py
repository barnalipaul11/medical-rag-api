import os
import shutil
import base64
import json
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from openai import OpenAI
from dotenv import load_dotenv

# 1. Load environment variables securely
load_dotenv()

# Initialize FastAPI
app = FastAPI(title="Multimodal Medical RAG API")

# Setup Directories
DATA_DIR = "./data"
CHROMA_DIR = "./chroma_db"
os.makedirs(DATA_DIR, exist_ok=True)

# Securely fetch API key
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY is missing. Please check your .env file.")

# Initialize OpenRouter Client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# Initialize the local HuggingFace embedding model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# List of multimodal fallback models
MODELS_TO_TRY = [
    "google/gemini-2.0-flash-001",                   # 1. Primary model
    "google/gemini-2.0-flash-lite-preview-02-05:free", # 2. Free, fast multimodal fallback
    "google/gemma-3-27b-it:free",                    # 3. Open-weights multimodal fallback
    "openrouter/free"                                # 4. Ultimate catch-all fallback router
]


@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """Uploads a PDF, chunks the text, and stores it in the Vector Database."""

    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    file_path = os.path.join(DATA_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            separators=["\n\n", "\n", ".", " ", ""]
        )

        chunks = text_splitter.split_documents(documents)

        Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_DIR
        )

        return {
            "status": "success",
            "message": f"Processed {len(chunks)} chunks into ChromaDB."
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
async def chat(
    query: str = Form(...),
    image: UploadFile = File(None)  # Image is optional
):
    """Retrieves context and generates a structured JSON answer using Gemini models."""

    if not os.path.exists(CHROMA_DIR):
        raise HTTPException(status_code=400, detail="Database empty. Please upload a PDF first.")

    try:
        # 1. Retrieve text context from ChromaDB
        vectorstore = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings
        )

        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        relevant_docs = retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        sources_retrieved = len(relevant_docs)

        # 2. Construct the Strict System Prompt for JSON Formatting
        system_prompt = (
            "You are a highly accurate medical AI assistant. "
            "Answer the user's query using the provided text context and the uploaded image (if any). "
            "You MUST respond strictly with a valid JSON object and nothing else. Do not use markdown blocks. "
            "The JSON object must contain exactly these two keys: \n"
            '"disease": The name of the possible condition or "Unknown".\n'
            '"firstaid": Immediate actions to take and general advice based on the context.\n\n'
            f"Extracted PDF Context:\n{context}"
        )

        # 3. Format the User Message for OpenRouter Multimodal
        user_content = [{"type": "text", "text": query}]
        image_provided = False

        if image:
            image_bytes = await image.read()
            if image_bytes:
                base64_image = base64.b64encode(image_bytes).decode('utf-8')
                mime_type = image.content_type or "image/jpeg"

                user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{base64_image}"
                    }
                })
                image_provided = True

        # 4. Call LLM via OpenRouter with Fallbacks
        response = None
        last_error = ""
        used_model = ""

        for model_id in MODELS_TO_TRY:
            try:
                response = client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content}
                    ]
                )
                used_model = model_id
                break  # If successful, break out of the loop
            except Exception as e:
                last_error = str(e)
                continue  # If it fails, move to the next model
        
        if not response:
            raise HTTPException(status_code=503, detail=f"All models failed. Last error: {last_error}")

        # 5. Parse the output into a dictionary
        raw_answer = response.choices[0].message.content.strip()

        # Clean up markdown code blocks if the LLM ignores instructions
        if raw_answer.startswith("```json"):
            raw_answer = raw_answer[7:]
        elif raw_answer.startswith("```"):
            raw_answer = raw_answer[3:]
            
        if raw_answer.endswith("```"):
            raw_answer = raw_answer[:-3]
            
        raw_answer = raw_answer.strip()

        try:
            answer_dict = json.loads(raw_answer)
        except json.JSONDecodeError:
            # Safe fallback if parsing fails completely
            answer_dict = {
                "disease": "Error parsing disease.",
                "firstaid": raw_answer 
            }

        # 6. Return the final structured output
        return {
            "query": query,
            "image_provided": image_provided,
            "answer": answer_dict,
            "model_used": used_model,
            "sources_retrieved": sources_retrieved
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))