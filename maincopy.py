import os
import shutil
import base64
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from openai import OpenAI

# Initialize FastAPI
app = FastAPI(title="Multimodal Medical RAG API")

# Setup Directories
DATA_DIR = "./data"
CHROMA_DIR = "./chroma_db"

os.makedirs(DATA_DIR, exist_ok=True)

# Initialize OpenRouter Client
OPENROUTER_API_KEY = "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"

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# Initialize the local HuggingFace embedding model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


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
    """Retrieves context and generates an answer using Gemini 2.0 Flash (supports image input)."""

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

        # 2. Construct the Strict System Prompt
        system_prompt = (
            "You are a highly accurate medical AI assistant. "
            "Answer the user's query using the provided text context and the uploaded image (if any). "
            "If the answer cannot be determined from the context or image, state that clearly.\n\n"
            f"Extracted PDF Context:\n{context}"
        )

        # 3. Format the User Message for OpenRouter Multimodal
        user_content = [{"type": "text", "text": query}]

        # If an image was uploaded, convert to base64 and append to the content array
        if image:
            image_bytes = await image.read()
            base64_image = base64.b64encode(image_bytes).decode('utf-8')

            # Determine mime type simply from extension for this example
            mime_type = image.content_type or "image/jpeg"

            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{base64_image}"
                }
            })

        # 4. Call Gemini 2.0 Flash via OpenRouter
        response = client.chat.completions.create(
            model="google/gemini-2.0-flash-001",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]
        )

        return {
            "query": query,
            "image_provided": bool(image),
            "answer": response.choices[0].message.content,
            "sources_retrieved": len(relevant_docs)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))