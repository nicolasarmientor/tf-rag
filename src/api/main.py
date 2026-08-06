
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from src.retrieval.retriever import retrieve
from src.generation.generator import generate_answer
from src.api.schemas import AskRequest, AskResponse, SourceInfo
from src.retrieval.vector_store import collection

app = FastAPI()

templates = Jinja2Templates(directory="src/api/templates")
app.mount("/static", StaticFiles(directory="src/api/static"), name="static")

@app.get("/")
def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {
        "chunk_count": collection.count()
    })

@app.post("/ask-ui")
def ask_ui(request: Request, question: str = Form(...), api_key: str = Form(None)):
    if not api_key:
        return templates.TemplateResponse(request, "answer.html", {
            "question": question,
            "answer": "Please enter your Anthropic API key above to ask a question.",
            "sources": [],
            "model": "",
            "total_tokens": 0
        })

    chunks = retrieve(question, top_k=5)
    result = generate_answer(question, chunks, user_api_key=api_key)

    return templates.TemplateResponse(request, "answer.html", {
        "question": question,
        "answer": result["answer"],
        "sources": result["sources"],
        "model": result["model"],
        "total_tokens": result["input_tokens"] + result["output_tokens"]
    })

@app.post("/ask", response_model=AskResponse)
def ask(request_body: AskRequest):
    chunks = retrieve(request_body.question, top_k=request_body.top_k)
    result = generate_answer(request_body.question, chunks)

    return AskResponse(
        answer=result["answer"],
        sources=[SourceInfo(**s) for s in result["sources"]]
    )