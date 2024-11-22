from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.llms import FakeListLLM, HuggingFaceHub
from langchain_openai.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
import nest_asyncio
import uvicorn
import os

chave_openai = "sua_openai_api_key_aqui"
chave_huggingface = "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"
os.environ["OPENAI_API_KEY"] = chave_openai
os.environ["HUGGINGFACEHUB_API_TOKEN"] = chave_huggingface

nest_asyncio.apply()

aplicativo = FastAPI()

class EntradaTexto(BaseModel):
    texto: str

@aplicativo.post("/fake-llm/")
async def resposta_llm_simulada(entrada: EntradaTexto):
    try:
        llm_simulado = FakeListLLM(responses=["Olá!", "Oi!", "Bonjour!", "Ciao!"])
        resposta = llm_simulado.invoke(entrada.texto)
        return {"resposta": resposta}
    except Exception as erro:
        raise HTTPException(status_code=500, detail=str(erro))

@aplicativo.post("/openai-traduzir/")
async def traducao_openai(entrada: EntradaTexto):
    try:
        llm_openai = ChatOpenAI(model="gpt-3.5-turbo", api_key=chave_openai)
        prompt = ChatPromptTemplate.from_messages([
            {"role": "system", "content": "Você é um tradutor do inglês para o francês."},
            {"role": "user", "content": "Traduza o seguinte texto: {texto}"}
        ])
        resposta = llm_openai.invoke([HumanMessage(content=prompt.format_messages(texto=entrada.texto)[1].content)])
        return {"texto_traduzido": resposta.content}
    except Exception as erro:
        raise HTTPException(status_code=500, detail=str(erro))

@aplicativo.post("/huggingface-traduzir/")
async def traducao_huggingface(entrada: EntradaTexto):
    try:
        llm_huggingface = HuggingFaceHub(
            repo_id="Helsinki-NLP/opus-mt-en-de",
            huggingfacehub_api_token=chave_huggingface
        )
        resposta = llm_huggingface.invoke(entrada.texto)
        return {"texto_traduzido": resposta}
    except Exception as erro:
        raise HTTPException(status_code=500, detail=str(erro))

if __name__ == "__main__":
    uvicorn.run(aplicativo, host="127.0.0.1", port=5000)
