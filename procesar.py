"""
Procesa las frases pendientes (issues abiertas del repo, creadas por el
Atajo de iOS) contra el modelo local, y guarda el resultado estructurado
en data/expresiones.json. Pensado para ejecutarse una vez al día vía
GitHub Actions en un runner self-hosted (con Ollama corriendo en la
misma máquina).
"""

import json
import os
from pathlib import Path
from typing import List

import instructor
import requests
from openai import OpenAI
from pydantic import BaseModel

REPO = os.environ["GITHUB_REPOSITORY"]  # "usuario/repo", lo pone Actions solo
TOKEN = os.environ["GITHUB_TOKEN"]
API = f"https://api.github.com/repos/{REPO}/issues"
HEADERS = {"Authorization": f"Bearer {TOKEN}", "Accept": "application/vnd.github+json"}

DATA_FILE = Path("data/expresiones.json")

client = instructor.from_openai(
    OpenAI(base_url="http://localhost:11434/v1", api_key="ollama"),
    mode=instructor.Mode.JSON,
)
MODEL = "qwen2.5:14b"


class ExpresionOutput(BaseModel):
    expresion: str
    traduccion: str
    registro: str
    definicion: str
    ejemplos: List[str]
    sinonimos: List[str]
    contexto_uso: str


SYSTEM_PROMPT = (
    "Eres un profesor de inglés especializado en preparación para el examen "
    "C2 Advanced/Proficiency de Cambridge. Estructura la expresión que "
    "recibas siendo preciso con el registro: en el C2 penalizan mezclar "
    "expresiones coloquiales en textos formales."
)


def obtener_pendientes() -> list[dict]:
    resp = requests.get(API, headers=HEADERS, params={"state": "open"})
    resp.raise_for_status()
    return resp.json()


def procesar(frase: str) -> ExpresionOutput:
    return client.chat.completions.create(
        model=MODEL,
        response_model=ExpresionOutput,
        max_retries=2,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": frase},
        ],
    )


def cerrar_issue(numero: int) -> None:
    requests.patch(f"{API}/{numero}", headers=HEADERS, json={"state": "closed"})


def main() -> None:
    pendientes = obtener_pendientes()
    if not pendientes:
        print("No hay frases nuevas.")
        return

    DATA_FILE.parent.mkdir(exist_ok=True)
    existentes = json.loads(DATA_FILE.read_text()) if DATA_FILE.exists() else []

    for issue in pendientes:
        frase = issue["title"]
        try:
            resultado = procesar(frase)
        except Exception as exc:
            print(f"Error procesando '{frase}': {exc}")
            continue

        existentes.append(resultado.model_dump())
        cerrar_issue(issue["number"])
        print(f"Procesada: {frase}")

    DATA_FILE.write_text(json.dumps(existentes, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()