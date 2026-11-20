"""
Procesa las frases pendientes (issues abiertas del repo, creadas por el
Atajo de iOS) contra el modelo local, y guarda el resultado estructurado
en data/expresiones.json. Pensado para ejecutarse una vez al día vía
GitHub Actions en un runner self-hosted (con Ollama corriendo en la
misma máquina).
"""

import json
import os
import re
from pathlib import Path
from typing import List

import instructor
import requests
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()  # no hace nada si no hay .env (caso de GitHub Actions)

REPO = os.environ["GITHUB_REPOSITORY"]  # "usuario/repo"
TOKEN = os.environ["GITHUB_TOKEN"]
API = f"https://api.github.com/repos/{REPO}/issues"
HEADERS = {"Authorization": f"Bearer {TOKEN}", "Accept": "application/vnd.github+json"}

DATA_FILE = Path("data/expresiones.json")

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:14b")

client = instructor.from_openai(
    OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama"),
    mode=instructor.Mode.JSON,
)


class ExpresionOutput(BaseModel):
    expresion: str
    traduccion: str
    registro: str
    definicion: str
    ejemplos: List[str]


SYSTEM_PROMPT = """
Actúas como profesor experto de inglés para estudiantes de nivel C2. Tu tarea
es analizar la expresión proporcionada y generar una ficha clara, precisa y
útil para memorizarla.

Interpreta todos los datos de entrada:
- La expresión identifica la unidad que debes explicar. Respeta su forma y no
  la sustituyas por otra expresión que solo sea parecida.
- El contexto opcional describe la situación de uso. Úsalo para precisar el
  significado, el registro, la intención comunicativa y los matices.
- La frase opcional muestra un uso real de la expresión. Utilízala como prueba
  principal de cuál de sus posibles sentidos aplica. No la vuelvas a incluir
  automáticamente como ejemplo.
- Si faltan contexto y frase, analiza el sentido habitual de la expresión y no
  inventes circunstancias concretas.

Para cada campo:
- expresion: conserva la expresión en inglés, normalizando únicamente errores
  evidentes de mayúsculas o puntuación.
- traduccion: ofrece un equivalente natural al español, no una traducción
  literal si suena artificial. Si existen varios sentidos comunes relevantes,
  menciónalos brevemente.
- registro: indica el nivel apropiado (p. ej., formal, neutro, coloquial o
  muy coloquial) y, si es útil, el ámbito y las restricciones principales de
  uso. Sé específico; no uses frases genéricas sobre el examen.
- definicion: explica en español el significado y los matices esenciales con
  frases concisas. Incluye patrones gramaticales relevantes, pero no añadas
  información insegura.
- ejemplos: escribe entre 2 y 4 oraciones completas, naturales y variadas,
  completamente en inglés. Deben mostrar la expresión en uso gramaticalmente
  correcto y acorde con su registro. No traduzcas los ejemplos ni añadas
  explicaciones dentro de este campo.

Si la expresión es ambigua, prioriza el sentido respaldado por el contexto o
la frase. Si la frase y el contexto parecen incompatibles, privilegia la frase
porque muestra el uso concreto. Si hay datos insuficientes, no inventes
certezas. No devuelvas texto fuera de la estructura solicitada.
""".strip()


def obtener_pendientes() -> list[dict]:
    resp = requests.get(API, headers=HEADERS, params={"state": "open"})
    resp.raise_for_status()
    return resp.json()


def extraer_campos_opcionales(cuerpo: str | None) -> tuple[str | None, str | None]:
    """Extrae los campos opcionales escritos como texto o formulario de issue."""
    if not cuerpo:
        return None, None

    try:
        datos = json.loads(cuerpo)
    except (json.JSONDecodeError, TypeError):
        datos = None

    if isinstance(datos, dict):
        contexto = datos.get("contexto") or datos.get("context")
        frase = datos.get("frase donde aparece") or datos.get("sentence")
        return (
            contexto.strip() if isinstance(contexto, str) and contexto.strip() else None,
            frase.strip() if isinstance(frase, str) and frase.strip() else None,
        )

    etiquetas = {
        "contexto": "contexto",
        "context": "contexto",
        "frase donde aparece": "frase",
        "sentence": "frase",
    }
    valores: dict[str, list[str]] = {"contexto": [], "frase": []}
    campo_actual = None

    for linea in cuerpo.splitlines():
        limpia = linea.strip()
        coincidencia = re.match(
            r"^(?:#{1,6}\s*)?(?:\*\*)?(contexto|context|frase donde aparece|sentence)"
            r"(?:\*\*)?\s*:?\s*(.*)$",
            limpia,
            flags=re.IGNORECASE,
        )
        if coincidencia:
            campo_actual = etiquetas[coincidencia.group(1).lower()]
            valor = coincidencia.group(2).strip()
            if valor and valor.lower() != "_no response_":
                valores[campo_actual].append(valor)
        elif limpia.startswith("#"):
            campo_actual = None
        elif campo_actual and limpia and limpia.lower() != "_no response_":
            valores[campo_actual].append(limpia)

    contexto = "\n".join(valores["contexto"]).strip() or None
    frase = "\n".join(valores["frase"]).strip() or None
    return contexto, frase


def construir_solicitud(
    expresion: str, contexto: str | None = None, frase: str | None = None
) -> str:
    partes = [f"Expresión: {expresion.strip()}"]
    if contexto:
        partes.append(f"Contexto proporcionado por el usuario: {contexto.strip()}")
    if frase:
        partes.append(f"Frase donde aparece: {frase.strip()}")
    return "\n".join(partes)


def procesar(
    expresion: str, contexto: str | None = None, frase: str | None = None
) -> ExpresionOutput:
    return client.chat.completions.create(
        model=MODEL,
        response_model=ExpresionOutput,
        max_retries=2,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": construir_solicitud(expresion, contexto, frase),
            },
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
    contenido = DATA_FILE.read_text().strip() if DATA_FILE.exists() else ""
    existentes = json.loads(contenido) if contenido else []
    
    for issue in pendientes:
        expresion = issue["title"]
        contexto, frase = extraer_campos_opcionales(issue.get("body"))
        try:
            resultado = procesar(expresion, contexto, frase)
        except Exception as exc:
            print(f"Error procesando '{expresion}': {exc}")
            continue

        existentes.append(resultado.model_dump())
        cerrar_issue(issue["number"])
        print(f"Procesada: {expresion}")

    DATA_FILE.write_text(json.dumps(existentes, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()