from datasets import load_dataset
from openai import OpenAI
import json, os, time, random

OUTPUT_FILE = "dataset.jsonl"
TEST_MODE = False
MAX_RECORDS = 1 if TEST_MODE else None

DATASETS = [
    "Nicky0007/cointelegraph_noticias_Es",
    "bertin-project/alpaca-spanish"
]

PROVIDER = os.getenv("PROVIDER", "local")

if PROVIDER == "local":
    BASE_URL = os.getenv("LOCAL_BASE_URL", "http://localhost:1234/v1")
    API_KEY = os.getenv("LOCAL_API_KEY", "lm-studio")
    MODEL_ID = os.getenv("LOCAL_MODEL_ID", "model-identifier")
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
elif PROVIDER == "chatgpt":
    API_KEY = os.getenv("OPENAI_API_KEY", "")
    MODEL_ID = os.getenv("OPENAI_MODEL_ID", "gpt-4o-mini")
    client = OpenAI(api_key=API_KEY)
else:
    raise ValueError("PROVIDER debe ser 'local' o 'chatgpt'")

def contar_procesados(path, dataset_name):
    if not os.path.exists(path):
        return 0
    c = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                if json.loads(line).get("dataset") == dataset_name:
                    c += 1
            except json.JSONDecodeError:
                continue
    return c

def escribir_jsonl(path, dataset_name, instruction, response):
    with open(path, "a", encoding="utf-8") as f:
        json.dump({"dataset": dataset_name, "instruction": instruction, "response": response}, f, ensure_ascii=False)
        f.write("\n")

def _prompt_base(texto, tipo):
    ejemplos = ["parce", "qué más", "hágale", "melo", "bacano", "fresco", "pilas", "vaina", "no dar papaya", "camellar"]
    random.shuffle(ejemplos)
    extras = ", ".join(ejemplos[:6])
    return f"""
IMPORTANTE: Reformula esta {tipo} con ACENTO COLOMBIANO MUY MARCADO.
Usa expresiones como: {extras} y cualquier otra frase típica colombiana.
No expliques nada, devuélvela en una sola línea.

TEXTO ORIGINAL:
{texto}
""".strip()

def reformular(texto, tipo, model=MODEL_ID, retries=3):
    for i in range(retries):
        contenido = _prompt_base(texto, tipo)
        print(f"\n=== PROMPT {tipo.upper()} ENVIADO AL LLM ===\n{contenido}\n=== FIN DEL PROMPT ===\n")
        try:
            msg = [{"role": "user", "content": contenido}]
            resp = client.chat.completions.create(model=model, messages=msg, temperature=0.7)
            reform = resp.choices[0].message.content.strip()
            if reform.lower() == texto.lower():
                time.sleep(1)
                continue
            return reform
        except Exception as e:
            print(f"Error intento {i+1} ({tipo}): {e}")
            time.sleep(2)
            continue
    return texto

def procesar_dataset(nombre):
    ds = load_dataset(nombre, split="train")
    start = contar_procesados(OUTPUT_FILE, nombre)
    total = len(ds)
    limite = min(total - start, MAX_RECORDS) if MAX_RECORDS else total - start
    done = 0
    for idx in range(start, total):
        if MAX_RECORDS and done >= limite:
            break
        ex = ds[idx]
        if nombre == "Nicky0007/cointelegraph_noticias_Es":
            q = (ex.get("title") or "").strip()
            a = (ex.get("description") or "").strip()
        else:
            instr = (ex.get("instruction") or "").strip()
            inp = (ex.get("input") or "").strip()
            q = f"{instr} : {inp}" if inp else instr
            a = (ex.get("output") or "").strip()
        rq = reformular(q, "pregunta")
        ra = reformular(a, "respuesta")
        escribir_jsonl(OUTPUT_FILE, nombre, rq, ra)
        done += 1
        print(f"{idx+1}/{total} listo [{nombre}]")

def main():
    for n in DATASETS:
        procesar_dataset(n)

if __name__ == "__main__":
    main()
