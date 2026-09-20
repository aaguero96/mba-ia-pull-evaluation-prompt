"""
Script para fazer pull de prompts do LangSmith Prompt Hub.

Este script:
1. Conecta ao LangSmith usando credenciais do .env
2. Faz pull dos prompts do Hub
3. Salva localmente em prompts/bug_to_user_story_v1.yml

SIMPLIFICADO: Usa serialização nativa do LangChain para extrair prompts.
"""

import os
import sys
from datetime import date
from pathlib import Path
from dotenv import load_dotenv
from langchain import hub
from utils import save_yaml, check_env_vars, print_section_header

load_dotenv()

# No Windows a saída redirecionada cai em cp1252 e quebra nos símbolos ✓ / ❌
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Prompt de baixa qualidade publicado pelo professor (ponto de partida do desafio)
SOURCE_PROMPT = "leonanluppi/bug_to_user_story_v1"

# Onde o prompt puxado deve ser salvo (exigido pelo enunciado)
OUTPUT_FILE = "prompts/bug_to_user_story_v1.yml"

# Chave raiz usada dentro do YAML
PROMPT_KEY = "bug_to_user_story_v1"


def extract_messages(prompt_template) -> dict:
    """
    Extrai system_prompt e user_prompt de um ChatPromptTemplate do LangChain.

    O Hub devolve um objeto do LangChain (normalmente ChatPromptTemplate).
    Aqui percorremos as mensagens e separamos o texto de cada papel.

    Args:
        prompt_template: Objeto retornado por hub.pull()

    Returns:
        Dicionário com as chaves 'system_prompt' e 'user_prompt'
    """
    messages = {"system_prompt": "", "user_prompt": ""}

    # Prompt simples (PromptTemplate): não tem mensagens, só o template
    if not hasattr(prompt_template, "messages"):
        messages["system_prompt"] = getattr(prompt_template, "template", str(prompt_template))
        return messages

    for message in prompt_template.messages:
        # MessagesPlaceholder e afins não possuem template de texto
        inner = getattr(message, "prompt", None)
        if inner is None:
            continue

        text = getattr(inner, "template", "")
        if not text:
            continue

        role = type(message).__name__.lower()

        if "system" in role:
            messages["system_prompt"] += text
        elif "human" in role or "user" in role:
            messages["user_prompt"] += text
        else:
            # Qualquer outro papel entra no system para não perder conteúdo
            messages["system_prompt"] += f"\n{text}"

    return messages


def get_hub_metadata(prompt_name: str) -> dict:
    """
    Busca metadados do prompt no LangSmith (descrição, tags, visibilidade).

    Args:
        prompt_name: Nome completo do prompt no Hub (owner/nome)

    Returns:
        Dicionário com metadados (vazio se a consulta falhar)
    """
    try:
        from langsmith import Client

        client = Client()
        prompt_info = client.get_prompt(prompt_name)

        if prompt_info is None:
            return {}

        return {
            "description": getattr(prompt_info, "description", "") or "",
            "tags": list(getattr(prompt_info, "tags", []) or []),
            "is_public": getattr(prompt_info, "is_public", None),
        }

    except Exception as e:
        print(f"   ⚠️  Não foi possível ler os metadados do Hub: {e}")
        return {}


def pull_prompts_from_langsmith():
    """
    Faz pull do prompt de baixa qualidade e salva em prompts/bug_to_user_story_v1.yml.

    Returns:
        True se o prompt foi puxado e salvo com sucesso, False caso contrário
    """
    print(f"Puxando prompt do LangSmith Hub: {SOURCE_PROMPT}")

    try:
        prompt_template = hub.pull(SOURCE_PROMPT)
    except Exception as e:
        print(f"\n❌ Falha ao puxar '{SOURCE_PROMPT}': {e}\n")
        print("Verifique:")
        print("  - LANGSMITH_API_KEY está correta no .env")
        print("  - O nome do prompt está correto")
        print("  - Sua conexão com a internet está funcionando")
        return False

    print("   ✓ Prompt carregado do Hub")

    messages = extract_messages(prompt_template)

    if not messages["system_prompt"] and not messages["user_prompt"]:
        print("❌ O prompt veio vazio: nenhuma mensagem de texto encontrada.")
        return False

    metadata = get_hub_metadata(SOURCE_PROMPT)

    prompt_data = {
        PROMPT_KEY: {
            "description": metadata.get("description")
            or "Prompt para converter relatos de bugs em User Stories",
            "system_prompt": messages["system_prompt"],
            "user_prompt": messages["user_prompt"],
            "version": "v1",
            "source": SOURCE_PROMPT,
            "pulled_at": date.today().isoformat(),
            "input_variables": list(getattr(prompt_template, "input_variables", [])),
            "tags": metadata.get("tags") or ["bug-analysis", "user-story", "product-management"],
        }
    }

    if not save_yaml(prompt_data, OUTPUT_FILE):
        return False

    print(f"   ✓ Prompt salvo em: {OUTPUT_FILE}")

    print("\nResumo do prompt puxado:")
    print(f"   - Variáveis de entrada: {prompt_data[PROMPT_KEY]['input_variables']}")
    print(f"   - Tamanho do system_prompt: {len(messages['system_prompt'])} caracteres")
    print(f"   - Tamanho do user_prompt: {len(messages['user_prompt'])} caracteres")

    return True


def main():
    """Função principal"""
    print_section_header("PULL DE PROMPTS DO LANGSMITH HUB")

    if not check_env_vars(["LANGSMITH_API_KEY"]):
        return 1

    if not pull_prompts_from_langsmith():
        return 1

    print("\n✅ Pull concluído com sucesso!")
    print("\nPróximos passos:")
    print(f"1. Analise o prompt em {OUTPUT_FILE}")
    print("2. Refatore-o em prompts/bug_to_user_story_v2.yml")
    print("3. Publique a versão otimizada: python src/push_prompts.py")

    return 0


if __name__ == "__main__":
    sys.exit(main())
