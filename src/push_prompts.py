"""
Script para fazer push de prompts otimizados ao LangSmith Prompt Hub.

Este script:
1. Lê os prompts otimizados de prompts/bug_to_user_story_v2.yml
2. Valida os prompts
3. Faz push PÚBLICO para o LangSmith Hub
4. Adiciona metadados (tags, descrição, técnicas utilizadas)

SIMPLIFICADO: Código mais limpo e direto ao ponto.
"""

import os
import sys
import inspect
from dotenv import load_dotenv
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate
from utils import load_yaml, check_env_vars, print_section_header

load_dotenv()

# No Windows a saída redirecionada cai em cp1252 e quebra nos símbolos ✓ / ❌
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Arquivo com o prompt otimizado e chave raiz dentro do YAML
PROMPTS_FILE = "prompts/bug_to_user_story_v2.yml"
PROMPT_KEY = "bug_to_user_story_v2"


def build_chat_prompt(prompt_data: dict) -> ChatPromptTemplate:
    """
    Monta o ChatPromptTemplate a partir dos campos do YAML.

    A separação System vs User é intencional: o system carrega persona,
    regras, formato e exemplos; o user carrega apenas o relato do bug.

    Args:
        prompt_data: Dados do prompt lidos do YAML

    Returns:
        ChatPromptTemplate pronto para push
    """
    return ChatPromptTemplate.from_messages([
        ("system", prompt_data["system_prompt"]),
        ("human", prompt_data["user_prompt"]),
    ])


def build_readme(prompt_data: dict) -> str:
    """
    Gera o README publicado junto do prompt no Hub.

    Args:
        prompt_data: Dados do prompt lidos do YAML

    Returns:
        Texto em Markdown com as técnicas aplicadas
    """
    techniques = prompt_data.get("techniques_applied", [])
    linhas = [
        f"# {PROMPT_KEY}",
        "",
        prompt_data.get("description", ""),
        "",
        "## Técnicas de Prompt Engineering aplicadas",
        "",
    ]
    linhas += [f"- {tecnica}" for tecnica in techniques]
    linhas += [
        "",
        "## Entrada",
        "",
        "- `bug_report`: relato de bug em texto livre (de uma frase até um laudo com várias falhas).",
        "",
        "## Saída",
        "",
        "User Story no formato padrão (`Como um... eu quero... para que...`) com Critérios de",
        "Aceitação em Gherkin (Dado / Quando / Então / E), escalando para blocos adicionais",
        "quando o relato é técnico ou multifacetado.",
    ]
    return "\n".join(linhas)


def validate_prompt(prompt_data: dict) -> tuple[bool, list]:
    """
    Valida estrutura básica de um prompt (versão simplificada).

    Args:
        prompt_data: Dados do prompt

    Returns:
        (is_valid, errors) - Tupla com status e lista de erros
    """
    errors = []

    for campo in ("description", "system_prompt", "user_prompt", "version"):
        if not str(prompt_data.get(campo, "")).strip():
            errors.append(f"Campo obrigatório vazio ou ausente: {campo}")

    system_prompt = str(prompt_data.get("system_prompt", ""))
    user_prompt = str(prompt_data.get("user_prompt", ""))

    if "TODO" in system_prompt or "TODO" in user_prompt:
        errors.append("O prompt ainda contém TODOs")

    if "{bug_report}" not in user_prompt:
        errors.append("O user_prompt precisa conter a variável {bug_report}")

    if "{bug_report}" in system_prompt:
        errors.append(
            "O system_prompt não deve repetir {bug_report} "
            "(o relato entra apenas pelo user_prompt)"
        )

    techniques = prompt_data.get("techniques_applied", [])
    if len(techniques) < 2:
        errors.append(f"Mínimo de 2 técnicas requeridas, encontradas: {len(techniques)}")

    return (len(errors) == 0, errors)


def push_prompt_to_langsmith(prompt_name: str, prompt_data: dict) -> bool:
    """
    Faz push do prompt otimizado para o LangSmith Hub (PÚBLICO).

    Args:
        prompt_name: Nome do prompt
        prompt_data: Dados do prompt

    Returns:
        True se sucesso, False caso contrário
    """
    try:
        chat_prompt = build_chat_prompt(prompt_data)
    except Exception as e:
        print(f"❌ Erro ao montar o ChatPromptTemplate: {e}")
        return False

    techniques = prompt_data.get("techniques_applied", [])

    # Tags: as do YAML + a versão + as técnicas (metadados exigidos pelo enunciado)
    tags = list(prompt_data.get("tags", []))
    tags.append(prompt_data.get("version", "v2"))
    for tecnica in techniques:
        slug = tecnica.split("(")[0].strip().lower().replace(" ", "-")
        if slug and slug not in tags:
            tags.append(slug)

    description = prompt_data.get("description", "")

    print(f"Publicando: {prompt_name}")
    print(f"   Técnicas: {', '.join(techniques)}")
    print(f"   Tags: {', '.join(tags)}")
    print("   Visibilidade: PÚBLICO")

    # A assinatura de hub.push mudou entre versões do LangChain/LangSmith:
    # filtramos os argumentos realmente aceitos para não quebrar o push.
    kwargs_desejados = {
        "new_repo_is_public": True,
        "new_repo_description": description,
        "description": description,
        "is_public": True,
        "readme": build_readme(prompt_data),
        "tags": tags,
    }

    try:
        parametros = inspect.signature(hub.push).parameters
        kwargs = {k: v for k, v in kwargs_desejados.items() if k in parametros}
    except (TypeError, ValueError):
        kwargs = {"new_repo_is_public": True}

    try:
        url = hub.push(prompt_name, chat_prompt, **kwargs)
        print(f"   ✓ Publicado com sucesso")
        print(f"   🔗 {url}")
        return True

    except Exception as e:
        print(f"   ❌ Erro ao publicar: {e}\n")
        print("Verifique:")
        print("  - LANGSMITH_API_KEY tem permissão de escrita")
        print("  - USERNAME_LANGSMITH_HUB é exatamente o seu handle do Hub")
        print("  - Você já criou um workspace/handle no LangSmith")
        return False


def main():
    """Função principal"""
    print_section_header("PUSH DE PROMPTS OTIMIZADOS PARA O LANGSMITH HUB")

    if not check_env_vars(["LANGSMITH_API_KEY", "USERNAME_LANGSMITH_HUB"]):
        return 1

    username = os.getenv("USERNAME_LANGSMITH_HUB", "").strip()

    prompts = load_yaml(PROMPTS_FILE)
    if not prompts:
        print(f"❌ Não foi possível carregar {PROMPTS_FILE}")
        return 1

    prompt_data = prompts.get(PROMPT_KEY)
    if not prompt_data:
        print(f"❌ Chave '{PROMPT_KEY}' não encontrada em {PROMPTS_FILE}")
        return 1

    print(f"Validando {PROMPTS_FILE}...")
    is_valid, errors = validate_prompt(prompt_data)

    if not is_valid:
        print("❌ Prompt inválido:")
        for erro in errors:
            print(f"   - {erro}")
        return 1

    print("   ✓ Prompt válido\n")

    prompt_name = f"{username}/{PROMPT_KEY}"

    if not push_prompt_to_langsmith(prompt_name, prompt_data):
        return 1

    print("\n✅ Push concluído com sucesso!")
    print("\nPróximos passos:")
    print(f"1. Confira o prompt em: https://smith.langchain.com/prompts/{PROMPT_KEY}")
    print("2. Execute a avaliação: python src/evaluate.py")

    return 0


if __name__ == "__main__":
    sys.exit(main())
