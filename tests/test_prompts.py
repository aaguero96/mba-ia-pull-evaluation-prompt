"""
Testes automatizados para validação de prompts.
"""
import re
import pytest
import yaml
import sys
from pathlib import Path

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils import validate_prompt_structure

PROJECT_ROOT = Path(__file__).parent.parent
PROMPT_FILE = PROJECT_ROOT / "prompts" / "bug_to_user_story_v2.yml"
PROMPT_KEY = "bug_to_user_story_v2"


def load_prompts(file_path: str):
    """Carrega prompts do arquivo YAML."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def prompt() -> dict:
    """Dados do prompt otimizado (v2)."""
    prompts = load_prompts(PROMPT_FILE)

    assert prompts is not None, f"Arquivo vazio ou inválido: {PROMPT_FILE}"
    assert PROMPT_KEY in prompts, f"Chave '{PROMPT_KEY}' não encontrada em {PROMPT_FILE}"

    return prompts[PROMPT_KEY]


class TestPrompts:
    def test_prompt_has_system_prompt(self, prompt):
        """Verifica se o campo 'system_prompt' existe e não está vazio."""
        assert "system_prompt" in prompt, "Campo 'system_prompt' ausente no YAML"

        system_prompt = prompt["system_prompt"]

        assert isinstance(system_prompt, str), "'system_prompt' deve ser uma string"
        assert system_prompt.strip(), "'system_prompt' está vazio"
        assert len(system_prompt.strip()) > 200, (
            "'system_prompt' curto demais para conter persona, regras, formato e exemplos"
        )

    def test_prompt_has_role_definition(self, prompt):
        """Verifica se o prompt define uma persona (ex: "Você é um Product Manager")."""
        system_prompt = prompt["system_prompt"].lower()

        marcadores_de_persona = [
            "você é um",
            "você é uma",
            "voce e um",
            "atue como",
            "aja como",
        ]

        assert any(m in system_prompt for m in marcadores_de_persona), (
            "Nenhuma definição de persona encontrada no system_prompt "
            f"(esperado um destes marcadores: {marcadores_de_persona})"
        )

        assert "product manager" in system_prompt, (
            "A persona deve ser explícita e do domínio de produto (ex: Product Manager)"
        )

    def test_prompt_mentions_format(self, prompt):
        """Verifica se o prompt exige formato Markdown ou User Story padrão."""
        system_prompt = prompt["system_prompt"].lower()

        exige_markdown = "markdown" in system_prompt

        exige_user_story_padrao = all(
            trecho in system_prompt
            for trecho in ("como um", "eu quero", "para que")
        )

        assert exige_markdown or exige_user_story_padrao, (
            "O prompt não exige formato Markdown nem o formato padrão de User Story "
            "('Como um... eu quero... para que...')"
        )

        assert "critérios de aceitação" in system_prompt, (
            "O prompt não exige Critérios de Aceitação na saída"
        )

    def test_prompt_has_few_shot_examples(self, prompt):
        """Verifica se o prompt contém exemplos de entrada/saída (técnica Few-shot)."""
        system_prompt = prompt["system_prompt"]

        entradas = len(re.findall(r"^\s*ENTRADA:", system_prompt, flags=re.MULTILINE))
        saidas = len(re.findall(r"^\s*SAÍDA:", system_prompt, flags=re.MULTILINE))

        assert entradas >= 2, f"Few-shot exige ao menos 2 entradas de exemplo, encontradas: {entradas}"
        assert saidas >= 2, f"Few-shot exige ao menos 2 saídas de exemplo, encontradas: {saidas}"
        assert entradas == saidas, (
            f"Cada exemplo precisa de entrada e saída: {entradas} entradas x {saidas} saídas"
        )

    def test_prompt_no_todos(self, prompt):
        """Garante que você não esqueceu nenhum `[TODO]` no texto."""
        marcadores = ("[TODO]", "TODO", "FIXME", "XXX", "<preencher>")

        for campo, valor in prompt.items():
            if not isinstance(valor, str):
                continue

            for marcador in marcadores:
                assert marcador not in valor, (
                    f"Marcador '{marcador}' esquecido no campo '{campo}'"
                )

    def test_minimum_techniques(self, prompt):
        """Verifica (através dos metadados do yaml) se pelo menos 2 técnicas foram listadas."""
        assert "techniques_applied" in prompt, (
            "Metadado 'techniques_applied' ausente no YAML"
        )

        tecnicas = prompt["techniques_applied"]

        assert isinstance(tecnicas, list), "'techniques_applied' deve ser uma lista"
        assert len(tecnicas) >= 2, (
            f"Mínimo de 2 técnicas requeridas, encontradas: {len(tecnicas)}"
        )

        tecnicas_normalizadas = " ".join(tecnicas).lower()
        assert "few-shot" in tecnicas_normalizadas, (
            "Few-shot Learning é obrigatório segundo o enunciado"
        )

        # Reaproveita a validação oficial de src/utils.py
        is_valid, errors = validate_prompt_structure(prompt)
        assert is_valid, f"Estrutura do prompt inválida: {errors}"


class TestPromptTemplate:
    """Testes extras de segurança do template (além dos 6 obrigatórios)."""

    def test_user_prompt_has_bug_report_variable(self, prompt):
        """O relato do bug deve entrar pelo user_prompt, e só por ele."""
        assert "{bug_report}" in prompt["user_prompt"], (
            "O user_prompt precisa conter a variável {bug_report}"
        )
        assert "{bug_report}" not in prompt["system_prompt"], (
            "O system_prompt não deve repetir {bug_report} (erro do prompt v1)"
        )

    def test_system_prompt_has_no_stray_braces(self, prompt):
        """Chaves soltas no system_prompt quebrariam o ChatPromptTemplate no push."""
        system_prompt = prompt["system_prompt"]

        assert "{" not in system_prompt and "}" not in system_prompt, (
            "O system_prompt contém chaves, que o ChatPromptTemplate interpretaria "
            "como variáveis de entrada"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
