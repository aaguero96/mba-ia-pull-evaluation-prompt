# Pull, Otimização e Avaliação de Prompts com LangChain e LangSmith

Projeto do desafio de Prompt Engineering: puxar um prompt de baixa qualidade do **LangSmith Prompt Hub**, refatorá-lo com técnicas avançadas, republicá-lo e provar a melhoria com 5 métricas automáticas (LLM-as-Judge), todas com nota mínima de **0.8**.

O prompt converte **relatos de bugs** em **User Stories** prontas para o backlog.

| | |
|---|---|
| Prompt de origem (v1) | `leonanluppi/bug_to_user_story_v1` |
| Prompt otimizado (v2) | `test-5728e9689/bug_to_user_story_v2` (público) |
| Dataset de avaliação | 15 bugs (5 simples, 7 médios, 3 complexos) |
| LLM de resposta e de avaliação | `qwen3.8:27b` servido por Ollama |
| Plataforma | LangSmith (Hub, Datasets e Tracing) |

---

## Stack

| Camada | Tecnologia |
|---|---|
| Linguagem | Python 3.12 |
| Framework | LangChain 0.3.13 |
| Gestão de prompts | LangSmith Prompt Hub (`langchain.hub`) |
| Avaliação e tracing | LangSmith 0.2.7 |
| Formato dos prompts | YAML |
| Testes | pytest 8.3.4 |

### Provedor de LLM

O projeto é **plugável por `.env`** e suporta três configurações, sem alterar uma linha de código:

1. **Ollama local** *(usado neste projeto)* — o Ollama expõe uma API compatível com OpenAI, então basta usar `LLM_PROVIDER=openai` e apontar `OPENAI_API_BASE` para o servidor local. O `src/utils.py` (arquivo que o desafio proíbe alterar) instancia `ChatOpenAI`, que lê essa variável do ambiente. Custo zero.
2. **Google Gemini** — `LLM_PROVIDER=google` com `GOOGLE_API_KEY`.
3. **OpenAI** — `LLM_PROVIDER=openai` com `OPENAI_API_KEY` real.

O enunciado não fixa modelos ("*Este desafio não fixa modelos... o provedor que você escolher*"), e a avaliação continua 100% no LangSmith em qualquer das três opções.

---

## Estrutura

```
├── .env.example                      # Template das variáveis de ambiente
├── requirements.txt                  # Dependências Python
├── prompts/
│   ├── bug_to_user_story_v1.yml      # Prompt ruim, baixado do Hub pelo pull
│   └── bug_to_user_story_v2.yml      # Prompt otimizado (entregável)
├── datasets/
│   └── bug_to_user_story.jsonl       # 15 exemplos de bugs (não alterado)
├── src/
│   ├── pull_prompts.py               # Pull do Hub  -> implementado
│   ├── push_prompts.py               # Push ao Hub  -> implementado
│   ├── evaluate.py                   # Avaliação    -> pronto, não alterado
│   ├── metrics.py                    # 5 métricas   -> pronto, não alterado
│   └── utils.py                      # Auxiliares   -> pronto, não alterado
└── tests/
    └── test_prompts.py               # 6 testes obrigatórios + 2 extras
```

---

## A) Técnicas Aplicadas (Fase 2)

### Diagnóstico do prompt v1

O pull (`python src/pull_prompts.py`) trouxe este prompt:

```
system: Você é um assistente que ajuda a transformar relatos de bugs de usuários em
        tarefas para desenvolvedores.
        Analise o relato de bug abaixo e crie uma user story a partir dele.
        Relato de Bug:
        --- {bug_report} ---
        User Story gerada:
user:   {bug_report}
```

Problemas identificados:

| # | Problema | Efeito na saída |
|---|---|---|
| 1 | `{bug_report}` duplicado no system **e** no user prompt | O relato chega duas vezes ao modelo, desperdiçando contexto e confundindo o papel de cada mensagem |
| 2 | Persona genérica ("um assistente") | Saída sem vocabulário de produto, oscilando entre tarefa técnica e user story |
| 3 | Instrução vaga ("crie uma user story") | Formato varia a cada chamada: às vezes parágrafo, às vezes lista |
| 4 | Zero exemplos | O modelo não sabe o nível de detalhe esperado |
| 5 | Nenhum formato de saída definido | Sem "Como um... eu quero... para que...", sem critérios de aceitação |
| 6 | Nenhuma regra de comportamento | Preâmbulos ("Aqui está a user story:"), perguntas de volta, invenção de dados |
| 7 | Nenhum tratamento de edge case | Relatos complexos (vários bugs num texto só) viram uma story rasa |

### Técnicas escolhidas

Foram aplicadas **4 técnicas** — Few-shot (obrigatória) e outras três.

#### 1. Role Prompting

**Por quê:** as métricas Clarity e Precision são julgadas contra referências escritas em linguagem de produto. Um "assistente" genérico escreve como suporte técnico; um Product Manager escreve valor de negócio.

**Como apliquei:**

```
Você é um Product Manager Sênior com 10 anos de experiência em times ágeis de produto
digital (e-commerce, SaaS, ERP, CRM e aplicativos mobile). Você é referência no time por
transformar relatos confusos de bugs em User Stories que o time de desenvolvimento
consegue pegar e implementar sem fazer nenhuma pergunta de volta.
```

Os domínios citados são exatamente os do dataset, e o trecho "sem fazer nenhuma pergunta de volta" já elimina o comportamento de pedir esclarecimento.

#### 2. Chain of Thought (CoT) — silencioso

**Por quê:** transformar bug em user story exige inferência em cadeia (quem é afetado → o que quebrou → qual o comportamento correto → qual o benefício). Mas raciocínio visível na resposta destrói Clarity e Precision, porque a referência contém só a story final.

**Como apliquei:** seis etapas explícitas de raciocínio, declaradas como internas.

```
# PROCESSO DE RACIOCÍNIO (pense passo a passo, em silêncio)
1. Identifique QUEM é afetado pelo bug...
2. Identifique O QUE está quebrado hoje e qual é o comportamento CORRETO esperado.
3. Traduza o comportamento correto em BENEFÍCIO de negócio (o "para que").
4. Classifique o relato em NÍVEL 1, 2 ou 3...
5. Derive um critério de aceitação para CADA sintoma citado...
6. Revise: todo dado numérico do relato foi aproveitado? Inventei alguma informação...

Depois da etapa 6, escreva APENAS a User Story final.
```

A etapa 6 é uma auto-revisão anti-alucinação, que ataca diretamente a métrica Precision.

#### 3. Skeleton of Thought

**Por quê:** o dataset tem três níveis de complexidade com referências de formatos bem diferentes — os relatos simples esperam story + critérios; os complexos esperam blocos `=== USER STORY PRINCIPAL ===`, `=== CRITÉRIOS TÉCNICOS ===`, `=== TASKS TÉCNICAS SUGERIDAS ===`. Um formato único perderia F1 nas duas pontas: verboso demais nos simples, raso demais nos complexos.

**Como apliquei:** um esqueleto de saída por nível, com regra de classificação.

```
## NÍVEL 1 — relato simples (uma ou duas frases, um único sintoma, sem detalhe técnico)
Como um <persona>, eu quero <capacidade desejada>, para que <benefício>.
Critérios de Aceitação:
- Dado que <contexto inicial>
...
## NÍVEL 2 — relato técnico (passos de reprodução, logs, números...)
Mesma estrutura do Nível 1, com 5 a 7 critérios, seguida de: Contexto Técnico:
## NÍVEL 3 — relato complexo (dois ou mais problemas distintos, seções numeradas)
=== USER STORY PRINCIPAL === / === CRITÉRIOS DE ACEITAÇÃO === (A, B, C, D) /
=== CRITÉRIOS TÉCNICOS === / === CONTEXTO DO BUG === / === TASKS TÉCNICAS SUGERIDAS ===
```

#### 4. Few-shot Learning *(obrigatório)*

**Por quê:** descrever o formato não basta; o modelo precisa ver o tom, a densidade e o vocabulário. Três exemplos cobrem exatamente os três níveis do esqueleto.

**Como apliquei:** exemplos `ENTRADA:` / `SAÍDA:` **inéditos** — redefinição de senha (nível 1), CSV com acentos quebrados (nível 2) e upload de documentos com duas falhas (nível 3). São bugs que **não** estão no dataset de avaliação, de propósito: o ganho vem da generalização do padrão, não de decorar respostas.

```
## EXEMPLO 1 — Nível 1
ENTRADA:
Ao clicar em "Esqueci minha senha" na tela de login, nada acontece.

SAÍDA:
Como um usuário que esqueceu a senha, eu quero solicitar a redefinição pela tela de
login, para que eu possa recuperar o acesso à minha conta sem depender do suporte.

Critérios de Aceitação:
- Dado que estou na tela de login
- Quando clico em "Esqueci minha senha"
...
```

### Outros requisitos do prompt otimizado

- **System vs User Prompt:** o system carrega papel, processo, formato, regras e exemplos; o user carrega **apenas** o relato. A duplicação de `{bug_report}` do v1 foi eliminada — há inclusive um teste automatizado que falha se ela voltar.
- **Regras explícitas de comportamento:** 11 regras, entre elas "comece diretamente por *Como um*", "nunca faça perguntas", "não invente dados — use marcador entre colchetes como `[nome do gateway de pagamento]`" e "critérios de aceitação descrevem comportamento observável; detalhe de implementação vai para Contexto Técnico".
- **Tratamento de edge cases:** 8 casos cobertos — relato vago, relato sem ator humano (a persona vira "o sistema"), pedido de melhoria em vez de defeito, relato multiproblema, questão de segurança (registra severidade e OWASP), relato em inglês, relato que já aponta a causa raiz e relato com impacto de negócio.
- **Regra de metas numéricas:** quando o relato traz um número que descreve o *problema* (ex.: "demora mais de 2 minutos", "timeout em 120s"), o critério exige um alvo **melhor** e realista, e o número ruim vai para o Contexto Técnico. Essa regra nasceu de uma iteração: sem ela, o modelo aceitava os 120 s do timeout como meta.

---

## B) Resultados Finais

### Dashboard do LangSmith

- **Prompt v2 publicado (público):** https://smith.langchain.com/hub/test-5728e9689/bug_to_user_story_v2
- **Projeto de tracing:** `bug-to-user-story-optimization`
- **Dataset de avaliação:** `bug-to-user-story-optimization-eval` (15 exemplos)

### Comparativo v1 × v2

<!-- RESULTADOS -->

### Iterações

<!-- ITERACOES -->

---

## C) Como Executar

### Pré-requisitos

- **Python 3.12** (as dependências são pinadas e não suportam 3.13+)
- Conta no **LangSmith** com uma API Key
- Um provedor de LLM: **Ollama** rodando localmente, ou chave da OpenAI, ou chave do Google Gemini

### 1. Clonar e criar o ambiente virtual

```bash
git clone https://github.com/aaguero96/mba-ia-pull-evaluation-prompt.git
cd mba-ia-pull-evaluation-prompt

python3 -m venv venv
source venv/bin/activate        # No Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configurar as variáveis de ambiente

```bash
cp .env.example .env            # No Windows: copy .env.example .env
```

Preencha no `.env`:

```env
LANGSMITH_API_KEY=lsv2_pt_...
LANGSMITH_PROJECT=bug-to-user-story-optimization
USERNAME_LANGSMITH_HUB=seu-handle-do-hub

# Ollama local (padrão deste projeto)
LLM_PROVIDER=openai
LLM_MODEL=qwen3.8:27b
EVAL_MODEL=qwen3.8:27b
OPENAI_API_KEY=ollama
OPENAI_API_BASE=http://localhost:11434/v1
```

> **Windows:** os scripts imprimem `✓` e `❌`. Se a saída for redirecionada para arquivo ou pipe, rode com `PYTHONUTF8=1` (ex.: `$env:PYTHONUTF8=1` no PowerShell) para evitar `UnicodeEncodeError`.

> O `USERNAME_LANGSMITH_HUB` é o handle do seu workspace. Ele é definido **uma única vez** pelo LangSmith e não pode ser trocado depois pela API.

### 3. Pull do prompt de baixa qualidade

```bash
python src/pull_prompts.py
```

Baixa `leonanluppi/bug_to_user_story_v1` e grava em `prompts/bug_to_user_story_v1.yml`.

### 4. Refatorar o prompt

O prompt otimizado já está em `prompts/bug_to_user_story_v2.yml`. Para iterar, edite esse arquivo e repita os passos 5 e 6.

### 5. Push do prompt otimizado

```bash
python src/push_prompts.py
```

Valida o YAML, monta o `ChatPromptTemplate` (system + human) e publica **público** em `{USERNAME_LANGSMITH_HUB}/bug_to_user_story_v2`, com descrição, README e tags (versão + técnicas aplicadas).

### 6. Avaliar

```bash
python src/evaluate.py
```

Cria o dataset no LangSmith (se ainda não existir), puxa o prompt v2 do Hub, roda os 15 exemplos e calcula as 5 métricas. Cada rodada faz 60 chamadas ao LLM (15 respostas + 45 julgamentos).

### 7. Rodar os testes

```bash
pytest tests/test_prompts.py -v
```

| Teste | O que garante |
|---|---|
| `test_prompt_has_system_prompt` | `system_prompt` existe, é string e não está vazio |
| `test_prompt_has_role_definition` | Há persona explícita ("Você é um Product Manager...") |
| `test_prompt_mentions_format` | Exige Markdown ou o formato padrão de User Story, com Critérios de Aceitação |
| `test_prompt_has_few_shot_examples` | Há ao menos 2 pares `ENTRADA:` / `SAÍDA:`, em quantidades iguais |
| `test_prompt_no_todos` | Nenhum `[TODO]`, `FIXME` ou `XXX` sobrou no texto |
| `test_minimum_techniques` | `techniques_applied` tem ≥ 2 técnicas, inclui Few-shot e passa no `validate_prompt_structure` do projeto |
| `test_user_prompt_has_bug_report_variable` *(extra)* | `{bug_report}` está no user prompt e **não** no system — o bug do v1 não volta |
| `test_system_prompt_has_no_stray_braces` *(extra)* | Nenhuma chave solta no system prompt, que quebraria o `ChatPromptTemplate` no push |
