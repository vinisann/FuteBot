# FuteBot

FuteBot é uma aplicação em **Streamlit** para acompanhar jogos da Copa do Mundo, consultar histórico de partidas, simular torneios e gerar previsões estatísticas com um modelo baseado em **Poisson + ELO**, com uma camada conservadora de **calibração incremental**.

O projeto foi pensado para funcionar tanto online quanto offline: quando há conexão e token configurado, ele sincroniza dados externos; quando não há, continua operando com dados locais de fallback.

## Principais recursos

- Acompanhamento de partidas da Copa do Mundo 2026.
- Modo offline com dados locais de fallback.
- Sincronização de partidas finalizadas via OpenFootball.
- Camada opcional de tempo real e agenda futura via Football-Data.org.
- Previsões pré-jogo com probabilidades de vitória, empate e derrota.
- Estimativa de gols esperados e placar mais provável.
- Simulação de partidas e torneio.
- Página de estatísticas por seleção.
- Histórico de Copas com filtros por edição e fase.
- Página de acurácia com backtest temporal sem vazamento de dados.
- Calibração incremental baseada em previsões salvas antes dos jogos.
- Documentação estática e fluxograma Mermaid em `docs/`.

## Stack

- Python
- Streamlit
- Pandas
- NumPy
- SciPy
- Plotly
- SQLite
- Requests
- Pytest

## Como rodar localmente

Crie e ative um ambiente virtual:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Instale as dependências:

```powershell
pip install -r requirements.txt
```

Execute o app:

```powershell
streamlit run app.py
```

Na primeira execução, o banco SQLite local é criado automaticamente em `data/futebot.db`.

## Token opcional da API

O app funciona sem token em modo offline. Para usar a camada de tempo real do Football-Data.org, configure a variável `FOOTBALL_DATA_API_KEY`.

Opção 1: variável de ambiente no PowerShell:

```powershell
$env:FOOTBALL_DATA_API_KEY="seu-token-aqui"
```

Opção 2: secrets do Streamlit:

```toml
# .streamlit/secrets.toml
FOOTBALL_DATA_API_KEY = "seu-token-aqui"
```

Nunca versione `.streamlit/secrets.toml`. O repositório inclui apenas `.streamlit/secrets.example.toml`, sem token real.

## Fontes de dados

O FuteBot trabalha com diferentes fontes e marca a origem dos dados no banco:

- `seed`: dados locais de fallback para uso offline.
- `openfootball`: partidas finalizadas importadas do projeto OpenFootball.
- `api`: partidas sincronizadas via Football-Data.org.
- `manual`: dados inseridos ou atualizados localmente.

O app usa:

- [OpenFootball World Cup JSON](https://github.com/openfootball/worldcup.json) para resultados finalizados quando disponível.
- [Football-Data.org](https://www.football-data.org/) como camada opcional para jogos ao vivo e partidas futuras.

## Como o modelo funciona

O modelo estatístico combina:

- Histórico de partidas finalizadas.
- Força ofensiva e defensiva por seleção.
- Ajuste por ELO.
- Distribuição de Poisson para estimar placares.
- Matriz de probabilidades para calcular vitória, empate e derrota.

Para evitar vazamento de dados, a página de acurácia prevê cada partida usando apenas jogos anteriores à data daquela partida.

## Calibração incremental

Além do modelo base, o projeto salva previsões pré-jogo em uma tabela local chamada `previsoes_partidas`.

Quando a partida termina, o app compara:

- placar previsto;
- placar real;
- acerto de resultado;
- acerto de placar;
- erro de gols;
- Brier Score.

Esses dados alimentam uma calibração conservadora, com limites para evitar que poucos jogos distorçam o modelo. Jogos de fallback `seed`, jogos simulados e partidas sem placar real não entram na calibração.

## Estrutura do projeto

```text
FuteBot/
  app.py                         # Página principal Streamlit
  pages/                         # Páginas secundárias do app
  src/
    api_client.py                # Cliente Football-Data.org
    config.py                    # Leitura de secrets/env vars
    database.py                  # SQLite, sync e persistência
    ML_models.py                 # Modelo Poisson/ELO e simulações
    model_calibration.py         # Calibração incremental
    statistics.py                # Estatísticas agregadas por seleção
    utils.py                     # Bandeiras, nomes e helpers visuais
  tests/                         # Testes automatizados
  docs/
    index.html                   # Documentação estática
    architecture-flow.mmd        # Fluxograma Mermaid
  data/
    .gitkeep                     # Mantém a pasta no Git
```

## Documentação

Além deste README:

- `docs/index.html`: documentação estática do projeto.
- `docs/architecture-flow.mmd`: fluxograma Mermaid da arquitetura.

O Mermaid pode ser visualizado diretamente pelo GitHub ou por extensões compatíveis.

## Testes e QA

Instale as dependências de desenvolvimento:

```powershell
pip install -r requirements-dev.txt
```

Rode as verificações principais:

```powershell
$env:PYTHONIOENCODING="utf-8"
python -m compileall app.py src pages
pytest -q
```

Os testes cobrem pontos como:

- ausência de token hardcoded;
- exclusão de dados `seed` da acurácia real;
- histórico temporal sem vazamento de dados;
- integridade do banco SQLite;
- deduplicação de partidas sincronizadas;
- status especiais da API;
- calibração incremental.

## Deploy no Streamlit Community Cloud

Para publicar:

1. Suba o repositório para o GitHub.
2. Acesse [Streamlit Community Cloud](https://share.streamlit.io/).
3. Crie um app apontando para:
   - repositório: `vinisann/FuteBot`;
   - branch: `main`;
   - arquivo principal: `app.py`.
4. Configure o secret `FOOTBALL_DATA_API_KEY` nas configurações do app, se quiser usar a API.

O app também funciona sem token, usando o modo offline/local.

## Segurança e versionamento

Arquivos sensíveis e locais ficam fora do Git:

- `.streamlit/secrets.toml`
- `.env`
- `data/*.db`
- caches Python e de testes
- `scratch/`

O banco SQLite é recriado localmente no primeiro uso, então o repositório pode ser clonado e executado sem versionar dados locais.

## Limitações conhecidas

- O SQLite local não deve ser tratado como banco permanente em deploy público.
- Em ambientes como Streamlit Cloud, o banco pode ser recriado após reinícios ou redeploys.
- A camada de tempo real depende da disponibilidade e limites do Football-Data.org.
- A calibração incremental melhora com o acúmulo de previsões avaliadas, mas começa neutra quando ainda há pouca amostra.

## Roadmap possível

- Persistência externa com Postgres, Supabase ou Neon.
- Dashboard específico para calibração do modelo.
- Comparação visual entre modelo base e modelo calibrado.
- Exportação de relatórios de acurácia.
- Melhorias de responsividade mobile.

## Licença

Este projeto ainda não define uma licença explícita. Antes de uso público amplo, adicione uma licença adequada ao objetivo do repositório.
