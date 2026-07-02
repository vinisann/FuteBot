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
- ELO dinâmico temporal para backtesting histórico sem usar ratings atuais em jogos antigos.
- Forma recente ponderada por recência.
- Correção Dixon-Coles para placares baixos.
- Contexto de jogo, sinais externos, prováveis escalações e impacto conservador de jogadores.
- Ensemble ponderado entre variantes do modelo.
- Backtesting profundo com baseline ELO, Brier Score, Log Loss, curva de calibração, segmentos fracos e alerta de overconfidence.
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

## Imagens do projeto:

Home:
(Página inicial da aplicação)
<img width="1907" height="966" alt="image" src="https://github.com/user-attachments/assets/12592f0d-e592-48a0-a873-b492f790bd22" />

Estatísticas:
(Onde contêm mais detalhamentos voltado a parte estatística do projeto
<img width="1917" height="956" alt="image" src="https://github.com/user-attachments/assets/594ab8e5-0bb4-4b1b-840e-cafae2326d68" />
(Detalhamento da parte de estatísticas):
<img width="1412" height="662" alt="image" src="https://github.com/user-attachments/assets/ec61a082-9419-44fd-8265-4df0e892cd8c" />

Previsões:
(Nessa parte você consegue simular um confronto entre duas equipes de sua escolha)
<img width="1842" height="908" alt="image" src="https://github.com/user-attachments/assets/9da33cca-1302-4c8e-bb7b-6599083ff33f" />

Histórico:
(Detalhamento dos jogos históricos com dados atualizados de 2026 + histórico até 2018)
<img width="1912" height="957" alt="image" src="https://github.com/user-attachments/assets/c46770e3-d96e-4eb3-a656-d3a1c1c00199" />

Acurácia:
(Faz um comparativo entre previsões x jogos finalizados e se auto calibra para próximo confronto)
<img width="1918" height="952" alt="image" src="https://github.com/user-attachments/assets/6d8518aa-28f9-4dc0-8c85-2c433dd01e8b" />

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
- Ajuste por ELO atual e ELO dinâmico temporal.
- Forma recente com peso maior para jogos mais novos.
- Distribuição de Poisson para estimar placares.
- Correção Dixon-Coles para melhorar a calibração de placares baixos.
- Contexto de jogo, como fase, descanso, clima/venue quando disponível.
- Sinais externos: notícias, prováveis escalações e impacto de jogadores/desfalques.
- Ensemble ponderado por desempenho histórico.
- Matriz de probabilidades para calcular vitória, empate e derrota.

Para evitar vazamento de dados, a página de acurácia prevê cada partida usando apenas jogos anteriores à data daquela partida.

## Backtesting profundo

A página de acurácia compara diferentes variações do modelo:

- `Baseline ELO simples`: régua mínima para saber se o modelo estatístico realmente agrega valor.
- `Base Poisson-ELO`: modelo base com forças ofensivas/defensivas e Poisson.
- `ELO dinâmico`: ratings pré-jogo calculados em ordem cronológica.
- `ELO dinâmico + forma/calibração`: adiciona forma recente e calibração incremental.
- `ELO dinâmico + forma/calibração + Dixon-Coles`: corrige placares baixos.
- `ELO dinâmico + forma/calibração + Dixon-Coles + contexto`: inclui contexto do jogo.
- `Ensemble ponderado`: combina variantes conforme desempenho histórico.

O backtesting mede acurácia 1X2, placar exato, erro de gols, Brier Score, Log Loss, erro de calibração, overconfidence e desempenho por segmentos como Copa, fase e faixa de confiança.

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
    dynamic_elo.py               # ELO temporal e forma recente
    dixon_coles.py               # Correção Dixon-Coles
    ML_models.py                 # Modelo Poisson/ELO e simulações
    model_calibration.py         # Calibração incremental
    model_ensemble.py            # Ensemble ponderado
    model_evaluation.py          # Backtesting profundo e métricas
    match_context.py             # Contexto de jogo
    player_impact.py             # Impacto conservador de jogadores/desfalques
    statistics.py                # Estatísticas agregadas por seleção
    utils.py                     # Bandeiras, nomes e helpers visuais
  tests/                         # Testes automatizados
  docs/
    index.html                   # Documentação estática
    architecture-flow.mmd        # Fluxograma Mermaid
    modeling.md                  # Detalhamento do modelo estatístico
  data/
    .gitkeep                     # Mantém a pasta no Git
```

## Documentação

Além deste README:

- `docs/index.html`: documentação estática do projeto.
- `docs/architecture-flow.mmd`: fluxograma Mermaid da arquitetura.
- `docs/modeling.md`: detalhes do modelo, métricas, backtesting e limitações.

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
- ELO dinâmico, forma recente, Dixon-Coles, contexto, ensemble, jogadores/desfalques e backtesting profundo.

## Segurança e versionamento

Arquivos sensíveis e locais ficam fora do Git:

- `.streamlit/secrets.toml`
- `.env`
- `data/*.db`
- caches Python e de testes
- `scratch/`

O banco SQLite é recriado localmente no primeiro uso, então o repositório pode ser clonado e executado sem versionar dados locais.

## Licença

MIT
