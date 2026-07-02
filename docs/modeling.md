# Modelo estatistico do FuteBot

Este documento resume como o FuteBot calcula previsoes, como evita vazamento de dados e como mede a qualidade do modelo.

## Objetivo

O modelo nao tenta eliminar o acaso do futebol. A proposta e gerar probabilidades mais bem calibradas para:

- resultado 1X2;
- placar mais provavel;
- gols esperados;
- simulacao de partidas;
- simulacao de torneio;
- avaliacao de acuracia ao longo do tempo.

## Fontes usadas pelo modelo

O FuteBot usa apenas dados considerados confiaveis para aprendizado e avaliacao real:

- partidas finalizadas com placar valido;
- resultados sincronizados via OpenFootball;
- jogos sincronizados via Football-Data.org;
- dados manuais marcados como confiaveis.

Dados `seed` sao fallback offline e nao devem ser tratados como verdade estatistica para acuracia real, calibracao ou backtesting confiavel.

## Pipeline de previsao

1. Carrega historico real anterior ao jogo previsto.
2. Calcula forca ofensiva e defensiva das selecoes.
3. Aplica ELO atual ou ELO dinamico pre-jogo, dependendo do contexto.
4. Ajusta forma recente com peso maior para jogos mais novos.
5. Estima lambdas de gols esperados.
6. Aplica calibracao incremental quando ha amostra suficiente.
7. Aplica contexto de jogo quando disponivel.
8. Aplica sinais externos, como noticias, provaveis escalacoes e impacto conservador de jogadores.
9. Monta a matriz de placares por Poisson.
10. Aplica Dixon-Coles para corrigir placares baixos quando a variante exigir.
11. Calcula probabilidades de mandante, empate e visitante.
12. Combina variantes no ensemble ponderado para analise comparativa.

## Variantes avaliadas

A pagina de acuracia compara:

- `Baseline ELO simples`;
- `Base Poisson-ELO`;
- `ELO dinamico`;
- `ELO dinamico + forma/calibracao`;
- `ELO dinamico + forma/calibracao + Dixon-Coles`;
- `ELO dinamico + forma/calibracao + Dixon-Coles + contexto`;
- `Ensemble ponderado`.

O baseline ELO simples existe como regua minima. Se uma variante sofisticada perde consistentemente para ele em Brier Score ou Log Loss, isso indica que o ajuste estatistico precisa ser revisto.

## Backtesting temporal

Para cada partida avaliada, o backtest usa apenas jogos anteriores a `data_hora` daquela partida. Isso evita que o proprio resultado previsto entre no historico usado pela previsao.

O backtest tambem guarda:

- quantidade de jogos historicos usados;
- cutoff temporal do historico;
- faixa de confianca da previsao;
- lado favorito;
- probabilidade atribuida ao resultado real;
- indicador de zebra quando favorito forte erra.

## Metricas

As principais metricas sao:

- **Acuracia 1X2:** percentual de acerto de vencedor/empate.
- **Placar exato:** percentual de acerto do placar completo.
- **Erro de gols:** distancia absoluta entre placar previsto e real.
- **Brier Score:** mede qualidade probabilistica para 1X2; menor e melhor.
- **Log Loss:** pune previsoes muito confiantes que erram; menor e melhor.
- **Erro de calibracao:** distancia entre confianca media e acuracia real.
- **Overconfidence:** quando a confianca media fica acima da acuracia observada.

## Calibracao incremental

Previsoes pre-jogo podem ser salvas no SQLite antes de uma partida terminar. Quando o jogo vira `FINISHED`, o app avalia a previsao e salva:

- placar real;
- acerto 1X2;
- acerto de placar;
- erro de gols;
- Brier Score.

A calibracao so ativa com amostra minima e possui limites conservadores para evitar reacao exagerada a poucos jogos.

## Sinais externos e jogadores

O modelo pode considerar:

- noticias;
- provaveis escalacoes;
- termos ligados a lesao, suspensao, pressao e estabilidade;
- impacto conservador de jogadores/desfalques com caps pequenos.

Esses sinais sao auxiliares. Eles ajustam o modelo de forma limitada e nunca substituem o historico estatistico.

## Limitacoes conhecidas

- Futebol tem alta variancia; mesmo um bom modelo erra jogos isolados.
- Escalacoes e noticias podem estar incompletas ou desatualizadas.
- Ratings de jogadores sao proxies conservadores, nao uma base proprietaria completa.
- Odds reais ainda sao usadas apenas como referencia quando disponiveis, nao como motor principal.
- Amostras pequenas devem ser interpretadas como diagnostico inicial, nao conclusao definitiva.

## Proximas evolucoes possiveis

- Integrar xG real e estatisticas avancadas de finalizacao.
- Melhorar ratings de jogadores com fonte externa confiavel.
- Adicionar lesoes/suspensoes estruturadas.
- Comparar contra odds reais com historico.
- Criar selecao automatica de modelo por fase ou segmento.
- Usar relatorio de qualidade para recomendar ajustes estatisticos no app.
