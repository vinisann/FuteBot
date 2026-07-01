def _favorite(prediction):
    options = [
        (
            "mandante",
            prediction.get("mandante", prediction.get("mandante_nome", "Mandante")),
            float(prediction.get("prob_vitoria_mandante", prediction.get("prob_mandante", 0.0))),
        ),
        ("empate", "Empate", float(prediction.get("prob_empate", 0.0))),
        (
            "visitante",
            prediction.get("visitante", prediction.get("visitante_nome", "Visitante")),
            float(prediction.get("prob_vitoria_visitante", prediction.get("prob_visitante", 0.0))),
        ),
    ]
    return max(options, key=lambda item: item[2])


def _confidence_label(probability):
    if probability >= 0.60:
        return "alta"
    if probability >= 0.40:
        return "media"
    return "baixa"


def _factor(tipo, titulo, descricao, impacto="neutro"):
    return {
        "tipo": tipo,
        "titulo": titulo,
        "descricao": descricao,
        "impacto": impacto,
    }


def _format_weights(weights):
    if not weights:
        return ""
    ordered = sorted(weights.items(), key=lambda item: item[1], reverse=True)
    return ", ".join(f"{name}: {value * 100:.0f}%" for name, value in ordered)


def explain_prediction(prediction):
    side, label, probability = _favorite(prediction)
    confidence = _confidence_label(probability)
    mandante = prediction.get("mandante", prediction.get("mandante_nome", "Mandante"))
    visitante = prediction.get("visitante", prediction.get("visitante_nome", "Visitante"))
    xg_m = float(prediction.get("xG_mandante", 0.0))
    xg_v = float(prediction.get("xG_visitante", 0.0))

    fatores = [
        _factor(
            "xg",
            "Gols esperados",
            f"{mandante} tem xG {xg_m:.2f} contra {xg_v:.2f} de {visitante}.",
            "positivo" if xg_m > xg_v else "negativo" if xg_m < xg_v else "neutro",
        )
    ]
    alertas = []

    elo_m = prediction.get("elo_mandante_usado")
    elo_v = prediction.get("elo_visitante_usado")
    if elo_m is not None and elo_v is not None:
        diff = float(elo_m) - float(elo_v)
        fatores.append(
            _factor(
                "elo",
                "Forca Elo",
                f"Diferenca de ELO usada no modelo: {diff:+.0f} pontos.",
                "positivo" if diff > 50 else "negativo" if diff < -50 else "neutro",
            )
        )

    if prediction.get("modelo_ensemble"):
        fatores.append(
            _factor(
                "ensemble",
                "Ensemble ponderado",
                f"Pesos usados: {_format_weights(prediction.get('pesos_ensemble', {}))}.",
                "positivo",
            )
        )

    if prediction.get("modelo_calibrado"):
        fatores.append(
            _factor(
                "calibracao",
                "Calibracao incremental",
                "Ajustes locais foram aplicados com base em previsoes ja avaliadas.",
                "positivo",
            )
        )
    else:
        fatores.append(
            _factor(
                "calibracao",
                "Calibracao incremental",
                "A calibracao ainda esta neutra ou sem amostra suficiente.",
                "neutro",
            )
        )
        alertas.append("Calibracao incremental inativa ou com amostra insuficiente.")

    if prediction.get("modelo_dixon_coles"):
        fatores.append(
            _factor(
                "dixon_coles",
                "Dixon-Coles",
                f"Correcao de placares baixos ativa com rho {float(prediction.get('rho_dixon_coles', 0.0)):.2f}.",
                "positivo",
            )
        )

    if prediction.get("modelo_com_contexto"):
        fatores.append(
            _factor(
                "contexto",
                "Contexto do jogo",
                str(prediction.get("contexto_resumo", "Contexto aplicado.")),
                "positivo",
            )
        )
    else:
        alertas.append("Contexto neutro ou sem dados contextuais confiaveis.")

    fator_zebra = float(prediction.get("fator_zebra", 0.0) or 0.0)
    if fator_zebra > 0:
        fatores.append(
            _factor(
                "zebra",
                "Volatilidade/zebra",
                f"Redistribuicao conservadora de {fator_zebra * 100:.1f}% para cenarios menos provaveis.",
                "alerta",
            )
        )

    resumo = (
        f"{label} aparece como resultado mais provavel com {probability * 100:.1f}% "
        f"de probabilidade. A confianca do sinal e {confidence}."
    )
    return {
        "resumo": resumo,
        "fatores": fatores,
        "confianca": confidence,
        "alertas": alertas,
    }


def format_explanation_markdown(explanation):
    lines = [
        "#### Por que essa previsão?",
        f"**Resumo:** {explanation.get('resumo', '')}",
        f"**Confiança:** {explanation.get('confianca', 'baixa')}",
        "",
        "**Fatores:**",
    ]
    for factor in explanation.get("fatores", []):
        lines.append(
            f"- **{factor.get('titulo', factor.get('tipo', 'Fator'))}:** "
            f"{factor.get('descricao', '')} _Impacto: {factor.get('impacto', 'neutro')}._"
        )
    alertas = explanation.get("alertas", [])
    if alertas:
        lines.extend(["", "**Alertas:**"])
        lines.extend(f"- {alert}" for alert in alertas)
    return "\n".join(lines)
