"""Fonte única dos nomes, tipos, domínios e unidades das 28 colunas."""

# original, nome interno, rótulo, natureza, unidade, significado
_COLUMNS = [
    ("Team_ID", "time_id", "ID do time", "identificador", "sem unidade", "Identifica a equipe; números usados como código, não como medida."),
    ("Game_ID", "partida_id", "ID da partida", "identificador", "sem unidade", "Identifica a partida; a chave do registro é time + partida."),
    ("GAME_DATE", "data_partida", "Data da partida", "data", "data", "Dia do jogo, sem horário ou fuso informado."),
    ("MATCHUP", "confronto", "Confronto", "texto", "sem unidade", "Equipes e mando: vs. indica casa; @ indica visitante."),
    ("WL", "venceu", "Vitória", "binária", "0 ou 1", "Resultado da partida: 1 = vitória, 0 = derrota."),
    ("W", "vitorias_acumuladas", "Vitórias acumuladas", "contagem acumulada", "vitórias", "Vitórias acumuladas até a partida, incluindo o jogo atual."),
    ("L", "derrotas_acumuladas", "Derrotas acumuladas", "contagem acumulada", "derrotas", "Derrotas acumuladas até a partida, incluindo o jogo atual."),
    ("W_PCT", "proporcao_vitorias", "Aproveitamento acumulado", "proporção acumulada", "%", "Vitórias divididas por vitórias + derrotas; armazenada em [0, 1]."),
    ("MIN", "minutos_equipe", "Minutos da equipe", "contagem", "minutos-equipe/jogo", "Valores compatíveis com soma de minutos dos jogadores (240, 265, 290). Interpretação ainda a confirmar na fonte."),
    ("FGM", "arremessos_quadra_convertidos", "Cestas de quadra", "contagem", "cestas/jogo", "Arremessos de quadra convertidos, incluindo cestas de três."),
    ("FGA", "arremessos_quadra_tentados", "Tentativas de quadra", "contagem", "tentativas/jogo", "Todos os arremessos de quadra tentados."),
    ("FG_PCT", "aproveitamento_quadra", "Aproveitamento de quadra", "proporção", "%", "Razão FGM/FGA, armazenada em [0, 1]."),
    ("FG3M", "cestas_tres_convertidas", "Cestas de três", "contagem", "cestas de três/jogo", "Arremessos de três pontos convertidos."),
    ("FG3A", "cestas_tres_tentadas", "Tentativas de três", "contagem", "tentativas/jogo", "Arremessos de três pontos tentados."),
    ("FG3_PCT", "aproveitamento_tres", "Aproveitamento de três", "proporção", "%", "Razão FG3M/FG3A, armazenada em [0, 1]."),
    ("FTM", "lances_livres_convertidos", "Lances livres convertidos", "contagem", "lances livres/jogo", "Lances livres convertidos."),
    ("FTA", "lances_livres_tentados", "Lances livres tentados", "contagem", "tentativas/jogo", "Lances livres tentados."),
    ("FT_PCT", "aproveitamento_lance_livre", "Aproveitamento de lance livre", "proporção", "%", "Razão FTM/FTA, armazenada em [0, 1]."),
    ("OREB", "rebotes_ofensivos", "Rebotes ofensivos", "contagem", "rebotes/jogo", "Rebotes recuperados no ataque."),
    ("DREB", "rebotes_defensivos", "Rebotes defensivos", "contagem", "rebotes/jogo", "Rebotes recuperados na defesa."),
    ("REB", "rebotes", "Rebotes", "contagem", "rebotes/jogo", "Total de rebotes ofensivos e defensivos."),
    ("AST", "assistencias", "Assistências", "contagem", "assistências/jogo", "Passes registrados como assistência."),
    ("STL", "roubos_bola", "Roubos de bola", "contagem", "roubos/jogo", "Roubos de bola registrados; zero é válido."),
    ("BLK", "tocos", "Tocos", "contagem", "tocos/jogo", "Bloqueios de arremesso; zero é válido."),
    ("TOV", "desperdicios_bola", "Perdas de bola", "contagem", "perdas/jogo", "Posses perdidas (turnovers)."),
    ("PF", "faltas_pessoais", "Faltas pessoais", "contagem", "faltas/jogo", "Faltas pessoais cometidas pela equipe."),
    ("PTS", "pontos", "Pontos", "contagem", "pontos/jogo", "Pontuação total da equipe na partida."),
    ("TEAM_NAME", "time", "Time", "categórica", "sem unidade", "Nome da equipe, associado a um ID de time."),
]

SCHEMA = {
    name: {"original": original, "label": label, "kind": kind,
           "unit": unit, "description": description}
    for original, name, label, kind, unit, description in _COLUMNS
}

RENAME = {v["original"]: k for k, v in SCHEMA.items()}

NUMERIC = tuple(k for k, v in SCHEMA.items() if v["kind"] in
                {"contagem", "contagem acumulada", "proporção", "proporção acumulada", "binária"})
                
MEASUREMENTS = tuple(k for k, v in SCHEMA.items() if v["kind"] in {"contagem", "proporção"})


def dictionary_rows(df):
    rows = []
    for name, meta in SCHEMA.items():
        kind = meta["kind"]
        domain = ("inteiro ≥ 0" if "contagem" in kind else "[0, 1]" if "proporção" in kind
                  else "{0, 1}" if kind == "binária" else "data válida" if kind == "data"
                  else "código / texto não vazio")
        rows.append({"Original": meta["original"], "Nome interno": name,
                     "Rótulo": meta["label"], "Natureza": kind,
                     "Tipo Python/pandas": str(df[name].dtype), "Unidade": meta["unit"],
                     "Domínio": domain, "Significado": meta["description"]})
    return rows

