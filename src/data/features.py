# src/data/features.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import pandas as pd


@dataclass
class FeatureConfig:
    rolling_n: int = 5
    use_season_baseline: bool = True  # usar média da liga por temporada
    min_periods: int = 1              # rolling min periods
    eps: float = 1e-9                 # evitar divisão por zero


def _ensure_datetime(df: pd.DataFrame, col: str = "data") -> pd.DataFrame:
    df = df.copy()
    df[col] = pd.to_datetime(df[col], errors="coerce")
    if df[col].isna().any():
        raise ValueError(f"Datas inválidas encontradas em '{col}'.")
    return df


def _basic_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria colunas auxiliares:
    - gf_home, ga_home
    - gf_away, ga_away
    - y (H/D/A)
    """
    df = df.copy()

    df["gf_home"] = df["gols_mandante"].astype(float)
    df["ga_home"] = df["gols_visitante"].astype(float)

    df["gf_away"] = df["gols_visitante"].astype(float)
    df["ga_away"] = df["gols_mandante"].astype(float)

    # Se não tiver resultado, cria a partir do placar
    if "resultado" not in df.columns or df["resultado"].isna().all():
        def _res(row):
            if row["gols_mandante"] > row["gols_visitante"]:
                return "H"
            if row["gols_mandante"] < row["gols_visitante"]:
                return "A"
            return "D"
        df["resultado"] = df.apply(_res, axis=1)

    df["resultado"] = df["resultado"].astype(str).str.upper()
    return df


def _season_league_baselines(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    """
    Calcula baseline de liga (média de gols) por temporada:
    - league_gf_home_mean
    - league_gf_away_mean
    - league_total_goals_mean
    """
    df = df.copy()

    grp = df.groupby("ano_campeonato", observed=True)
    df["league_gf_home_mean"] = grp["gols_mandante"].transform("mean")
    df["league_gf_away_mean"] = grp["gols_visitante"].transform("mean")
    df["league_total_goals_mean"] = grp.apply(
        lambda g: (g["gols_mandante"].mean() + g["gols_visitante"].mean())
    ).reindex(df["ano_campeonato"]).values

    return df


def _home_advantage(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    """
    Vantagem de mando:
    - home_adv_season = média(gols_mandante - gols_visitante) por temporada
    - home_winrate_season = taxa de vitória mandante por temporada
    """
    df = df.copy()
    df["home_goal_diff"] = df["gols_mandante"].astype(float) - df["gols_visitante"].astype(float)
    df["home_win"] = (df["resultado"] == "H").astype(int)

    grp = df.groupby("ano_campeonato", observed=True)
    df["home_adv_season"] = grp["home_goal_diff"].transform("mean")
    df["home_winrate_season"] = grp["home_win"].transform("mean")

    # também uma baseline global (útil se você quiser usar como feature fixa)
    df["home_adv_global"] = df["home_goal_diff"].mean()
    df["home_winrate_global"] = df["home_win"].mean()

    return df


def _team_form_features(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    """
    Features temporais por time SEM leakage.
    Estratégia:
      - construir uma tabela "long" por time e jogo (2 linhas por partida)
      - calcular rolling/expanding por time usando shift(1)
      - depois juntar de volta no df original como features home_* e away_*
    """
    df = df.copy()

    # Tabela long: uma linha para o mandante e outra para o visitante
    home_long = df[["data", "ano_campeonato", "time_mandante", "gf_home", "ga_home", "resultado"]].copy()
    home_long = home_long.rename(columns={
        "time_mandante": "team",
        "gf_home": "gf",
        "ga_home": "ga",
    })
    home_long["is_home"] = 1
    home_long["points"] = np.select(
        [home_long["resultado"].eq("H"), home_long["resultado"].eq("D"), home_long["resultado"].eq("A")],
        [3, 1, 0],
        default=0
    )

    away_long = df[["data", "ano_campeonato", "time_visitante", "gf_away", "ga_away", "resultado"]].copy()
    away_long = away_long.rename(columns={
        "time_visitante": "team",
        "gf_away": "gf",
        "ga_away": "ga",
    })
    away_long["is_home"] = 0
    # pontos do visitante (inverte o resultado)
    away_long["points"] = np.select(
        [away_long["resultado"].eq("A"), away_long["resultado"].eq("D"), away_long["resultado"].eq("H")],
        [3, 1, 0],
        default=0
    )

    long_df = pd.concat([home_long, away_long], ignore_index=True)
    long_df = long_df.sort_values(["team", "data"]).reset_index(drop=True)

    # shift(1) para não usar o jogo atual
    for col in ["gf", "ga", "points", "is_home"]:
        long_df[f"{col}_lag1"] = long_df.groupby("team", observed=True)[col].shift(1)

    # rolling últimos N jogos (usando lag1)
    n = cfg.rolling_n
    mp = cfg.min_periods

    def _roll(s):
        return s.rolling(n, min_periods=mp).mean()

    g = long_df.groupby("team", observed=True)

    long_df["gf_roll"] = g["gf_lag1"].transform(_roll)
    long_df["ga_roll"] = g["ga_lag1"].transform(_roll)
    long_df["pts_roll"] = g["points_lag1"].transform(_roll)

    # forma mandante/visitante: rolling condicionado
    # (aproximação robusta: rolling só nos jogos em casa/fora)
    # Para isso, criamos colunas que só têm valor quando is_home==1/0 e rolling nelas.
    long_df["gf_lag1_home"] = np.where(long_df["is_home"] == 1, long_df["gf_lag1"], np.nan)
    long_df["ga_lag1_home"] = np.where(long_df["is_home"] == 1, long_df["ga_lag1"], np.nan)
    long_df["pts_lag1_home"] = np.where(long_df["is_home"] == 1, long_df["points_lag1"], np.nan)

    long_df["gf_lag1_away"] = np.where(long_df["is_home"] == 0, long_df["gf_lag1"], np.nan)
    long_df["ga_lag1_away"] = np.where(long_df["is_home"] == 0, long_df["ga_lag1"], np.nan)
    long_df["pts_lag1_away"] = np.where(long_df["is_home"] == 0, long_df["points_lag1"], np.nan)

    # Rolling separado (vai ignorar NaN naturalmente, mas pode gerar muitos NaNs no começo)
    long_df["gf_roll_home"] = g["gf_lag1_home"].transform(lambda s: s.rolling(n, min_periods=mp).mean())
    long_df["ga_roll_home"] = g["ga_lag1_home"].transform(lambda s: s.rolling(n, min_periods=mp).mean())
    long_df["pts_roll_home"] = g["pts_lag1_home"].transform(lambda s: s.rolling(n, min_periods=mp).mean())

    long_df["gf_roll_away"] = g["gf_lag1_away"].transform(lambda s: s.rolling(n, min_periods=mp).mean())
    long_df["ga_roll_away"] = g["ga_lag1_away"].transform(lambda s: s.rolling(n, min_periods=mp).mean())
    long_df["pts_roll_away"] = g["pts_lag1_away"].transform(lambda s: s.rolling(n, min_periods=mp).mean())

    # Jogos acumulados (expanding) como medida de experiência/estabilidade
    long_df["games_played_lag1"] = g.cumcount()  # já é "antes do jogo atual" porque cumcount começa em 0
    long_df["pts_expanding"] = g["points_lag1"].transform(lambda s: s.expanding(min_periods=mp).mean())
    long_df["gf_expanding"] = g["gf_lag1"].transform(lambda s: s.expanding(min_periods=mp).mean())
    long_df["ga_expanding"] = g["ga_lag1"].transform(lambda s: s.expanding(min_periods=mp).mean())

    # Agora junta de volta no df original
    # Precisamos de uma chave única por "linha de jogo" no long_df.
    # Como temos duas linhas por jogo, usamos (data, ano, team, is_home) como chave de merge.
    # Para garantir unicidade, usamos o índice do df original criando game_id.
    df["game_id"] = np.arange(len(df))

    home_key = df[["game_id", "data", "ano_campeonato", "time_mandante"]].copy()
    home_key = home_key.rename(columns={"time_mandante": "team"})
    home_key["is_home"] = 1

    away_key = df[["game_id", "data", "ano_campeonato", "time_visitante"]].copy()
    away_key = away_key.rename(columns={"time_visitante": "team"})
    away_key["is_home"] = 0

    # Cria uma coluna "join_key" no long_df para evitar colisões por (team, data) repetido no mesmo dia
    # Vamos fazer merge usando (team, data, is_home) e depois filtrar por game_id via ordem.
    # Como long_df não tem game_id, fazemos merge em duas etapas com cuidado.
    home_feats = home_key.merge(
        long_df,
        on=["team", "data", "is_home", "ano_campeonato"],
        how="left",
        validate="many_to_one"
    )
    away_feats = away_key.merge(
        long_df,
        on=["team", "data", "is_home", "ano_campeonato"],
        how="left",
        validate="many_to_one"
    )

    # Seleciona apenas features e renomeia prefixos
    keep_cols = [
        "game_id",
        "gf_roll", "ga_roll", "pts_roll",
        "gf_roll_home", "ga_roll_home", "pts_roll_home",
        "gf_roll_away", "ga_roll_away", "pts_roll_away",
        "games_played_lag1",
        "pts_expanding", "gf_expanding", "ga_expanding"
    ]

    home_feats = home_feats[keep_cols].add_prefix("home_")
    home_feats = home_feats.rename(columns={"home_game_id": "game_id"})

    away_feats = away_feats[keep_cols].add_prefix("away_")
    away_feats = away_feats.rename(columns={"away_game_id": "game_id"})

    df = df.merge(home_feats, on="game_id", how="left").merge(away_feats, on="game_id", how="left")

    # Limpeza
    df = df.drop(columns=["game_id"])

    return df


def _attack_defense_strength(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    """
    Força de ataque/defesa baseada em médias (rolling) relativas à liga.
    Exemplo:
      attack_home = home_gf_roll / league_gf_home_mean
      defense_home = home_ga_roll / league_gf_away_mean
    """
    df = df.copy()
    eps = cfg.eps

    # Baselines por temporada (já calculadas antes)
    if "league_gf_home_mean" not in df.columns:
        df = _season_league_baselines(df, cfg)

    # ataque/defesa relativos
    df["home_attack_roll"] = df["home_gf_roll"] / (df["league_gf_home_mean"] + eps)
    df["home_defense_roll"] = df["home_ga_roll"] / (df["league_gf_away_mean"] + eps)

    df["away_attack_roll"] = df["away_gf_roll"] / (df["league_gf_away_mean"] + eps)
    df["away_defense_roll"] = df["away_ga_roll"] / (df["league_gf_home_mean"] + eps)

    # “gap” entre times (features fortes)
    df["attack_gap"] = df["home_attack_roll"] - df["away_attack_roll"]
    df["defense_gap"] = df["away_defense_roll"] - df["home_defense_roll"]  # menor melhor p/ defesa

    # Forma em pontos (gap)
    df["pts_gap_roll"] = df["home_pts_roll"] - df["away_pts_roll"]

    return df


def build_features(df: pd.DataFrame, cfg: Optional[FeatureConfig] = None) -> pd.DataFrame:
    """
    Pipeline principal:
      - normaliza datas e ordena
      - cria targets/auxiliares
      - baselines de liga e mando
      - features por time (rolling/expanding sem leakage)
      - força ataque/defesa
    Retorna dataframe com features + colunas originais.
    """
    cfg = cfg or FeatureConfig()

    df = _ensure_datetime(df, "data")
    df = df.sort_values(["data", "ano_campeonato"]).reset_index(drop=True)

    df = _basic_targets(df)

    if cfg.use_season_baseline:
        df = _season_league_baselines(df, cfg)

    df = _home_advantage(df, cfg)
    df = _team_form_features(df, cfg)
    df = _attack_defense_strength(df, cfg)

    return df


def make_model_matrix(
    df_feat: pd.DataFrame,
    drop_na: bool = True
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separa X e y.
    Mantém somente features numéricas e algumas de baseline.
    """
    df = df_feat.copy()

    y = df["resultado"].astype(str)

    # removemos colunas não numéricas / identificadores
    non_features = {
        "data", "estadio", "arbitro",
        "time_mandante", "time_visitante",
        "tecnico_mandante", "tecnico_visitante",
        "resultado"
    }
    cols = [c for c in df.columns if c not in non_features]

    # mantém só numéricas (features)
    X = df[cols].select_dtypes(include=[np.number]).copy()

    if drop_na:
        mask = X.notna().all(axis=1)
        X = X.loc[mask].reset_index(drop=True)
        y = y.loc[mask].reset_index(drop=True)

    return X, y


def split_by_year(
    df_feat: pd.DataFrame,
    train_end: int = 2023,
    test_year: int = 2024,
    context_year: int = 2025
) -> dict:
    """
    Retorna dict com:
      - train: <= train_end
      - test: == test_year
      - context: == context_year (para estado atual / calibração / simulação)
    """
    out = {}
    out["train"] = df_feat[df_feat["ano_campeonato"] <= train_end].copy()
    out["test"] = df_feat[df_feat["ano_campeonato"] == test_year].copy()
    out["context"] = df_feat[df_feat["ano_campeonato"] == context_year].copy()
    return out
