# -*- coding: utf-8 -*-
"""
team_aliases.py

Canonicalização de nomes de clubes para alinhar múltiplas fontes (CBF, football-data, etc.)
"""

TEAM_ALIASES = {
    # SUDESTE — RJ
    "flamengo rj": "flamengo",
    "botafogo rj": "botafogo",

    # SUDESTE — SP
    "santos fc": "santos",
    "sao paulo": "sao paulo",
    "palmeiras": "palmeiras",
    "corinthians": "corinthians",

    # SUDESTE — MG
    "atletico mg": "atletico mineiro",

    # SUL — PR
    "atletico pr": "athletico paranaense",
    "athletico pr": "athletico paranaense",
    "coritiba fc": "coritiba",

    # SUL — SC
    "avai fc": "avai",
    "figueirense fc": "figueirense",
    "criciuma ec": "criciuma",
    "chapecoense sc": "chapecoense",

    # SUL — RS
    "gremio": "gremio",
    "internacional": "internacional",

    # CENTRO-OESTE
    "cuiaba mt": "cuiaba",

    # NORDESTE
    "ec bahia": "bahia",
    "ec vitoria": "vitoria",
    "ceara sc": "ceara",
    "sport recife": "sport",
    "nautico": "nautico",

    # AJUSTES “NOMES LONGOS” (CBF → canônico)
    "rb bragantino": "bragantino",
    "vasco da gama": "vasco",
    "joinville sc": "joinville",
    "goias ec": "goias",
}

def canon_team(name: str) -> str:
    """Normaliza texto e aplica aliases canônicos."""
    if name is None:
        return name

    n = str(name).lower().strip()
    n = n.replace("-", " ")

    # remover acentos (sem depender de lib externa)
    n = n.replace("á","a").replace("ã","a").replace("â","a")
    n = n.replace("é","e").replace("ê","e")
    n = n.replace("í","i")
    n = n.replace("ó","o").replace("ô","o")
    n = n.replace("ú","u")
    n = n.replace("ç","c")

    # normalizar espaços
    n = " ".join(n.split())

    return TEAM_ALIASES.get(n, n)
