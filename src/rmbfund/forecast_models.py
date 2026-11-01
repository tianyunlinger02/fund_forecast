"""Public catalog of implemented forecast methods, not algorithm output versions."""
from __future__ import annotations

NAV_MODEL_ID = "nav_ridge_empirical_v2"
INVESTMENT_MODEL_ID = "exposure_factor_ridge_empirical_v1"
NAV_MODEL = {
    "id": NAV_MODEL_ID, "name": "净值趋势预测",
    "description": "只使用本基金已公布净值的涨跌、波动与回撤；资料少时采用历史经验分布。",
    "supported_kinds": ("open_fund",), "horizons": (1, 5, 20),
}
INVESTMENT_MODEL = {
    "id": INVESTMENT_MODEL_ID, "name": "投资因素预测（试验）",
    "description": (
        "同时使用净值、新兴市场与半导体价格、人民币中间价和美债利率，"
        "估计之后1、5、20笔净值。当前映射仅核对018147，不代表更准，"
        "也没有把经理任期或当前持仓倒填为历史特征。"
    ),
    "supported_kinds": ("open_fund",), "horizons": (1, 5, 20),
}


def model_family(model_id: str | None) -> str | None:
    return "nav_ridge_empirical" if model_id in {"nav_ridge_empirical_v1", NAV_MODEL_ID} else model_id


def model_name(model_id: str | None) -> str | None:
    if model_id == "nav_ridge_empirical_v1":
        return "净值趋势预测（输出v1）"
    return next((row["name"] for row in (NAV_MODEL, INVESTMENT_MODEL) if row["id"] == model_id), None)


def model_catalog(kind: str | None = None, *, symbol: str | None = None) -> list[dict]:
    """Presentation metadata only; actual factor history/profile readiness is separate."""
    return [{
        **row, "supported_kinds": list(row["supported_kinds"]), "horizons": list(row["horizons"]),
        "available": (kind is None or kind in row["supported_kinds"])
                     and (row["id"] == NAV_MODEL_ID or symbol == "018147"),
        "unavailable_reason": ("这只基金还没有核对投资因素映射。"
                               if row["id"] == INVESTMENT_MODEL_ID and symbol != "018147" else None),
    } for row in (NAV_MODEL, INVESTMENT_MODEL)]
