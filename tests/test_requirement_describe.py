from real_estate_agent.models import CustomerRequirement


def test_budget_and_area_without_one_side_have_no_stray_dash():
    """'3억 이하' 가 '예산=-~3억원' 으로 보이던 것 -> '예산=~3억원'."""
    d = CustomerRequirement(budget_max=30_000, area_min_m2=661.2).describe()
    assert "예산=~3억원" in d and "-~" not in d and "~-" not in d
    assert "면적=200평~" in d
