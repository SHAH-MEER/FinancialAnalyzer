def run_scenario(portfolio, scenario_type='market_crash'):
    scenarios = {
        'market_crash': -0.4,
        'recession': -0.2,
        'boom': 0.3
    }
    return portfolio.calculate_portfolio_value() * (1 + scenarios[scenario_type])
