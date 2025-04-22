import json
import streamlit as st

DEFAULT_LAYOUT = {
    "metrics": {
        "enabled": True,
        "order": ["portfolio_value", "sharpe_ratio", "alpha", "beta", "var", "volatility", "drawdown"],
        "columns": 4
    },
    "charts": {
        "enabled": True,
        "order": ["cumulative_returns", "portfolio_composition", "drawdowns", "monthly_returns"],
        "columns": 2
    },
    "tables": {
        "enabled": True,
        "order": ["holdings", "performance"]
    }
}

class DashboardConfig:
    @staticmethod
    def load_config():
        """Load dashboard configuration from session state or defaults"""
        if 'dashboard_config' not in st.session_state:
            st.session_state.dashboard_config = DEFAULT_LAYOUT
        return st.session_state.dashboard_config
    
    @staticmethod
    def save_config(config):
        """Save dashboard configuration to session state"""
        st.session_state.dashboard_config = config
    
    @staticmethod
    def export_config():
        """Export configuration as JSON"""
        return json.dumps(st.session_state.dashboard_config, indent=2)
    
    @staticmethod
    def import_config(json_str):
        """Import configuration from JSON"""
        try:
            config = json.loads(json_str)
            st.session_state.dashboard_config = config
            return True
        except:
            return False
