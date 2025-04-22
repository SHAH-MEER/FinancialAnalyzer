CHART_THEME = {
    'light': {
        'bg_color': '#FFFFFF',
        'grid_color': '#F3F4F6',
        'text_color': '#1F2937',
        'axis_color': '#9CA3AF',
        'positive_color': '#10B981',
        'negative_color': '#EF4444',
        'neutral_color': '#3B82F6'
    },
    'dark': {
        'bg_color': '#1F2937',
        'grid_color': '#374151',
        'text_color': '#F9FAFB',
        'axis_color': '#9CA3AF',
        'positive_color': '#34D399',
        'negative_color': '#F87171',
        'neutral_color': '#60A5FA'
    }
}

def get_chart_theme(mode='light'):
    """Get Plotly chart theme configuration"""
    theme = CHART_THEME[mode]
    return {
        'layout': {
            'paper_bgcolor': theme['bg_color'],
            'plot_bgcolor': theme['bg_color'],
            'font': {'color': theme['text_color']},
            'xaxis': {
                'gridcolor': theme['grid_color'],
                'linecolor': theme['axis_color']
            },
            'yaxis': {
                'gridcolor': theme['grid_color'],
                'linecolor': theme['axis_color']
            }
        },
        'colors': {
            'positive': theme['positive_color'],
            'negative': theme['negative_color'],
            'neutral': theme['neutral_color']
        }
    }