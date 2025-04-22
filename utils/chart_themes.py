CHART_THEMES = {
    'light': {
        'background': '#FFFFFF',
        'paper_bgcolor': '#FFFFFF',
        'plot_bgcolor': '#F8FAFC',
        'grid_color': '#E2E8F0',
        'text_color': '#1F2937',
        'axis_color': '#64748B',
        'colors': {
            'primary': '#3B82F6',
            'success': '#10B981',
            'danger': '#EF4444',
            'neutral': '#6B7280',
            'accent': '#8B5CF6'
        }
    },
    'dark': {
        'background': '#1F2937',
        'paper_bgcolor': '#1F2937',
        'plot_bgcolor': '#111827',
        'grid_color': '#374151',
        'text_color': '#F9FAFB',
        'axis_color': '#9CA3AF',
        'colors': {
            'primary': '#60A5FA',
            'success': '#34D399',
            'danger': '#F87171',
            'neutral': '#9CA3AF',
            'accent': '#A78BFA'
        }
    }
}

def get_chart_layout(theme='light', title=None, height=None):
    """Get base chart layout with theme"""
    theme_config = CHART_THEMES[theme]
    
    layout = {
        'template': 'plotly_dark' if theme == 'dark' else 'plotly_white',
        'paper_bgcolor': theme_config['paper_bgcolor'],
        'plot_bgcolor': theme_config['plot_bgcolor'],
        'font': {'color': theme_config['text_color']},
        'xaxis': {
            'gridcolor': theme_config['grid_color'],
            'linecolor': theme_config['axis_color'],
            'showgrid': True
        },
        'yaxis': {
            'gridcolor': theme_config['grid_color'],
            'linecolor': theme_config['axis_color'],
            'showgrid': True
        }
    }
    
    if title:
        layout['title'] = {
            'text': title,
            'font': {'size': 20}
        }
    
    if height:
        layout['height'] = height
        
    return layout