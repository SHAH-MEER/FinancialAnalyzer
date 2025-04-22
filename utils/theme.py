"""Theme configuration for the application."""

__all__ = ['THEME']

THEME = {
    'light': {
        'primary': '#2563EB',
        'secondary': '#6B7280',
        'success': '#059669',
        'danger': '#DC2626',
        'warning': '#D97706',
        'info': '#2563EB',
        'background': '#F8FAFC',
        'surface': '#FFFFFF',
        'border': '#E5E7EB',
        'text': {
            'primary': '#111827',
            'secondary': '#4B5563',
            'muted': '#9CA3AF'
        },
        'chart': {
            'background': '#FFFFFF',
            'grid': '#F3F4F6',
            'text': '#111827'
        }
    },
    'dark': {
        'primary': '#3B82F6',
        'secondary': '#9CA3AF',
        'success': '#10B981',
        'danger': '#EF4444',
        'warning': '#F59E0B',
        'info': '#3B82F6',
        'background': 'transparent',  # Changed to transparent
        'surface': '#262730',  # Changed to match Streamlit's dark mode surface color
        'border': '#374151',
        'text': {
            'primary': '#F9FAFB',
            'secondary': '#D1D5DB',
            'muted': '#6B7280'
        },
        'chart': {
            'background': '#262730',  # Changed to match surface color
            'grid': '#374151',
            'text': '#F9FAFB'
        }
    }
}