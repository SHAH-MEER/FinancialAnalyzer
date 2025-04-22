import pandas as pd
import plotly.io as pio
from fpdf import FPDF
import streamlit as st
from datetime import datetime

class PortfolioReport:
    def __init__(self, analyzer):
        self.analyzer = analyzer
        
    def generate_pdf(self):
        """Generate PDF report with portfolio analysis"""
        pdf = FPDF()
        pdf.add_page()
        
        # Add header
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'Portfolio Analysis Report', 0, 1, 'C')
        pdf.ln(10)
        
        # Add summary
        summary, _ = self.analyzer.generate_summary_report()
        
        pdf.set_font('Arial', '', 12)
        for key, value in summary.items():
            pdf.cell(0, 10, f'{key}: {value}', 0, 1)
            
        # Add plots as images
        plots = {
            'Returns': self.analyzer.plot_cumulative_returns(),
            'Composition': self.analyzer.plot_portfolio_composition(),
            'Risk': self.analyzer.plot_drawdowns()
        }
        
        for name, fig in plots.items():
            img_path = f'temp_{name}.png'
            pio.write_image(fig, img_path)
            pdf.image(img_path, x=10, w=190)
            
        return pdf.output(dest='S').encode('latin1')