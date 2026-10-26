"""
Automated Compliance Tests for Streamlit Interface and Partial Dependence Plots (PDP).
"""
import os
import joblib
import pandas as pd
from streamlit.testing.v1 import AppTest

def test_pdp_figures_exist():
    """Verify PDP figures exist and have non-zero file sizes."""
    pdp_path = os.path.join("reports", "figures", "pdp_analysis.png")
    pdp_alt = os.path.join("reports", "figures", "partial_dependence_plots.png")
    assert os.path.exists(pdp_path), f"Missing {pdp_path}"
    assert os.path.exists(pdp_alt), f"Missing {pdp_alt}"
    assert os.path.getsize(pdp_path) > 10000, "PDP figure file size is too small"

def test_research_report_pdp_content():
    """Verify research report and PDF document PDP analysis."""
    report_path = os.path.join("reports", "research_report.md")
    assert os.path.exists(report_path), "Missing research_report.md"
    with open(report_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert "Partial Dependence" in content
    assert "NumOfProducts" in content
    assert "Age" in content

    pdf_path = os.path.join("reports", "research_report.pdf")
    assert os.path.exists(pdf_path), "Missing research_report.pdf"
    assert os.path.getsize(pdf_path) > 1000, "PDF file is empty or corrupted"

def test_streamlit_app_execution_and_prediction():
    """Verify Streamlit app initializes and processes predictions without exceptions."""
    at = AppTest.from_file("streamlit_app.py").run(timeout=10)
    assert len(at.exception) == 0, f"Streamlit initialization error: {at.exception}"

    # Click calculate churn risk button
    at.button[0].click().run(timeout=10)
    assert len(at.exception) == 0, f"Streamlit risk calculation error: {at.exception}"

    # Click what-if button
    at.button[1].click().run(timeout=10)
    assert len(at.exception) == 0, f"Streamlit what-if simulation error: {at.exception}"
