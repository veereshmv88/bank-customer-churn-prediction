"""
PDF Report Generator using ReportLab.
Compiles research report and executive summary into a formal academic PDF.
"""
import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable

def generate_pdf():
    pdf_path = "reports/research_report.pdf"
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        rightMargin=45,
        leftMargin=45,
        topMargin=45,
        bottomMargin=45
    )

    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#0B1220"),
        spaceAfter=10
    )

    h1_style = ParagraphStyle(
        "Heading1_Custom",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#1E3A8A"),
        spaceBefore=12,
        spaceAfter=6
    )

    body_style = ParagraphStyle(
        "Body_Custom",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#334155"),
        spaceAfter=8
    )

    elements = []

    # Title & Metadata
    elements.append(Paragraph("Predictive Modeling & Risk Scoring for Bank Customer Churn", title_style))
    elements.append(Paragraph("<b>Academic Research & Executive Technical Report</b> | Enterprise AI Platform v1.0", body_style))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#3B82F6"), spaceAfter=14))

    # Executive Abstract
    elements.append(Paragraph("1. Executive Abstract", h1_style))
    elements.append(Paragraph(
        "This project implements an end-to-end full-stack machine learning architecture for predicting bank customer churn and computing operational risk scores. "
        "Using 10,000 retail banking records across France, Germany, and Spain, five models were benchmarked through 5-fold stratified cross-validation. "
        "The winning Gradient Boosting Pipeline achieves an ROC-AUC of 0.8690, PR-AUC of 0.7184, Accuracy of 86.85%, and Brier Score of 0.0994 on an untouched test set of 2,000 customers. "
        "At the calibrated decision threshold of 0.28, minority churn recall reaches 67.32% with an F1-score of 64.02%.",
        body_style
    ))

    # Methodology & Hygiene
    elements.append(Paragraph("2. Data Hygiene & Feature Engineering", h1_style))
    elements.append(Paragraph(
        "Customer identifiers (CustomerId, Surname, RowNumber) were quarantined and excluded from predictive features to prevent target leakage. "
        "Domain-engineered features include BalanceSalaryRatio (deposit-to-income balance), EngagementProductInteraction (active membership × product count), "
        "and ProductDensityIndicator (NumOfProducts / (Tenure + 1)).",
        body_style
    ))

    # Model Performance Table
    elements.append(Paragraph("3. Cross-Validation Model Comparison", h1_style))
    data = [
        ["Model Architecture", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "PR-AUC"],
        ["Gradient Boosting (Selected)", "86.29%", "75.12%", "48.22%", "0.5869", "0.8607", "0.6924"],
        ["XGBoost", "80.44%", "51.13%", "73.37%", "0.6025", "0.8602", "0.6938"],
        ["Random Forest", "83.36%", "56.95%", "67.06%", "0.6158", "0.8577", "0.6835"],
        ["Logistic Regression", "75.96%", "43.57%", "77.67%", "0.5583", "0.8308", "0.6025"],
        ["Decision Tree", "75.46%", "43.02%", "79.32%", "0.5583", "0.8231", "0.6073"]
    ]
    t = Table(data, colWidths=[160, 55, 55, 55, 55, 60, 60])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1E293B")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor("#EFF6FF")),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 10))

    # Risk Scoring Framework
    elements.append(Paragraph("4. Operational Risk Scoring Framework", h1_style))
    elements.append(Paragraph(
        "Risk scores are calculated as: <b>Risk Score = Churn Probability × 100</b>.<br/>"
        "• <b>Low Risk (0 - 24.9%):</b> Standard engagement. (69.8% of portfolio)<br/>"
        "• <b>Medium Risk (25.0 - 49.9%):</b> Proactive relationship management review. (18.1% of portfolio)<br/>"
        "• <b>High Risk (&ge; 50.0%):</b> Executive intervention and customized retention terms. (12.1% of portfolio)",
        body_style
    ))

    # Explainability & Partial Dependence
    elements.append(Paragraph("5. Model Explainability & Partial Dependence (PDP)", h1_style))
    elements.append(Paragraph(
        "Global feature attribution is governed by TreeSHAP and Partial Dependence Plots (PDP):<br/>"
        "• <b>Age:</b> Pronounced non-linear escalation between ages 40 and 60, where marginal churn probability climbs from ~10.5% to >45%.<br/>"
        "• <b>NumOfProducts:</b> Non-linear U-curve; 2 products yield optimal customer retention (~11.3% risk), whereas 3 or 4 products surge to ~76.0% and ~86.6%.<br/>"
        "• <b>IsActiveMember:</b> Active engagement provides an average marginal risk reduction of ~10.3 percentage points (~16.1% vs ~26.4%).<br/>"
        "• <b>Balance:</b> Modest positive gradient, rising from ~20.6% for zero balance to ~25%–28% for deposits exceeding €100,000.",
        body_style
    ))

    # Responsible Use
    elements.append(Paragraph("6. Responsible Use & Governance", h1_style))
    elements.append(Paragraph(
        "Predictions represent statistical estimates derived from historical patterns. "
        "The model is designated as a decision-support instrument and must not be used to automatically restrict credit, deny banking services, or enforce punitive pricing without human review.",
        body_style
    ))

    doc.build(elements)
    print(f"PDF successfully generated at: {pdf_path}")

if __name__ == "__main__":
    generate_pdf()
