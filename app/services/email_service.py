import smtplib
import os
import asyncio
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# Load environment configuration
SENDER_EMAIL = os.getenv("SENDER_EMAIL", "kumaribeere2005@gmail.com")
APP_PASSWORD = os.getenv("APP_PASSWORD", "barmvlntnyzrkeam")

# Predefined premium HTML templates
def get_html_template(candidate_name: str, title: str, description: str, info_rows: list, cta_text: str = None, cta_url: str = None) -> str:
    rows_html = ""
    for label, val in info_rows:
        rows_html += f"""
        <div class="info-row">
            <span class="info-label">{label}</span>
            <span class="info-value">{val}</span>
        </div>
        """

    cta_html = ""
    if cta_text and cta_url:
        cta_html = f"""
        <div class="button-container">
            <a href="{cta_url}" class="button" target="_blank">{cta_text}</a>
        </div>
        """

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background-color: #0b0f19;
    color: #f3f4f6;
    margin: 0;
    padding: 0;
  }}
  .container {{
    max-width: 600px;
    margin: 40px auto;
    background-color: #111827;
    border: 1px solid #1f2937;
    border-radius: 16px;
    overflow: hidden;
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
  }}
  .header {{
    background: linear-gradient(135deg, #6366f1 0%, #312e81 100%);
    padding: 32px;
    text-align: center;
  }}
  .logo {{
    font-size: 24px;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -0.025em;
  }}
  .content {{
    padding: 32px;
  }}
  .title {{
    font-size: 20px;
    font-weight: 700;
    color: #ffffff;
    margin-top: 0;
    margin-bottom: 16px;
  }}
  .message {{
    font-size: 14px;
    line-height: 1.6;
    color: #d1d5db;
    margin-bottom: 24px;
  }}
  .info-card {{
    background-color: #1f2937;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 24px;
    border-left: 4px solid #6366f1;
  }}
  .info-row {{
    display: flex;
    justify-content: space-between;
    margin-bottom: 8px;
    font-size: 13px;
  }}
  .info-row:last-child {{
    margin-bottom: 0;
  }}
  .info-label {{
    color: #9ca3af;
    font-weight: 500;
  }}
  .info-value {{
    color: #ffffff;
    font-weight: 600;
  }}
  .button-container {{
    text-align: center;
    margin-top: 32px;
    margin-bottom: 16px;
  }}
  .button {{
    display: inline-block;
    background-color: #6366f1;
    color: #ffffff !important;
    text-decoration: none;
    font-weight: 600;
    font-size: 14px;
    padding: 12px 24px;
    border-radius: 8px;
  }}
  .footer {{
    padding: 24px 32px;
    background-color: #0b0f19;
    border-top: 1px solid #1f2937;
    text-align: center;
    font-size: 12px;
    color: #6b7280;
  }}
</style>
</head>
<body>
  <div class="container">
    <div class="header">
      <div class="logo">Kaira Assessments</div>
    </div>
    <div class="content">
      <div class="title">{title}</div>
      <p class="message">Hi {candidate_name},</p>
      <p class="message">{description}</p>
      <div class="info-card">
        {rows_html}
      </div>
      {cta_html}
    </div>
    <div class="footer">
      This is an automated message from Kaira. Please do not reply directly to this email.
    </div>
  </div>
</body>
</html>
"""

def _sync_send_email(to_email: str, subject: str, html_content: str):
    """Synchronous send handler executed in separate thread to prevent blocking event loop."""
    if not APP_PASSWORD:
        raise ValueError("APP_PASSWORD is not set in environment.")

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = f"Kaira <{SENDER_EMAIL}>"
    msg["To"] = to_email
    msg.attach(MIMEText(html_content, "html"))

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(SENDER_EMAIL, APP_PASSWORD)
        server.sendmail(SENDER_EMAIL, to_email, msg.as_string())

async def send_email(to_email: str, subject: str, html_content: str):
    """Asynchronous wrapper for email sending."""
    await asyncio.to_thread(_sync_send_email, to_email, subject, html_content)

async def notify_stage_scheduled(to_email: str, candidate_name: str, stage_name: str, details: dict):
    subject = f"Scheduled: {stage_name} Assessment — Kaira"
    title = f"Your {stage_name} assessment is scheduled"
    desc = f"A new stage has been scheduled for your assessment cycle. You can complete this from your candidate dashboard."
    
    info_rows = [
        ("Assessment Type", stage_name),
        ("Complexity", details.get("complexity", "Medium")),
        ("Questions count", str(details.get("num_questions", 5))),
        ("Time Limit", f"{details.get('time_limit', 45)} mins" if "time_limit" in details else "N/A"),
    ]
    if details.get("scheduled_time"):
        info_rows.append(("Start Time", details["scheduled_time"]))
    if details.get("deadline"):
        info_rows.append(("Deadline", details["deadline"]))

    html = get_html_template(
        candidate_name=candidate_name,
        title=title,
        description=desc,
        info_rows=info_rows,
        cta_text="Start Assessment",
        cta_url="http://localhost:8085/dashboard"
    )
    await send_email(to_email, subject, html)

async def notify_stage_completed(to_email: str, candidate_name: str, stage_name: str, attempt_num: int, score: float = None):
    subject = f"Completed: {stage_name} Assessment — Kaira"
    title = f"Successfully completed: {stage_name}"
    desc = f"Congratulations on completing your {stage_name} round. Your answers have been analyzed by Kaira AI and are now under recruiter review."
    
    info_rows = [
        ("Assessment Type", stage_name),
        ("Attempt Number", f"#{attempt_num}"),
    ]
    if score is not None:
        info_rows.append(("Final Score", f"{score}/10"))
    
    html = get_html_template(
        candidate_name=candidate_name,
        title=title,
        description=desc,
        info_rows=info_rows,
        cta_text="View Dashboard",
        cta_url="http://localhost:8085/dashboard"
    )
    await send_email(to_email, subject, html)

async def notify_stage_decision(to_email: str, candidate_name: str, stage_name: str, decision: str, notes: str = ""):
    subject = f"Review update: {stage_name} — Kaira"
    
    if decision == "shortlist":
        title = f"Congratulations! You've been shortlisted for the next stage."
        desc = f"The recruitment team has completed review of your {stage_name} submission and shortlisted you to move forward."
    elif decision == "reject":
        title = f"Update regarding your {stage_name} submission."
        desc = f"Thank you for completing your {stage_name} submission. Unfortunately, the team has decided not to move forward with your application at this time."
    else:
        title = f"Analysis completed: {stage_name}"
        desc = f"Your {stage_name} submission review is complete and the assessment is now under recruiter review."

    info_rows = [
        ("Assessment Type", stage_name),
        ("Status", "Shortlisted" if decision == "shortlist" else "Rejected" if decision == "reject" else "Reviewed"),
    ]
    if notes:
        info_rows.append(("Feedback", notes))

    html = get_html_template(
        candidate_name=candidate_name,
        title=title,
        description=desc,
        info_rows=info_rows,
        cta_text="View Updates",
        cta_url="http://localhost:8085/dashboard"
    )
    await send_email(to_email, subject, html)

async def notify_hiring_decision(to_email: str, candidate_name: str, decision: str, details: dict):
    if decision == "offer_sent":
        subject = f"Congratulations! Offer Letter from {details.get('company_name', 'Kaira Partner')}"
        title = "Congratulations! You have received an Offer Letter"
        desc = f"The recruitment team at {details.get('company_name', 'Kaira Partner')} is thrilled to offer you the role of {details.get('role', 'Engineer')}. Click the link below to view and download your formal offer details."
        info_rows = [
            ("Company", details.get("company_name")),
            ("Offered Role", details.get("role")),
            ("Decision", "Offer Issued"),
        ]
        cta_text = "View Offer Letter"
        cta_url = "http://localhost:8085/report"
    else:
        subject = "Update on your application status"
        title = "Application Update"
        desc = "Thank you for participating in our assessments and interview cycle. The hiring panel has concluded evaluations, and unfortunately, we are not moving forward with an offer at this stage."
        info_rows = [
            ("Hiring Panel", details.get("company_name", "Kaira")),
            ("Status", "Process Concluded"),
        ]
        cta_text = "Go to Platform"
        cta_url = "http://localhost:8085/"

    html = get_html_template(
        candidate_name=candidate_name,
        title=title,
        description=desc,
        info_rows=info_rows,
        cta_text=cta_text,
        cta_url=cta_url
    )
    await send_email(to_email, subject, html)
