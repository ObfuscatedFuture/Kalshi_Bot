from datetime import datetime, timedelta

def get_trends_window(report_date_str, days=30):
    """
    Returns (start_date, end_date) for Google Trends window.
    Example: (30 days before earnings, day before earnings).
    """
    report_date = datetime.strptime(report_date_str, "%Y-%m-%d")
    start = report_date - timedelta(days=days)
    end = report_date - timedelta(days=2)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
