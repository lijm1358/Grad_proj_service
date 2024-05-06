import datetime


def get_current_date_str() -> str:
    now = datetime.datetime.now()
    return now.strftime("%Y%m%d")
