from typing import List, Union
from alpaca.trading.models import Calendar
from datetime import date, datetime

def filter_trading_day(days: List[Calendar], trading_day: date) -> Union[Calendar, None]:
    filt = filter(lambda x: x.date == trading_day, days)

    list_of_days = list(filt)


    if list_of_days:
        return list_of_days[0]
    else:
        return None

def check_if_market_open(days: List[Calendar], trading_day: date = date.today()) -> bool:
    """Checks if the market is open

    Args:
        days (List[Calendar]): _description_
        trading_day (date, optional): _description_. Defaults to date.today().

    Returns:
        bool: _description_
    """

    calendar_day = filter_trading_day(
        days = days,
        trading_day=trading_day
    )

    if calendar_day:
        now = datetime.now()

        if now < calendar_day.close and now > calendar_day.open:
            return True
        else:
            return False

    else:
        return False

def get_market_time(days: List[Calendar], trading_day: date = date.today(), market_time: str = 'close') -> Union[datetime, None]:
    assert market_time in ['close', 'open'], "market_time has to be either close or open"
    trading_day_calendar = filter_trading_day(days = days, trading_day=trading_day)

    if trading_day_calendar:
        return getattr(trading_day_calendar, market_time)
    else:
        return None

