def isleapyear(year):
    """
    判断是否为闰年
    """
    if (year % 100 != 0) & (year % 4 == 0):
        return True
    elif year % 400 == 0:
        return True
    else:
        return False


def daynuminmon(year, month):
    """
    返回每月天数
    """
    if month == 2:
        if isleapyear(year):
            return 29
        else:
            return 28
    elif (month == 1) or (month == 3) or (month == 5) or (month == 7) or (month == 8) or (
            month == 10) or (month == 12):
        return 31
    elif (month == 4) or (month == 6) or (month == 9) or (month == 11):
        return 30


def schedule(year, days):
    """
    返回某年第某天的日期
    """
    count_days = days
    for month in range(1, 13):
        day_num = daynuminmon(year, month)
        count_days = count_days - day_num
        if count_days <= 0:
            day = count_days + day_num
            date = year * 10000 + month * 100 + day
            return date
    raise Exception("The input parameter 'days' exceeds the maximum number of days of the year!")


def uptimereport(startTime, nowTime):
    up_time = nowTime - startTime
    run_seconds = up_time % 60
    run_minutes = (up_time // 60) % 60
    run_hours = (up_time // (60 * 60)) % 24
    run_days = (up_time // (60 * 60 * 24))
    report_time = "%d days %d h %d min %d s" % (run_days, run_hours, run_minutes, run_seconds)
    return report_time
