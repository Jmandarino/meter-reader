# python 3.3 or newer
from datetime import datetime
from datetime import timedelta
import time
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

# TODO: python-dateutil for parsing and handling timezones
# TODO: fix timezone issues
# TODO: meter reading automation (open CV or ML)
# TODO: automate the creation of axvlines
# TODO: formalize/standardize comment style
# TODO: unify dates and meter reading
# TODO: should target values come first or last in method calls

# stores processed dates datetime objects
dates = []
# stores lines verticle lines for graph
axvlines_raw = []
# stores processed lines (times converted to epoch time)
axvlines = []
# stores times
t = [
    "2/5/18 0:00",
    "2/7/18 18:58",
    "2/7/18 23:16",
    "2/8/18 0:00",
    "2/8/18 8:05",
    "2/8/18 8:52",
    "2/8/18 18:30",
    "2/8/18 20:35",
    "2/8/18 23:17",
    "2/9/18 8:36",
    "2/9/18 16:35",
    "2/9/18 18:31",
    "2/9/18 21:12",
    "2/9/18 23:12",
    "2/10/18 7:13",
    "2/10/18 18:23",
    "2/10/18 19:33",
    "2/10/18 20:37",
    "2/10/18 22:01",
    "2/10/18 22:36",
    "2/11/18 9:45",
    "2/11/18 12:06",
    "2/11/18 18:29",
    "2/12/18 8:37",
    "2/12/18 18:05",
    "2/12/18 19:19"

]
# stores energy meter readings (kWh)
e = [
    52864,
    52900,
    52901,
    52901,
    52903,
    52903,
    52903,
    52906,
    52907,
    52909,
    52911,
    52912,
    52914,
    52917,
    52919,
    52922,
    52922,
    52925,
    52925,
    52926,
    52930,
    52935,
    52939,
    52944,
    52950,
    52952

]

"""
HELPER FUNCTIONS
"""


def _calc_slope(x,y,x1,y1):
    return (y1 - y)/(x1 - x)


def _calc_y_incpt(slope, x, y):
    return y - (slope * x)


def _calc_y_value(slope, x, b):
    return slope * x + b


def _is_between(date_tagert, date_before, date_after):
    return date_before < date_tagert and date_after > date_tagert




"""
FUNCTIONS
"""
# TODO: fix the format to be more readable


def calc_est_y(x_poi, x_before, y_before, x_after, y_after):
    """
    Calculate an estimate value for y for our x value Point of Interest

    1. find equation for line, y = mx +b
    2. plug in x to find y

    :param x_poi: x value we want to find y for
    :param x_before: x value before x_poi
    :param y_before: y value before x_poi
    :param x_after: x value after x_poi
    :param y_after: y value after _poi
    :return: Estimate of y value at that time
    """
    m = _calc_slope(x_before, y_before, x_after, y_after)
    b = _calc_y_incpt(m,x_before,y_before)
    return _calc_y_value(m, x_poi, b)


def date_str_to_epoch(str):
    """
    Converts from d/m/yy hh:mm to epoch time
    
    :param str: inputed time as a string
    :return: time in epoch time
    """
    # 2/5/18 00:00"
    d = datetime.strptime(str, "%m/%d/%y %H:%M")
    unixtime = time.mktime(d.timetuple())
    return unixtime


# TODO: in-> epoch out->epoch or datetime->dateime
def init_axvlines(min, max):
    """
    Creates an array of date markers in epoch time  from a start and end date
    :param min: start date (in epoch time)
    :param max: end date (in epoch time)
    :return: list() of timestamps
    """
    if min == None or min == max:
        return []

    output = []
    # get the midnight value (12:01/0:01) for the start time
    # indicate the the start of the first day
    cur = datetime.fromtimestamp(min)
    cur = cur.replace(hour=0, minute=0, second=0, microsecond=0)
    output.append(cur.timestamp())

    #TODO: only work on datetime objects and
    while(cur.timestamp() < max):
        cur = cur + timedelta(days=1)
        output.append(cur.timestamp())

    return output


def time_before_after(dates, target):
    """
    :param dates: list of datetime objects
    :param target: datetime object
    :return: datetime before and datetime after
    """
    # take care of edge case
    if dates == [] or target == None:
        return None, None

    # if target is either smaller than or greater than date range
    if dates[0] >= target:
        return None, dates[0]
    elif dates[-1] <= target:
        return dates[-1], None

    # maybe binary search?
    start = 0
    while start < len(dates) :

        if _is_between(target, dates[start], dates[start+1]):
            return dates[start], dates[start + 1]
        elif dates[start] ==  target:
            return dates[start], dates[start]
        else:
            start += 1


def usage_estimates(dates, samples_dates, meter_values):
    """
    :param dates: list(datetime obj) dates we are interested, should be axvlines
    :param samples_dates: list(datetime obj) dates that corrispond to meter_values
    :param meter_values: list(int) meter values for given dates
    :return: {datetime : (int)}
    """
    date_pairs = []
    # for a given date, find midnight on both sides and use that to estimate daily usage
    # find the closest date
    data = dict(zip(samples_dates, meter_values))
    # can this be done more efficiently with binary search or by limiting size of array
    for date in dates:
        before, after = time_before_after(samples_dates, date)

        # if we don't have complete data don't process
        if before is None or after is None:
            continue

        # if the time value is a value in our samples_dates
        if before == after:
            date_pairs.append((before, data[before]))
        else:
            y_before = data[before]
            y_after = data[after]

            y = calc_est_y(date.timestamp(), before.timestamp(), y_before, after.timestamp(), y_after)
            date_pairs.append((date, y))

    return date_pairs


def main():
    for x in t:
        date = datetime.strptime(x, "%m/%d/%y %H:%M")
        dates.append(date)

    start = date_str_to_epoch(t[0])
    end = date_str_to_epoch(t[-1])

    axvlines_raw = init_axvlines(start, end)
    for x in axvlines_raw:
        axvlines.append(datetime.fromtimestamp(x))

    # gets estimated usage
    pairs = (usage_estimates(axvlines, dates, e))
    est_x, est_y = zip(*pairs)

    # plots dates on x, energy usage on y
    fig, ax = plt.subplots()
    # format x-axis values
    xfmt = mdates.DateFormatter('%d-%m %H:%M')
    ax.xaxis.set_major_formatter(xfmt)

    plt.xticks(rotation=90)

    plt.plot(dates, e)
    # plot estimates
    plt.plot(est_x, est_y, color="black")
    plt.xticks(dates,fontsize='small')

    # graphs vertical lines for reference of days
    for x in axvlines:
        plt.axvline(x=x, color='red', linestyle='--')

    plt.show()


if __name__ == '__main__':
    main()

