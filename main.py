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
# TODO: formalize/standardize comment style
# TODO: unify dates and meter reading
# TODO: should target values come first or last in method calls

# stores processed dates datetime objects
dates = []
# stores lines vertical lines for graph
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
    "2/12/18 19:19",
    "2/13/18 8:37",
    "2/13/18 19:00",
    "2/14/18 8:31",
    "2/14/18 17:58",
    "2/15/18 9:13",
    "2/16/18 9:11",
    "2/16/18 18:42",
    "2/16/18 23:21",
    "2/17/18 12:41",
    "2/17/18 19:12",
    "2/18/18 8:34",
    "2/18/18 20:50",
    "2/19/18 8:30",
    "2/19/18 18:28",
    "2/20/18 8:29",
    "2/20/18 18:08",
    "2/21/18 8:22",
    "2/21/18 19:27",
    "2/22/18 8:27",
    "2/22/18 18:52",
    "2/22/18 19:28",
    "2/23/18 8:27",
    "2/23/18 9:48",
    "2/23/18 18:08",
    "2/25/18 11:13",
    "2/25/18 16:15",
    "2/25/18 21:35",
    "2/26/18 9:05",
    "2/27/18 8:38",
    "2/27/18 18:42",
    "2/28/18 8:58",
    "2/28/18 18:25",
    "3/1/18 8:28", 
    "3/1/18 17:52",
    "3/2/18 8:44",
    "3/2/18 18:47",
    "3/3/18 10:45",
    "3/3/18 17:12",
    "3/4/18 12:23",
    "3/4/18 17:52",
    "3/5/18 8:37",
    "3/5/18 20:39",
    "3/6/18 8:36",
    "3/6/18 18:09",
    "3/8/18 8:32",
    "3/9/18 12:11",
    "3/10/18 8:57",
    "3/10/18 8:57",
    "3/11/18 12:22",
    "3/12/18 8:32",
    "3/12/18 17:46",
    "3/13/18 8:32",
    "3/13/18 18:24",
    "3/14/18 8:40",
    "3/15/18 8:38",
    "3/15/18 18:02",
    "3/16/18 8:22",
    "3/17/18 9:58",
    "3/18/18 10:49",
    "3/18/18 23:23",
    "3/19/18 8:33",
    "3/20/18 8:33",
    "3/22/18 8:36",
    "3/22/18 19:21",
    "3/26/18 8:29",
    "3/27/18 8:30",
    "3/28/18 8:30",
    "3/30/18 8:30",
    "3/31/18 8:30",
    "4/2/18 8:28",
    "4/3/18 8:30",
    "4/4/18 8:30",
    "4/5/18 8:37",
    "4/5/18 18:04",
    "4/6/18 8:23",
    "4/7/18 8:15",
    "4/9/18 8:31",
    "4/10/18 8:17",
    "4/17/18 8:32",



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
    52952,
    52958,
    52961,
    52968,
    52973,
    52979,
    52987,
    52993,
    52995,
    53000,
    53010,
    53015,
    53021,
    53025,
    53029,
    53034,
    53040,
    53046,
    53056,
    53061,
    53069,
    53069,
    53075,
    53079,
    53081,
    53102,
    53104,
    53107,
    53112,
    53117,
    53119,
    53126,
    53127,
    53132,
    53134,
    53140,
    53144,
    53152,
    53154,
    53159,
    53164,
    53168,
    53175,
    53179,
    53183,
    53198,
    53216,
    53225,
    53225,
    53239,
    53249,
    53251,
    53257,
    53262,
    53267,
    53279,
    53282,
    53288,
    53302,
    53307,
    53321,
    53324,
    53336,
    53356,
    53361,
    53396,
    53405,
    53411,
    53430,
    53439,
    53455,
    53464,
    53475,
    53484,
    53489,
    53494,
    53500,
    53527,
    53535,
    53607,
]

"""
HELPER FUNCTIONS
"""


def _calc_slope(x, y, x1, y1):
    return (y1 - y)/(x1 - x)


def _calc_y_incpt(slope, x, y):
    return y - (slope * x)


def _calc_y_value(slope, x, b):
    return slope * x + b


def _is_between(date_target, date_before, date_after):
    return date_before < date_target < date_after


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
    b = _calc_y_incpt(m, x_before, y_before)
    return _calc_y_value(m, x_poi, b)


def date_str_to_epoch(string):
    """
    Converts from d/m/yy hh:mm to epoch time
    
    :param string: inputted time as a string
    :return: time in epoch time
    """
    # 2/5/18 00:00"
    d = datetime.strptime(string, "%m/%d/%y %H:%M")
    unixtime = time.mktime(d.timetuple())
    return unixtime


# TODO: in-> epoch out->epoch or datetime->dateime
def init_axvlines(min_, max_, days):
    """
    Creates an array of date markers in epoch time  from a start and end date
    :param days: 
    :param min_: start date (in epoch time)
    :param max_: end date (in epoch time)
    :return: list() of timestamps
    """
    if min_ is None or min_ == max_:
        return []

    output = []
    # get the midnight value (12:01/0:01) for the start time
    # indicate the the start of the first day
    cur = datetime.fromtimestamp(min_)
    cur = cur.replace(hour=0, minute=0, second=0, microsecond=0)
    output.append(cur.timestamp())

    # TODO: only work on datetime objects and
    while cur.timestamp() < max_:
        cur = cur + timedelta(days=days)
        output.append(cur.timestamp())

    return output


def time_before_after(dates, target):
    """
    :param dates: list of datetime objects
    :param target: datetime object
    :return: datetime before and datetime after
    """
    # take care of edge case
    if dates == [] or target is None:
        return None, None

    # if target is either smaller than or greater than date range
    if dates[0] >= target:
        return None, dates[0]
    elif dates[-1] <= target:
        return dates[-1], None

    # maybe binary search?
    start = 0
    while start < len(dates):

        if _is_between(target, dates[start], dates[start+1]):
            return dates[start], dates[start + 1]
        elif dates[start] == target:
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

    global axvlines_raw
    axvlines_raw = init_axvlines(start, end, 7)
    for x in axvlines_raw:
        axvlines.append(datetime.fromtimestamp(x))

    # gets estimated usage
    pairs = (usage_estimates(axvlines, dates, e))
    est_x, est_y = zip(*pairs)

    # pairs (date, meter_reading)
    daily_increase = []
    for x in range(2,len(pairs)):
        increase = pairs[x][1] - pairs[x - 1][1]
        daily_increase.append(increase)
        print(increase)
    print("Daily Average:" + str(sum(daily_increase)/float(len(daily_increase))))

    # plots dates on x, energy usage on y
    fig, ax = plt.subplots()
    # format x-axis values
    xfmt = mdates.DateFormatter('%d-%m %H:%M')
    ax.xaxis.set_major_formatter(xfmt)

    plt.xticks(rotation=90)

    plt.plot(dates, e)
    # plot estimates
    plt.plot(est_x, est_y, color="black")
    plt.xticks(dates, fontsize='small')

    # graphs vertical lines for reference of days
    for x in axvlines:
        plt.axvline(x=x, color='red', linestyle='--')

    plt.show()


if __name__ == '__main__':
    main()
