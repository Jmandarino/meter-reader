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


# stores processed dates in epoch time
dates = []

# stores lines verticle lines for graph
axvlines_raw = ["2/7/18 0:01", "2/8/18 0:01", "2/9/18 0:01", "2/10/18 0:01", "2/11/18 0:01", "2/12/18 0:01", "2/13/18 0:01"]
# stores processed lines (times converted to epoch time)
axvlines = []

"""
HELPER FUNCTIONS
"""
def _calc_slope(x,y,x1,y1):
    return (x1 - x)/(y1 - y)


def _calc_y_incpt(slope, x, y):
    return y - (slope * x)


def _calc_y_value(slope, x, b):
    return slope * x + b


"""
FUNCTIONS
"""
# TODO: fix the format to be more readable


def calc_est_y(x_poi, x,y, x1, y1):
    """
    Calculate an estimate value for y for our x value Point of Interest

    1. find equation for line, y = mx +b
    2. plug in x to find y

    :param x_poi: x value we want to find y for
    :param x: x value before x_poi
    :param y: y value before x_poi
    :param x1: x value after x_poi
    :param y1: y value after _poi
    :return: Estimate of y value at that time
    """
    m = _calc_slope(x,y,x1,y1)
    b = _calc_y_incpt(m,x,y)
    return _calc_y_value(m, x_poi, b)


def convert(str):
    """
    Converts from d/m/yy hh:mm to epoch time
    
    :param str: inputed time as a string
    :return: time in epoch time
    """
    # 2/5/18 00:00"
    d = datetime.strptime(str, "%m/%d/%y %H:%M")
    unixtime = time.mktime(d.timetuple())
    return unixtime


# TODO: in-> epoch out->epoch or datetime->datime
def init_axvlines(min, max):
    """
    Creates an array of date markers in epoch time  from a start and end date
    :param min: start date (in epoch time)
    :param max: end date (in epoch time)
    :return:
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


for x in t:
    date = datetime.strptime(x, "%m/%d/%y %H:%M")
    dates.append(date)

start = convert(t[0])
end = convert(t[-1])

axvlines_raw = init_axvlines(start, end)
for x in axvlines_raw:
    axvlines.append(datetime.fromtimestamp(x))

print(dates)

# plots dates on x, energy usage on y

fig, ax = plt.subplots()
# format x-axis values
xfmt = mdates.DateFormatter('%d-%m %H:%M')
ax.xaxis.set_major_formatter(xfmt)

plt.xticks(rotation=90)

plt.plot(dates, e)
plt.xticks(dates,fontsize='small')

# graphs verticle lines for reference of days
for x in axvlines:
    plt.axvline(x=x, color='red', linestyle='--')


plt.show()



