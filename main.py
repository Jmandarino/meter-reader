import matplotlib
from datetime import datetime
import pandas as pd
import time
from datetime import timezone
import matplotlib.pyplot as plt

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
"2/10/18 20:37"
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
]


# stores processed dates in epoch time
dates = []

# stores lines verticle lines for graph
axvlines_raw = ["2/7/18 0:01", "2/8/18 0:01", "2/9/18 0:01", "2/10/18 0:01", "2/11/18 0:01"]
# stores processed lines (times converted to epoch time)
axvlines = []


def convert(str):
    """
    Converts from d/m/yy hh:mm to epoch time
    
    :param str: inputed time as a string
    :return: time in epoch time
    """
    # 2/5/18 00:00"
    d = datetime.strptime(str, "%m/%d/%y %H:%M")
    unixtime = time.mktime(d.timetuple())
    return d


for x in t:
    dates.append(convert(x))

for x in axvlines_raw:
    axvlines.append(convert(x))


# plots dates on x, energy usage on y
plt.plot(dates, e)
plt.xticks(dates,fontsize='small')

# graphs verticle lines for reference of days
for x in axvlines:
    plt.axvline(x=x, color='red', linestyle='--')

plt.show()



