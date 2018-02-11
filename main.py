import matplotlib
from datetime import datetime
import pandas as pd
import time
from datetime import timezone
import matplotlib.pyplot as plt


t = [
"2/7/18 18:58",
"2/7/18 23:16",
"2/8/18 00:01",
"2/8/18 08:05",
"2/8/18 08:52",
"2/8/18 18:30",
"2/8/18 20:35",
"2/8/18 23:17",
"2/9/18 16:35",
"2/9/18 18:31",
"2/9/18 21:12"]

e = [
52900,
52901,
52901,
52903,
52903,
52903,
52906,
52907,
52911,
52912,
52914,
]

dates = []

def convert(str):
    # 2/5/18 00:00"
    d = datetime.strptime(str, "%m/%d/%y %H:%M")
    unixtime = time.mktime(d.timetuple())
    return d


for x in t:
    dates.append(convert(x))


print(dates)
plt.plot(dates, e)
plt.xticks(dates,fontsize='small')
plt.show()

