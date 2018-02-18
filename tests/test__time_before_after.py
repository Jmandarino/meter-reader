from unittest import TestCase
from main import time_before_after

from datetime import datetime

class Test_time_before_after(TestCase):

    def test__time_before_after(self):

        # test if we only have 1 date before the given target
        before_date = datetime(2015,1,1)
        after_date = datetime(3000, 1, 1)
        date1, date2 = time_before_after([before_date], datetime.now())

        self.assertEquals(date1, before_date)
        self.assertEquals(date2, None)

        # test if we only have 1 date after the given target
        date1, date2 = time_before_after([after_date], datetime.now())

        self.assertEquals(date1, None)
        self.assertEquals(date2, after_date)


        # if between dates return those dates
        date1, date2 = time_before_after([before_date, after_date], datetime.now())

        self.assertEquals(date1, before_date)
        self.assertEquals(date2, after_date)

        # test multiple dates

        immed_before = datetime(2016,1,1)
        immed_after = datetime(2100, 1, 1)

        date1, date2 = time_before_after([before_date, immed_before, immed_after, after_date], datetime.now())

        self.assertEquals(date1, immed_before)
        self.assertEquals(date2, immed_after)












