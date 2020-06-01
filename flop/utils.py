import time

def now_to_date(format_string="%Y-%m-%d-%H:%M:%S"):
  time_stamp = int(time.time())
  time_array = time.localtime(time_stamp)
  str_date = time.strftime(format_string, time_array)
  return str_date