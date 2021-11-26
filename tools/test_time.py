from time import gmtime, strftime, time

timme = strftime("%Y-%d-%m %H:%M:%S", gmtime())
print(timme)
print(timme[5:-3])
timme = timme[5:-3].replace('-', '_')
timme = timme.replace(' ', '_')
timme = timme.replace(':', '_')
print(timme)
hour = (str)((int)(timme.split('_')[-2]) + 8)

print(timme)