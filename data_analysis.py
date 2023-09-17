import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data = pd.read_csv("/Users/xavierparker/Desktop/IRP/dataFile.csv")

write = True

less = 0
iter_total = 0
less_1 = 0

for swarm, data_dict in data.iterrows():
    if write:
        #print(data_dict["global minimum"])
        minimum = abs(data_dict["global minimum"])
        swarm = abs(data_dict["Swarm minimum"])
        iteration = data_dict["iteration found"]
        iter_total += iteration
        #print(minimum)
        #print(swarm)
        #print(iteration)
        if data_dict["Swarm minimum"] <= data_dict["global minimum"]:
            less += 1
            
        else:
            percentage_dif = 100 * ((abs(minimum-swarm))/((swarm+minimum)/2))
            #print(percentage_dif)
            if percentage_dif <= 1:
                less_1 += 1

        #write = False


iter_average = iter_total / 1000
print("""Iteration average: {}
Swarm less than minimum: {}
Percentage difference less than 1%: {}""".format(iter_average, less, less_1))
