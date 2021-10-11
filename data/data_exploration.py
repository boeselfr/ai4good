import os
#import matplotlib.pyplot as plt
import geopandas as gpd
import earthpy as et

save_path = "/Users/fredericboesel/Documents/Data Science Master/Herbstsemester 2021/AI4Good/ai4good/data/2005/PDigital2005_AMZ_pol.shp"
sjer_plot_locations = gpd.read_file(save_path)

print(sjer_plot_locations.head(5))


