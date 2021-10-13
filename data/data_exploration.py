import os
import matplotlib.pyplot as plt
import geopandas as gpd
import earthpy as et
import contextily as ctx


save_path = "/Users/fredericboesel/Documents/Data Science Master/Herbstsemester 2021/AI4Good/ai4good/data/2005/PDigital2005_AMZ_pol.shp"
amz_shp_2005 = gpd.read_file(save_path)

print(amz_shp_2005.head(5))
print(amz_shp_2005.shape)


fig, ax = plt.subplots(figsize = (10,10))
amz_shp_2005.plot(ax=ax)
plt.show()

"""amz = amz_shp_2005.to_crs(epsg=3857)
ax = amz.plot(figsize=(10,10))
ctx.add_basemap(ax, url=ctx.providers.Stamen.Terrain)
ax.set_axis_off()
plt.show()"""

