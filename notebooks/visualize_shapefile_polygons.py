{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import dependencies\n",
    "import matplotlib.pyplot as plt\n",
    "import geopandas as gpd\n",
    "import earthpy as et\n",
    "import contextily as ctx\n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load the shape data\n",
    "save_path = \"path/to/shapefile\"\n",
    "amz_shp = gpd.read_file(save_path)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Plot the shape data\n",
    "amz = amz_shp.to_crs(epsg=3857)\n",
    "ax = amz.plot(figsize=(10,10))\n",
    "ctx.add_basemap(ax, url=ctx.providers.Stamen.Terrain)\n",
    "ax.set_axis_off()\n",
    "plt.show()"
   ]
  }
 ],
 "metadata": {
  "interpreter": {
   "hash": "1f29b3b476751f6839fd006815e784e36762a848ecd053bdbad9cac15b3615be"
  },
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}