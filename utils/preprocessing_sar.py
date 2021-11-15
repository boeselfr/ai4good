# some utils that can be applied to preprocess sar imagery
# using OpenSar Toolkit
from pathlib import Path
from ost import Sentinel1Scene

#index for specific image:
id = 'S1A_IW_GRDH_1SDV_20200803T090644_20200803T090709_033740_03E91F_AFFB'

s1 = Sentinel1Scene(id)

s1.info()

home = Path.home()

# create a processing directory
output_dir = home.joinpath('OST_Tutorials', 'Tutorial_1')
output_dir.mkdir(parents=True, exist_ok=True)
print(str(output_dir))



