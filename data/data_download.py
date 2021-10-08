import requests, zipfile, io, os

# give list of desired folder
save_folder = './'

#give list of year you want: min 2005, max 2018, last two year are different for now...
years = [2005,2006]


def get_url(year):
    # url for shapefile of amazon for a specific year (accumulated)
    url = "http://www.dpi.inpe.br/prodesdigital/dadosn/mosaicos/" + year + "/PDigital2000_" + year + "_AMZ_shp.zip"
    return url


for year in years:
    url = get_url(str(year))
    print(f'downloading data from: {url}')
    save_path = os.path.join(save_folder, str(year))
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    r = requests.get(url, allow_redirects=True, stream=True)
    # save zip file
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(save_path)




