import ee
from tqdm import tqdm

from .common import GEE_DEFAULT_CRS

def export_image_collection_to_gdrive(
    image_collection: ee.ImageCollection,
    folder: str,
    scale: int = 40,
):
    """Exports the given image collection to Google Drive.
    Each image is saved as a separate GeoTIFF file.
    
    :param image_collection: the image collection
    :param folder: the name of the folder on your Google Drive
    :param region: the polygon of the region of interest
    """

    num_images = image_collection.size().getInfo()
    image_collection_list = image_collection.toList(num_images)

    status_msg = """
    An export task has been started for image {}.
    You may inspect its status in the "Tasks" tab of the gee code editor.
    """

    for idx in tqdm(range(0, num_images)):
        image = ee.Image(image_collection_list.get(idx))

        # Use the image id as name
        image_id = image.id().getInfo()

        # Start an export task
        ee.batch.Export.image.toDrive(
            description=image_id,
            fileNamePrefix=image_id,
            image=image.toFloat(),  # Convert to float before starting the task
            folder=folder,
            scale=scale,
            crs=GEE_DEFAULT_CRS,
        ).start()

        print(status_msg.format(image_id))