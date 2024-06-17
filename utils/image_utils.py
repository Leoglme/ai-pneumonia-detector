import os
from PIL import Image

class ImageUtils:
    @staticmethod
    def filter_images(data_dir, img_size=(256, 256)):
        """
        Filter out images that are too small and gather statistics on the image sizes.
        """
        min_img_size = (img_size[0] * 2, img_size[1] * 2)
        image_stats = {
            'total_images': 0,
            'min_size': (float('inf'), float('inf')),
            'max_size': (0, 0),
            'avg_width': 0,
            'avg_height': 0,
            'filtered_images': 0
        }
        filepaths = []
        total_width, total_height = 0, 0

        for dirpath, _, filenames in os.walk(data_dir):
            for filename in filenames:
                if filename.lower().endswith(('png', 'jpg', 'jpeg')):
                    filepath = os.path.join(dirpath, filename)
                    with Image.open(filepath) as img:
                        width, height = img.size
                        image_stats['total_images'] += 1
                        total_width += width
                        total_height += height
                        if width < min_img_size[0] or height < min_img_size[1]:
                            image_stats['filtered_images'] += 1
                        else:
                            filepaths.append(filepath)
                            image_stats['min_size'] = (
                                min(image_stats['min_size'][0], width),
                                min(image_stats['min_size'][1], height)
                            )
                            image_stats['max_size'] = (
                                max(image_stats['max_size'][0], width),
                                max(image_stats['max_size'][1], height)
                            )
                            image_stats['avg_width'] += width
                            image_stats['avg_height'] += height

        remaining_images = image_stats['total_images'] - image_stats['filtered_images']
        if remaining_images > 0:
            image_stats['avg_width'] /= remaining_images
            image_stats['avg_height'] /= remaining_images

        image_stats['avg_width'] = round(image_stats['avg_width'])
        image_stats['avg_height'] = round(image_stats['avg_height'])

        # No second pass needed; we have already filtered images below the min size
        filtered_filepaths = [fp for fp in filepaths if
                              Image.open(fp).size[0] >= min_img_size[0] and Image.open(fp).size[1] >=
                              min_img_size[1]]
        return filtered_filepaths, image_stats