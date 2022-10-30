import cv2
from sklearn.cluster import KMeans
import numpy as np
import os

WINDOW_SIZE = 5

def binarization_image(image, n_cluster):
  sub_w = image.shape[1]
  sub_h = image.shape[0]
  if sub_w % 2 != 0 and sub_h % 2 != 0:
    vectorized = image[:-1]
    vectorized = vectorized.reshape((-1, 2))
  else:
    vectorized = image.reshape((-1, 2))
  kmeans = KMeans(n_clusters=n_cluster, random_state=0).fit(vectorized)
  centers = np.uint8(kmeans.cluster_centers_)
  center_mean = centers.mean()
  threshed_image = np.zeros(image.shape, np.uint8)
  threshed_image[np.where(image > center_mean)] = 255
  return threshed_image

if __name__ == "__main__":
  image_folder_path = '/home/gustavo/Documentos/tcc/images/300dpi'
  images = os.listdir(image_folder_path)
  for image_filename in images:
    image_path = f'{image_folder_path}/{image_filename}'
    clusters = list(range(1, 6))
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    (h, w) = image.shape[:2]
    height_range = np.linspace(0, h, num=WINDOW_SIZE).astype(int)
    width_range = np.linspace(0, w, num=WINDOW_SIZE).astype(int)
    for n_cluster in clusters:
      width_ref = None
      threshed_images = []
      for width in width_range:
        if width_ref is None:
          width_ref = width
        else:
          height_ref = None
          horizontal_images = []
          for height in height_range:
            if height_ref is None:
              height_ref = height
            else:
              sub_image = image[height_ref:height, width_ref:width]
              sub_image = binarization_image(sub_image, n_cluster)
              height_ref = height
              horizontal_images.append(sub_image)
          threshed_images.append(horizontal_images)
          width_ref = width
      horizontal_images = []
      for threshed_image in threshed_images:
        h_img = np.vstack(threshed_image)
        horizontal_images.append(h_img)
      v_img = np.hstack(horizontal_images)
      cv2.imwrite(f'C_{n_cluster}/{image_filename}', v_img)
      print(f"Saved: C_{n_cluster}/{image_filename}")
