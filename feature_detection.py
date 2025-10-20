import argparse
import os
import cv2
# import dlib
import yaml
import numpy as np
import mediapipe as mp
from google.colab.patches import cv2_imshow

def parse_args():
    parser = argparse.ArgumentParser(description='''outputs features''')
    parser.add_argument('-i', '--iteration', type=str, default='1', help='iteration number')
    parser.add_argument('-s', default=False, help='show images')

    args = parser.parse_args()
    return args

def format_id(id):
  id_str = str(id)
  while len(id_str) < 3:
    id_str = "0" + id_str
  return id_str

def process_file_names(path_id):
  all_ids = []
  path = format_id(path_id)
  fn_dict = {
      "01":{},
      "02":{}
  }
  file_names = []
  for filename in os.listdir(f'/content/multi_PIE_crop_128/{path}/'):
      file_names.append(filename)

  try:
    for file_name in file_names:
        split = file_name.split('_')
        absolute_file_name = f'multi_PIE_crop_128/{path}/{file_name}'
        if split[3] not in ['010', '110','120','240']:
          if split[4] in fn_dict[split[2]].keys():
            fn_dict[split[2]][split[4]]['img'].append(absolute_file_name)
          else:
            fn_dict[split[2]][split[4]] = {}
            fn_dict[split[2]][split[4]]['img'] = [absolute_file_name]
          if split[3]=='051':
            fn_dict[split[2]][split[4]]['imgGT'] = absolute_file_name
            id = int(f"{split[2]}{split[3]}{split[4]}")
            if not id in all_ids:
              all_ids.append(id)
            fn_dict[split[2]][split[4]]['id'] = all_ids.index(id)+1

  except:
    print(fn_dict, split, fn_dict[split[2]].keys())

  return fn_dict

def retrieve_feature_dict(feature_tuple, name):
  # x, y, width, height
  max_dims = {
    'eye':{'height':22, 'width':44},
    'nose':{'height':66, 'width':46},
    'mouth':{'height':25, 'width':69},
  }
  
  return {}

def plot_features_mediapipe(image, img, imgGT, id, show_img):
  # Initialize MediaPipe Face Mesh
  mp_face_mesh = mp.solutions.face_mesh
  # Create a Face Mesh instance
  face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)
  # Convert the image to RGB format
  image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

  # Process the image with Face Mesh
  results = face_mesh.process(image_rgb)

  offset_x = 10
  offset_y = 5
  offset_y_eyes = 3
  offset_x_eyes = 2
  offset_x_nose = 5
  offset_y_nose = 5
  offset_x_mouth = 2
  offset_y_mouth = 2

  result_dict = {}

  # Check if any faces are detected
  if results.multi_face_landmarks:
      # Get the facial landmarks
      landmarks = results.multi_face_landmarks[0].landmark

      height, width, _ = image.shape

      right_eye = (int(landmarks[226].x*width)-offset_x_eyes, int(landmarks[27].y*height)-offset_y_eyes, int(landmarks[133].x*width)-int(landmarks[226].x*width)+2*offset_x_eyes, int(landmarks[23].y*height)-int(landmarks[27].y*height)+2*offset_y_eyes)
      left_eye = (int(landmarks[463].x*width)-offset_x_eyes, int(landmarks[257].y*height)-offset_y_eyes, int(landmarks[446].x*width)-int(landmarks[463].x*width)+2*offset_x_eyes, int(landmarks[253].y*height)-int(landmarks[257].y*height)+2*offset_y_eyes)
      nose = (int(landmarks[98].x*width)-offset_x_nose, int(landmarks[168].y*height)-offset_y_nose, int(landmarks[327].x*width)-int(landmarks[98].x*width)+2*offset_x_nose, int(landmarks[2].y*height)-int(landmarks[168].y*height)+2*offset_y_nose)
      mouth = (int(landmarks[57].x*width)-offset_x_mouth, int(landmarks[0].y*height)-offset_y_mouth, int(landmarks[287].x*width)-int(landmarks[57].x*width)+2*offset_x_mouth, int(landmarks[17].y*height)-int(landmarks[0].y*height)+2*offset_y_mouth)
      
      # right_eye = (int(landmarks[226].x*width), int(landmarks[27].y*height), int(landmarks[133].x*width)-int(landmarks[226].x*width), int(landmarks[23].y*height)-int(landmarks[27].y*height))
      # left_eye = (int(landmarks[463].x*width), int(landmarks[257].y*height), int(landmarks[446].x*width)-int(landmarks[463].x*width), int(landmarks[253].y*height)-int(landmarks[257].y*height))
      # nose = (int(landmarks[98].x*width), int(landmarks[168].y*height), int(landmarks[327].x*width)-int(landmarks[98].x*width), int(landmarks[2].y*height)-int(landmarks[168].y*height))
      # mouth = (int(landmarks[57].x*width), int(landmarks[0].y*height), int(landmarks[287].x*width)-int(landmarks[57].x*width), int(landmarks[17].y*height)-int(landmarks[0].y*height))
      
      # retrieve_feature_dict(right_eye, 'eye')

      result_dict['Left eye'] = {'height':left_eye[3], 'width':left_eye[2], 'x':min(128-44,left_eye[1]), 'y':min(128-22,left_eye[0])}
      result_dict['Mouth'] = {'height':mouth[3], 'width':mouth[2], 'x':min(128-69,mouth[1]), 'y':min(128-25,mouth[0])}
      result_dict['Nose'] = {'height':nose[3], 'width':nose[2], 'x':min(128-46,nose[1]), 'y':min(128-66,nose[0])}
      result_dict['Right eye'] = {'height':right_eye[3], 'width':right_eye[2], 'x':min(128-44,right_eye[1]), 'y':min(128-22,right_eye[0])}
      result_dict['id'] = id
      result_dict['img'] = img
      result_dict['imgGT'] = imgGT

      cv2.rectangle(image, left_eye, (0, 255, 0), 2)
      cv2.rectangle(image, right_eye, (0, 255, 0), 2)
      cv2.rectangle(image, nose, (0, 255, 0), 2)
      cv2.rectangle(image, mouth, (0, 255, 0), 2)

      # Display the image
      if show_img:
        cv2_imshow(image)
  else:
      print("No faces detected.")

  return result_dict

def load_images_and_plot_features(files, show_img):
  images = []
  result_dict = {}
  for filename in files['img']:
      img = cv2.imread(filename)
      print("file", filename)
      if img is not None:
          # plot_features(img)
          # img = cv2.imread(filename)
          result = plot_features_mediapipe(img, filename, files['imgGT'], files['id'], show_img)
          if bool(result):
            result_dict[filename] = result
      else:
          print("not an image")
          pass
  return result_dict


args = parse_args()
iter = int(args.iteration)
show_img = args.s
results_dict = dict()
# for i in range(1,2):
print(f"{iter}th iteration")
fn_dict = process_file_names(iter)
#   print(fn_dict)
# for super_key in fn_dict.keys():
for key in fn_dict['01'].keys():
    result = load_images_and_plot_features(fn_dict['01'][key], show_img)
    results_dict.update(result)
for key in fn_dict['02'].keys():
    result = load_images_and_plot_features(fn_dict['02'][key], show_img)
    results_dict.update(result)
#   print(result)

with open(f'output-{iter}.yaml', 'w') as file:
    yaml.dump(results_dict, file)