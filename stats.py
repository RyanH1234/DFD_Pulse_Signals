from sklearn.externals import joblib
import os
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np

from run import read_from_file, save_to_file

def read_from_file(path):
    loaded_model = joblib.load(path)
    return loaded_model

def analyse_attribute(vid_stats, attribute_name):
  attr = [stat[attribute_name] for stat in vid_stats]
  attr_df = pd.DataFrame(attr)
  print("---------" + attribute_name + "----------")
  print(attr_df.describe())

def analyse_videos():
  print("-----------------STATISTICS FOR THE UNPROCESSED VIDEOS-------------")

  vid_stats = []
  
  if(os.path.exists("video_stats")):
      vid_stats = read_from_file("video_stats")

  analyse_attribute(vid_stats, "framerate")
  analyse_attribute(vid_stats, "no_frames")
  analyse_attribute(vid_stats, "width")
  analyse_attribute(vid_stats, "height")

def analyse_preprocessing():
  print("-----------------STATISTICS FOR THE PREPROCESSED VIDEOS-------------")

  stats = read_from_file("E:\preprocessing_stats")

  paths = [stat[0] for stat in stats]
  time = [stat[1] for stat in stats]
  err_rates = [stat[2] for stat in stats]

  print("NO OF VIDEOS PROCESSED : {}".format(len(paths)))

  avg_time = sum(time)/len(time)
  print("AVG TIME : {}s".format(round(avg_time, 2)))

  avg_err_rate = sum(err_rates)/len(err_rates) * 100
  print("AVG ERR RATE : {}%".format(round(avg_err_rate, 2)))

  # Visualising the data
  plt.subplot(1, 2, 1)
  plt.hist(err_rates)
  plt.xlabel("err rate")
  plt.ylabel("Freq")
  plt.title("Err Rates Across The Dataset")

  plt.subplot(1, 2, 2)
  plt.scatter(time, err_rates)
  plt.xlabel("time")
  plt.ylabel("error rate")
  plt.title("Err Rate Vs Avg Time")

  plt.show()

def display_faces_roi(data):
  fig = plt.figure(figsize=(8, 20))

  for i, (ROI, face) in enumerate(data):
    ax = fig.add_subplot(5, 2, 2*i+1)

    face = np.float32(face)
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = np.array(face)

    plt.imshow(face)
    ax.axis("off")
    ax.set_title("FACE")

    ax = fig.add_subplot(5, 2, 2*i + 2)

    ROI = np.float32(ROI)
    ROI = cv2.cvtColor(ROI, cv2.COLOR_BGR2RGB)
    ROI = np.array(ROI)

    plt.imshow(ROI)
    plt.axis("off")
    ax.set_title("ROI")

  plt.show()

def analyse_signal_data():
  data = []

  if(os.path.exists("E:\signal_data")):
    data = read_from_file("E:\signal_data")

  chroms = [len(chroms) for (ROIs, faces, chroms, label) in data]
  chroms_df = pd.DataFrame(chroms)
  print("---------CHROMS----------")
  print(chroms_df.describe())
  

def analyse_preprocessed_data():
  print("-----------------VISUALISING THE PREPROCESSED VIDEOS-------------")

  data = []

  if(os.path.exists("E:\signal_data")):
    data = read_from_file("E:\signal_data")

  real_data = []
  real_count = 0

  fake_data = []
  fake_count = 0

  for ROIs, faces, features, label in data:
      if(label == 0 and real_count < 5):  # e.g real
          real_data.append((ROIs[0], faces[0]))
          real_count += 1

      if(label == 1 and fake_count < 5):  # e.g. fake
          fake_data.append((ROIs[0], faces[0]))
          fake_count += 1
  
  display_faces_roi(real_data)
  display_faces_roi(fake_data)

  # real data
  # for i, (ROI, face) in enumerate(real_data):
  #     ax = fig.add_subplot(5, 2, 2*i+1)

  #     face = np.float32(face)
  #     face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
  #     face = np.array(face)

  #     plt.imshow(face)
  #     ax.axis("off")
  #     ax.set_title("FACE")

  #     ax = fig.add_subplot(5, 2, 2*i + 2)

  #     ROI = np.float32(ROI)
  #     ROI = cv2.cvtColor(ROI, cv2.COLOR_BGR2RGB)
  #     ROI = np.array(ROI)

  #     plt.imshow(ROI)
  #     plt.axis("off")
  #     ax.set_title("ROI")

  # plt.show()

  # # fake data
  # for i, (ROI, face) in enumerate(fake_data):
  #     ax = plt.subplot(10, 2, 2*i + 1)
  #     plt.imshow(face)
  #     plt.axis("off")
  #     ax.set_title("FACE")

  #     ax = plt.subplot(10, 2, 2*i + 2)
  #     plt.imshow(ROI)
  #     plt.axis("off")
  #     ax.set_title("ROI")

  # plt.show()

  # lengths = []
  # for (ROIS, faces, features, label) in data:
  #     lengths.append(len(features))

  # plt.hist(lengths)
  # plt.xlabel("Lengths Of Signal")
  # plt.ylabel("Freq")
  # plt.title("Length of Signal Across The Dataset")
  # plt.show()


def slice_data(data):
    sliced_data = []

    for (ROIS, faces, features, label) in data:
        features = features[0:265]
        sliced_data.append((ROIS, faces, features, label))

    return sliced_data


def visualise_signal_data():
  data = read_from_file("E:\signal_data")
  data = slice_data(data)

  pulse_signals = []
  labels = []

  for (ROIS, faces, features, label) in data:
    pulse_signals.append(features)
    labels.append(label)

  for pulse_signal, label in zip(pulse_signals, labels):
    if label == 1:
        plt.plot(pulse_signal)
        plt.xlabel("time")
        plt.ylabel("pulse signal")
        plt.show()
        break

  # pulse_signal = pulse_signals[0]
  # label = labels[0]

  # print(label)




# analyse_videos()
# analyse_preprocessing()
# analyse_preprocessed_data()
# analyse_signal_data()
# analyse_model()
visualise_signal_data()