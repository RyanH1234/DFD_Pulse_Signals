import os
import json
import cv2
import matplotlib.pyplot as plt
import random
import time
import numpy as np

from sklearn.externals import joblib

from facenet_pytorch import MTCNN

random.seed(0)


def get_metadata(path):
    # 1. Retrieve metadata
    metadata_path = path + "/metadata.json"

    with open(metadata_path) as f:
        metadata = json.load(f)

    # 2. Append full path to path of metadata
    revised_metadata = []

    for key in metadata:
        val = metadata[key]

        vid_path = path + "/" + key
        label = val['label']

        revised_metadata.append((vid_path, label))

    return revised_metadata


def get_paths_to_videos(root_dir):
    '''
      Assumes the following dir structure

      +-- root dir
        +-- parent dir
          +-- child dir
            +-- vid #1
            +-- vid #2
            +-- metadata.json
          ....
    '''

    data = []

    for parent_dir in os.listdir(root_dir):
        for child_dir in os.listdir(root_dir + parent_dir):
            child_dir_path = root_dir + parent_dir + "/" + child_dir
            metadata = get_metadata(child_dir_path)
            data.extend(metadata)

    return data


def encode_labels(data):
    '''
      Assumes data has the following data structure 

      [
        (path_to_video, label),
        .....
      ]

      Where label is either 'REAL' or 'FAKE'
    '''

    encoded_data = []

    for path, label in data:
        if label == 'REAL':
            encoded_data.append((path, 0))
        else:
            encoded_data.append((path, 1))

    return encoded_data


def analyse_metadata(data):
    '''
      Assumes data has the following data structure

      [
        (path_to_video, label)
        .....
      ]

      'Label' component is binary where 0 represents 'REAL' and 1 represents 'FAKE'
    '''

    print("------------METADATA SUMMARY-------------")

    no_of_videos = len(data)
    print("No of Videos {}".format(no_of_videos))

    # counting % of real and fake videos
    fake = real = 0

    for _, label in data:
        if label == 0:
            real += 1
        else:
            fake += 1

    perc_real = round(real/no_of_videos * 100, 2)
    perc_fake = round(fake/no_of_videos * 100, 2)

    print("% of Real Videos {}% (e.g. {})".format(perc_real, real))
    print("% of Fake Videos {}% (e.g. {})".format(perc_fake, fake))

    return (real, fake)


def save_to_file(name, data):
    joblib.dump(data, name)


def read_from_file(path):
    loaded_model = joblib.load(path)
    return loaded_model


def analyse_videos(data):
    '''
      Assumes data has the following data structure

      [
        (path_to_video, label)
        .....
      ]
    '''

    vid_stats = []

    # the following process is computationally expensive
    # therefore output is saved to a local file 'video_stats' for intermediate use
    if(os.path.exists("video_stats")):
        vid_stats = read_from_file("video_stats")
    else:
        for vid_path, _ in data:
            print('Processing : {}'.format(vid_path))

            cap = cv2.VideoCapture(vid_path)

            framerate = int(cap.get(cv2.CAP_PROP_FPS))
            no_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

            vid_stats.append({
                'video_path': vid_path,
                'framerate': framerate,
                'no_frames': no_frames,
                'width': width,
                'height': height
            })

        save_to_file('video_stats', vid_stats)

    framerate = [stat['framerate'] for stat in vid_stats]
    no_frames = [stat['no_frames'] for stat in vid_stats]
    width = [stat['width'] for stat in vid_stats]
    height = [stat['height'] for stat in vid_stats]

    # Visualising the data
    # plt.subplot(2, 2, 1)
    # plt.hist(framerate)
    # plt.xlabel("Framerate")
    # plt.ylabel("Freq")
    # plt.title("Framerates Across The Dataset")

    # plt.subplot(2, 2, 2)
    # plt.hist(no_frames)
    # plt.xlabel("No. Of Frames")
    # plt.ylabel("Freq")
    # plt.title("No. Of Frames Across The Dataset")

    # plt.subplot(2, 2, 3)
    # plt.hist(width)
    # plt.xlabel("Width")
    # plt.ylabel("Freq")
    # plt.title("Width Across The Dataset")

    # plt.subplot(2, 2, 4)
    # plt.hist(height)
    # plt.xlabel("Height")
    # plt.ylabel("Freq")
    # plt.title("Height Across The Dataset")

    # plt.show()

    return vid_stats


def count_real_fake(data):
    '''
      Assume data has the following structure
      [
        { <path>, <label> }
      ]
    '''
    real = 0
    fake = 0

    for path, label in data:
        if label == 1:
            fake += 1
        else:
            real += 1

    return real, fake


def balance_data(data, breakpoint):
    '''
      Assumes data has the following structure
      [
        { <path>, <label> }
      ]
      Breakpoint is the no. of each class that should exist in the output
    '''

    balanced_data = []
    no_of_fake = no_of_real = 0

    for path, label in data:
        if label == 1 and no_of_fake < breakpoint:  # fake
            balanced_data.append((path, label))
            no_of_fake += 1

        elif label == 0 and no_of_real < breakpoint:
            balanced_data.append((path, label))
            no_of_real += 1

    return balanced_data


def retrieve_keypoints(landmarks):
    return {
        'left_eye': landmarks[0][0][0],
        'right_eye': landmarks[0][0][1],
        'nose': landmarks[0][0][2],
        'left_mouth': landmarks[0][0][3],
        'right_mouth': landmarks[0][0][4]
    }


def extract_face(frame, landmarks):
    '''
      Frames and landmarks are returned by the MTCNN detector
    '''

    features = retrieve_keypoints(landmarks)

    left_eye_x, left_eye_y = features['left_eye']
    left_mouth_x, left_mouth_y = features['left_mouth']
    right_mouth_x, right_mouth_y = features['right_mouth']

    y1 = int(left_eye_y - 10)
    y2 = int(left_mouth_y + 10)
    x1 = int(left_mouth_x - 10)
    x2 = int(right_mouth_x + 10)

    face = frame[
        y1:y2,
        x1:x2
    ]

    return face


def extract_ROI(frame, landmarks):
    '''
      Frames and landmarks are returned by the MTCNN detector
    '''

    keypoints = retrieve_keypoints(landmarks)

    left_eye_x, _ = keypoints['left_eye']
    right_eye_x, right_eye_y = keypoints['right_eye']
    _, right_mouth_y = keypoints['right_mouth']
    nose_x, nose_y = keypoints['nose']

    height = right_mouth_y - right_eye_y
    padding_top = int(height/4)
    padding_bottom = int(height/2)

    width = right_eye_x - left_eye_x
    padding_sides = int(width/5)

    ROI = frame[
        int(nose_y - padding_top):int(nose_y + padding_bottom),
        int(left_eye_x - padding_sides):int(right_eye_x + padding_sides)
    ]

    return ROI


def resize_frame(frame, resize_factor):
    '''
      Frame is a 3D numpy array e.g. width x height x channels
      By default we resize to 50% of it's original size
    '''
    height, width, layers = frame.shape
    new_height = int(height * resize_factor)
    new_width = int(width * resize_factor)
    frame = cv2.resize(frame, (new_width, new_height))
    return frame


def preprocess_video(vid_path):
    '''
      Takes in a single path and preprocesses int two steps
      1. Resize each frame
      2. Extract face
      3. Extract ROI
      4. Normalise each frame
    '''
    detector = MTCNN(keep_all=True)

    cap = cv2.VideoCapture(vid_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    no_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    errs = 0

    ROIs = []
    faces = []

    print("Processing {}".format(vid_path))
    start = time.time()

    while(cap.isOpened()):
        ret, frame = cap.read()

        if ret:
            frame = resize_frame(frame, 0.25)  # 1

            boxes, _, landmarks = detector.detect([frame], landmarks=True)

            if len(boxes.shape) != 3:
                errs += 1
            else:
                face = extract_face(frame, landmarks)  # 2
                ROI = extract_ROI(frame, landmarks)  # 3

                faces.append(face/255)  # 4
                ROIs.append(ROI/255)  # 4
        else:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("% Of Frames Err'd : {}% (e.g. {} frames / {} frames)".format(errs /
                                                                        no_frames * 100, errs, no_frames))

    end = time.time()
    print("Time elapsed : {} \n".format(end-start))

    stats = (vid_path, end-start, errs/no_frames)

    return ROIs, faces, stats


def remove_outliers(data, vid_stats):
    '''
      data is of the form 
      [
        { vid_path, label }
      ]

      vid_stats is of the form
      [
        { video_path, framerate, no_frames, width, height }
      ]
    '''

    print("------------REMOVE OUTLIERS-------------")

    no_real, no_fake = count_real_fake(data)
    print("BEFORE : no. of real {} , no. of fake {}".format(no_real, no_fake))

    outliers = []

    for vid_stat in vid_stats:
        path = vid_stat['video_path']
        framerate = vid_stat['framerate']
        width = vid_stat['width']
        height = vid_stat['height']
        no_of_frames = vid_stat['no_frames']

        if framerate < 25 or no_of_frames < 250 or width < 1500 or height < 750:
            outliers.append(path)

    updated_data = []

    for (path, label) in data:
        if path not in outliers:
            updated_data.append((path, label))

    no_real, no_fake = count_real_fake(updated_data)
    print("AFTER : no. of real {} , no. of fake {}".format(no_real, no_fake))

    return updated_data


def analyse_preprocessing(stats):
    '''
      Returns a list of paths which should be removed from the dataset as
      their err rates exceed some predefined value
    '''

    print("------------ANALAYSING PREPROCESSING DATA-------------")

    paths = [stat[0] for stat in stats]
    time = [stat[1] for stat in stats]
    err_rates = [stat[2] for stat in stats]

    print("NO OF VIDEOS PROCESSED : {}".format(len(paths)))

    avg_time = sum(time)/len(time)
    print("AVG TIME : {}s".format(round(avg_time, 2)))

    avg_err_rate = sum(err_rates)/len(err_rates) * 100
    print("AVG ERR RATE : {}%".format(round(avg_err_rate, 2)))

    # Visualising the data
    # plt.subplot(1, 2, 1)
    # plt.hist(err_rates)
    # plt.xlabel("err rate")
    # plt.ylabel("Freq")
    # plt.title("Err Rates Across The Dataset")

    # plt.subplot(1, 2, 2)
    # plt.scatter(time, err_rates)
    # plt.xlabel("time")
    # plt.ylabel("error rate")
    # plt.title("Err Rate Vs Avg Time")

    # plt.show()

    # accumulate those videos which need to be removed
    remove_vids = []
    for i, err_rate in enumerate(err_rates):
        if err_rate > 0.10:
            remove_vids.append(i)

    return remove_vids


def preprocessing(data, vid_stats):
    # 1 - remove outliers e.g. videos whose characteristics existed outside
    #     the maj. pop. in `analyse_videos`
    data = remove_outliers(data, vid_stats)
    no_of_real, no_of_fake = analyse_metadata(data)

    # 2 - balance our dataset with roughly 50% real and 50% fake
    balance = min(no_of_real, no_of_fake)

    # we shuffle to ensure we aren't bias to a certain group of actors
    random.shuffle(data)
    data = balance_data(data, balance)
    random.shuffle(data)
    analyse_metadata(data)

    # 3 - preprocess videos in data
    # 4 - resize each frame in each video
    # 5 - normalise each frame in each video
    preprocessed_data = []
    stats = []

    if(not os.path.exists("E:\preprocessed_data")):
        print("------------PREPROCESSING DATA-------------")

        for vid_path, label in data:
            ROIs, faces, video_stats = preprocess_video(vid_path)
            preprocessed_data.append(((ROIs, faces), label))
            stats.append(video_stats)

        save_to_file('E:\preprocessed_data', preprocessed_data)
        save_to_file('E:\preprocessing_stats', stats)

    # 6 - analyse the prepocessing stats returned
    stats = read_from_file("E:\preprocessing_stats")
    outliers = analyse_preprocessing(stats)

    # 7 - remove any videos which were polluted with high err rates
    print("------------REMOVING DATA-------------")
    print("Removing {} videos".format(len(outliers)))

    preprocessed_data = read_from_file("E:\preprocessed_data")
    data = []

    for i, vid_attributes in enumerate(preprocessed_data):
        if i not in outliers:
            data.append(vid_attributes)

    print("Dataset now contains {} elements".format(len(data)))

    return data


def extract_channel(dataframe):
    r = dataframe[:, :, 0].flatten()
    g = dataframe[:, :, 1].flatten()
    b = dataframe[:, :, 2].flatten()

    return r, g, b


def compute_mean_rgb(img):
    r, g, b = extract_channel(img)

    mean_r = np.mean(r)
    mean_g = np.mean(g)
    mean_b = np.mean(b)

    return mean_r, mean_g, mean_b,


def project_chrominance(r, g, b):
    x = (3.0 * r) - (2.0 * g)
    y = (1.5 * r) + g + (1.5 * b)
    return x, y


def empty_array(array):
    array = np.array(array)
    shape = array.shape

    if len(shape) < 3:
        return True

    x, y, z = shape

    if x == 0 or y == 0 or z == 0:
        return True

    return False


def rPPG(data):
    '''
      Extract the rPPG signals for each video and append them to the current data object

      [
        {frames, chrom, label},
        ...
      ]

      e.g. where frames is the raw frames containing the faces, chrom is the rppg signal, label
      is the class
    '''

    print("------------rPPG-------------")

    processed_data = []

    for (ROIs, faces), label in data:
        chroms = []

        no_of_frames = len(ROIs)
        errs = 0

        for ROI in ROIs:
            if empty_array(ROI):
                chroms.append(0)
                errs += 1
            else:
                r, g, b = compute_mean_rgb(ROI)
                x, y = project_chrominance(r, g, b)
                S = (x/y) - 1
                chroms.append(S)

        print("Err % Rate : {}%".format(errs/no_of_frames * 100))
        processed_data.append((ROIs, faces, chroms, label))

    print("Saving to file..")
    save_to_file('E:\signal_data', processed_data)


if __name__ == "__main__":
    data = get_paths_to_videos("E:/dfdc_train_all/")
    data = data[0:18000]
    data = encode_labels(data)

    analyse_metadata(data)
    vid_stats = analyse_videos(data)
    data = preprocessing(data, vid_stats)
    rPPG(data)

# TODO
# - handle multiple faces in the frame
# - remove those videos which have loss rate > 20% e.g. errs encountered in videos > some constant
# - add in possible checkpoints for crashing e.t.c
# - maybe convert from tuples to dictionaries for ease of use?
