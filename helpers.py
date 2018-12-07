import os

import numpy as np

from constants import FRAMES_IN_SEQ, FACE_SHAPE


def batch_gen(dataset_type, batch_size):
    """Batch generator
    
    :param dataset_type: dataset_type (same as directory name with the dataset)
    :param batch_size: batch size
    :yield: batch and ground truth labels
    """
    files = [item for item in os.listdir(dataset_type) if item.endswith("npz")]
    while True:
        np.random.shuffle(files)
        steps = len(files) // batch_size
        batch = np.zeros((batch_size, FRAMES_IN_SEQ, FACE_SHAPE[0], FACE_SHAPE[1], 3), dtype=np.uint8)
        ground_truth = np.zeros((batch_size, 1))
        for i in range(steps):
            batch_files = files[i * batch_size : (i + 1) * batch_size]
            for idx, file in enumerate(batch_files):
                seq = np.load(os.path.join(dataset_type, file))["seq"]
                batch[idx] = seq
                ground_truth[idx] = 1 if "real" in file else 0
            yield batch, ground_truth

def num_steps_per_epoch(dataset_type, batch_size):
    """Calculate number of steps per epoch
    
    :param dataset_type: dataset type (same as directory name with the dataset)
    :param batch_size: batch size
    :return: number of steps per epoch
    """
    files = [item for item in os.listdir(dataset_type) if item.endswith("npz")]
    steps = len(files) // batch_size
    return steps

def get_face(frames, out_shape):
    """Returns subset of sequence of frames with only face in each frame
    
    :param frames: sequence of frames as numpy array
    :param out_shape: tuple of height and width to which the face region of the frame will be reshaped
    :return: boolean indicating whether face was found in the frames and face regions of the frames
        or None if no face was found
    """
    output = np.zeros((FRAMES_IN_SEQ, out_shape[0], out_shape[1], 3), dtype=np.uint8)
    for i in range(frames.shape[0]):
        coords = face_recognition.face_locations(frames[i])
        if coords:  # using coordinates of the face from the first frame it was found to save time
            break
    if coords:  # if found face at any of the frames
        top, right, bottom, left = coords[0]
        data = frames[:, top:bottom, left:right, :]
        for j in range(data.shape[0]):
            frame = cv2.resize(data[j], out_shape)
            output[j, ...] = frame
        return True, output
    else:
        return False, None
    
def get_path_and_indexes(gt, train_idx, val_idx, test_idx):
    """Generate output file path and update counters
    
    :param gt: ground truth label
    :param train_idx: current train index
    :param val_idx: current validation index
    :param test_idx: current test index
    :return: generated path and updated counters in the same order
    """
    r = random.random()
    if r < 0.9:
        out_path = "train/{}_{}.npz".format(train_idx, gt)
        train_idx += 1
    elif 0.9 <= r < 0.95:
        out_path = "val/{}_{}.npz".format(val_idx, gt)
        val_idx += 1
    else:
        out_path = "test/{}_{}.npz".format(test_idx, gt)
        test_idx += 1
    return out_path, train_idx, val_idx, test_idx