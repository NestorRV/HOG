import cv2
import numpy as np
import os
import re
import math
import pickle
import time
from sklearn import svm
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random

random.seed(0)


def atoi(text):
    return int(text) if text.isdigit() else text


def histogram2(angles, magnitudes):
    # Compute the histogram
    bins = [0, 20, 40, 60, 80, 100, 120, 140, 160]
    h = np.zeros(len(bins))

    for i in range(np.shape(angles)[0]):
        for j in range(np.shape(angles)[1]):
            my_list = [e for e, a in zip(range(len(bins)), bins) if a < angles[i][j]]

            index = 0
            if my_list:
                index = (my_list[-1] + 1) % len(bins)

            difference = np.abs(bins[index] - (angles[i][j] % 160))
            if difference == 160:
                proportion = 1
            else:
                proportion = difference / 20

            values = magnitudes[i][j] * proportion, magnitudes[i][j] * (1 - proportion)

            if angles[i][j] < 160:
                h[index - 1] += values[0]
                h[index] += values[1]
            else:
                h[index - 1] += values[1]
                h[index] += values[0]

    return h


def histogram(angles, magnitudes):
    # [0, 20, 40, 60, 80, 100, 120, 140, 160]
    h = np.zeros(10, dtype=np.float32)

    for i in range(angles.shape[0]):
        for j in range(angles.shape[1]):
            angles[i, j] = 160
            index_1 = int(angles[i, j] // 20)
            index_2 = int(angles[i, j] // 20 + 1)

            proportion = (index_2 * 20 - angles[i, j]) / 20

            value_1 = proportion * magnitudes[i, j]
            value_2 = (1 - proportion) * magnitudes[i, j]

            h[index_1] += value_1
            h[index_2] += value_2

    h[0] += h[-1]
    return h[0:9]


def make_cells(angles, magnitudes, cell_size):
    cells = []
    for i in range(0, np.shape(angles)[0], cell_size):
        row = []
        for j in range(0, np.shape(angles)[1], cell_size):
            row.append(np.array(
                histogram(angles[i:i + cell_size, j:j + cell_size], magnitudes[i:i + cell_size, j:j + cell_size]),
                dtype=np.float32))
        cells.append(row)

    return np.array(cells, dtype=np.float32)


def make_blocks(block_size, cells):
    before = int(block_size / 2)
    after = int(block_size / 2)

    if block_size % 2 != 0:
        after = after + 1

    first_stop = before
    second_stop = before

    if np.shape(cells)[0] % block_size == 0:
        first_stop = first_stop - 1

    if np.shape(cells)[1] % block_size == 0:
        second_stop = second_stop - 1

    blocks = []
    for i in range(int(block_size / 2.0), np.shape(cells)[0] - first_stop):
        for j in range(int(block_size / 2.0), np.shape(cells)[1] - second_stop):
            blocks.append(np.array(cells[i - before:i + after, j - before:j + after].flatten()))

    return blocks


def normalize_L1(block, threshold):
    norm = np.sum(block) + threshold
    if norm != 0:
        return block / norm
    else:
        return block


def normalize_L1_sqrt(block, threshold):
    norm = np.sum(block) + threshold
    if norm != 0:
        return np.sqrt(block / norm)
    else:
        return block


def normalize_L2(block, threshold):
    norm = np.sqrt(np.sum(block * block) + threshold * threshold)
    if norm != 0:
        return block / norm
    else:
        return block


def normalize_L2_Hys(block, threshold):
    norm = np.sqrt(np.sum(block * block) + threshold * threshold)

    if norm != 0:
        block_aux = block / norm
        block_aux[block_aux > 0.2] = 0.2
        norm = np.sqrt(np.sum(block_aux * block_aux) + threshold * threshold)
        if norm != 0:
            return block_aux / norm
        else:
            return block_aux
    else:
        return block


def normalize(block, type_norm, threshold=0):
    if type_norm == 0:
        return normalize_L2(block, threshold)
    elif type_norm == 1:
        return normalize_L2_Hys(block, threshold)
    elif type_norm == 2:
        return normalize_L1(block, threshold)
    elif type_norm == 3:
        return normalize_L1_sqrt(block, threshold)


def hog(img, cell_size=6, block_size=3, type_norm=0, all_norms=False):
    # Gamma correction : gamma = 0.2
    img = np.power(img, 0.2, dtype=np.float32)

    # Calculate gradient
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)

    # Gradient magnitude and direction (in degrees)
    magnitudes, angles = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    max_norm = np.argmax(magnitudes, axis=2)

    # For each pixel, we store the angle with the biggest magnitude
    m, n = np.shape(max_norm)
    I, J = np.ogrid[:m, :n]
    max_angles = angles[I, J, max_norm]
    max_magnitudes = magnitudes[I, J, max_norm]

    # Convert angles to be in range 0-180
    max_angles = max_angles % 180

    # Obtain the histogram for each region
    cells = make_cells(max_angles, max_magnitudes, cell_size)

    # Append the cells into blocks: 180 descriptors with 81 elements
    blocks = make_blocks(block_size, cells)

    # We need to apply a Gaussian kernel
    # kernel = [-1, 0, 1]
    # sigma = 0.5 * block_size
    # kernel = np.array([math.exp(-0.5 * (x ** 2 / sigma ** 2)) for x in kernel])
    # multiplier = np.append(np.append(np.ones(36) * kernel[0], np.ones(9) * kernel[1]), np.ones(36) * kernel[2])
    # blocks = [b * multiplier for b in blocks]

    # Now we need to normalize
    if all_norms:
        blocks_norm0 = np.concatenate([normalize(b, 0) for b in blocks])
        blocks_norm1 = np.concatenate([normalize(b, 1) for b in blocks])
        blocks_norm2 = np.concatenate([normalize(b, 2) for b in blocks])
        blocks_norm3 = np.concatenate([normalize(b, 3) for b in blocks])

        return blocks_norm0, blocks_norm1, blocks_norm2, blocks_norm3
    else:
        blocks = np.concatenate([normalize(b, type_norm) for b in blocks])
        # Concatenate all the blocks and this is the final descriptor
        return blocks


def plot_classifier(kernel, data_train, labels_train, data_test, labels_test, title, file):
    clf = svm.SVC(kernel=kernel)

    start_time = time.time()
    clf.fit(data_train, labels_train)
    elapsed_time = time.time() - start_time
    print("{} seconds".format(elapsed_time))

    pickle.dump(clf, open("./pickle/" + file + ".p", "wb"))

    predict = clf.predict(data_test)

    fpr, tpr, thresholds = roc_curve(labels_test, predict, pos_label=1)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    patch = mpatches.Patch(color='red', label='ROC curve. area = {}, error = {}'.format(np.round(roc_auc, 4),
                                                                                        np.round(1 - roc_auc, 4)))
    plt.legend(handles=[patch], loc='lower right')
    plt.plot(fpr, tpr, color='red')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.savefig(file, dpi=700)
    # plt.show()
    plt.clf()


def recognition(img, classifier, type_norm, step=4):
    positive_regions = []
    pyramid = [img]
    new_level = img

    while np.shape(new_level)[0] >= 128 and np.shape(new_level)[1] >= 64:
        # ksize = (size = 3 * 2 * sigma = 1 + 1, size = 3 * 2 * sigma = 1 + 1)
        new_level = cv2.GaussianBlur(src=new_level, ksize=(7, 7), sigmaX=1)
        # 0.8333333 is 1 / 1.2
        new_level = cv2.resize(new_level, dsize=(0, 0), fx=0.8333333, fy=0.8333333)
        pyramid.append(new_level)

    for level, img_pyramid in zip(range(len(pyramid)), pyramid):
        for i in range(0, np.shape(img_pyramid)[0] - 128, step):
            for j in range(0, np.shape(img_pyramid)[1] - 64, step * 2):
                sub_img = cv2.copyMakeBorder(img_pyramid[i:i + 128, j:j + 64], 2, 2, 1, 1, cv2.BORDER_REFLECT)
                h = hog(sub_img, type_norm=type_norm)
                prediction = classifier.predict(h.reshape(-1, h.shape[0]))
                if prediction[0] != 0.0:
                    positive_regions.append([level, j, i])

    return positive_regions


def main():
    ###########################################################################
    ############################### TRAIN #####################################
    ###########################################################################

    print("TRAIN POSITIVE")
    # len(pos_images_train) = 2416
    pos_images_train = []
    for dirName, subdirList, fileList in os.walk("./data/train/pos"):
        for fname in sorted(fileList, key=lambda x: [atoi(c) for c in re.split('(\d+)', x)]):
            if fname != ".DS_Store":
                pos_images_train.append(cv2.imread("./data/train/pos/" + fname))

    # top, bottom, left, right
    # add the necessary to make the correct partition in cells
    # use only the centered window
    pos_images_train = [cv2.copyMakeBorder(i[16:144, 16:80], 2, 2, 1, 1, cv2.BORDER_REFLECT) for i in pos_images_train]
    start_time = time.time()
    hog_positives_train = np.array([hog(img, all_norms=True) for img in pos_images_train])
    elapsed_time = time.time() - start_time
    print("TRAIN POSITIVE: {} seconds".format(elapsed_time))

    hog_positives_train_norm0 = hog_positives_train[:, 0, :]
    hog_positives_train_norm1 = hog_positives_train[:, 1, :]
    hog_positives_train_norm2 = hog_positives_train[:, 2, :]
    hog_positives_train_norm3 = hog_positives_train[:, 3, :]

    pickle.dump(hog_positives_train_norm0, open("./pickle/hog_positives_train_norm0.p", "wb"))
    pickle.dump(hog_positives_train_norm1, open("./pickle/hog_positives_train_norm1.p", "wb"))
    pickle.dump(hog_positives_train_norm2, open("./pickle/hog_positives_train_norm2.p", "wb"))
    pickle.dump(hog_positives_train_norm3, open("./pickle/hog_positives_train_norm3.p", "wb"))

    print("TRAIN NEGATIVE")
    # len(neg_files_train) = 1218
    neg_files_train = []
    for dirName, subdirList, fileList in os.walk("./data/train/neg"):
        for fname in sorted(fileList, key=lambda x: [atoi(c) for c in re.split('(\d+)', x)]):
            if fname != ".DS_Store":
                neg_files_train.append(cv2.imread("./data/train/neg/" + fname))

    elapsed_time = 0
    hog_negatives_train = []
    for img in neg_files_train:
        for i in range(10):
            row = random.randint(0, np.shape(img)[0] - 128)
            col = random.randint(0, np.shape(img)[1] - 64)
            sub_img = img[row:row + 128, col:col + 64]
            sub_img = cv2.copyMakeBorder(sub_img, 2, 2, 1, 1, cv2.BORDER_REFLECT)
            start_time = time.time()
            result = np.array(hog(sub_img, all_norms=True))
            elapsed_time += time.time() - start_time
            hog_negatives_train.append(result)

    print("TRAIN NEGATIVE: {} seconds".format(elapsed_time))

    hog_negatives_train_norm0 = np.array(hog_negatives_train)[:, 0, :]
    hog_negatives_train_norm1 = np.array(hog_negatives_train)[:, 1, :]
    hog_negatives_train_norm2 = np.array(hog_negatives_train)[:, 2, :]
    hog_negatives_train_norm3 = np.array(hog_negatives_train)[:, 3, :]

    pickle.dump(hog_negatives_train_norm0, open("./pickle/hog_negatives_train_norm0.p", "wb"))
    pickle.dump(hog_negatives_train_norm1, open("./pickle/hog_negatives_train_norm1.p", "wb"))
    pickle.dump(hog_negatives_train_norm2, open("./pickle/hog_negatives_train_norm2.p", "wb"))
    pickle.dump(hog_negatives_train_norm3, open("./pickle/hog_negatives_train_norm3.p", "wb"))

    ###########################################################################
    ############################### TEST ######################################
    ###########################################################################

    print("TEST POSITIVE")
    # len(pos_images_test) = 1126
    pos_images_test = []
    for dirName, subdirList, fileList in os.walk("./data/test/pos"):
        for fname in sorted(fileList, key=lambda x: [atoi(c) for c in re.split('(\d+)', x)]):
            if fname != ".DS_Store":
                pos_images_test.append(cv2.imread("./data/test/pos/" + fname))

    # top, bottom, left, right
    # add the necessary to make the correct partition in cells

    pos_images_test = [cv2.copyMakeBorder(i[3:131, 3:67], 2, 2, 1, 1, cv2.BORDER_REFLECT) for i in pos_images_test]
    start_time = time.time()
    hog_positives_test = np.array([hog(img, all_norms=True) for img in pos_images_test])
    elapsed_time = time.time() - start_time
    print("TEST POSITIVE: {} seconds".format(elapsed_time))

    hog_positives_test_norm0 = hog_positives_test[:, 0, :]
    hog_positives_test_norm1 = hog_positives_test[:, 1, :]
    hog_positives_test_norm2 = hog_positives_test[:, 2, :]
    hog_positives_test_norm3 = hog_positives_test[:, 3, :]

    pickle.dump(hog_positives_test_norm0, open("./pickle/hog_positives_test_norm0.p", "wb"))
    pickle.dump(hog_positives_test_norm1, open("./pickle/hog_positives_test_norm1.p", "wb"))
    pickle.dump(hog_positives_test_norm2, open("./pickle/hog_positives_test_norm2.p", "wb"))
    pickle.dump(hog_positives_test_norm3, open("./pickle/hog_positives_test_norm3.p", "wb"))

    print("TEST NEGATIVE")
    # len(neg_images_test) = 453
    neg_files_test = []
    for dirName, subdirList, fileList in os.walk("./data/test/neg"):
        for fname in sorted(fileList, key=lambda x: [atoi(c) for c in re.split('(\d+)', x)]):
            if fname != ".DS_Store":
                neg_files_test.append(cv2.imread("./data/test/neg/" + fname))

    elapsed_time = 0
    hog_negatives_test = []
    for img in neg_files_test:
        for i in range(10):
            row = random.randint(0, np.shape(img)[0] - 128)
            col = random.randint(0, np.shape(img)[1] - 64)
            sub_img = img[row:row + 128, col:col + 64]
            sub_img = cv2.copyMakeBorder(sub_img, 2, 2, 1, 1, cv2.BORDER_REFLECT)
            start_time = time.time()
            result = np.array(hog(sub_img, all_norms=True))
            elapsed_time += time.time() - start_time
            hog_negatives_test.append(result)

    print("TEST NEGATIVE: {} seconds".format(elapsed_time))

    hog_negatives_test_norm0 = np.array(hog_negatives_test)[:, 0, :]
    hog_negatives_test_norm1 = np.array(hog_negatives_test)[:, 1, :]
    hog_negatives_test_norm2 = np.array(hog_negatives_test)[:, 2, :]
    hog_negatives_test_norm3 = np.array(hog_negatives_test)[:, 3, :]

    pickle.dump(hog_negatives_test_norm0, open("./pickle/hog_negatives_test_norm0.p", "wb"))
    pickle.dump(hog_negatives_test_norm1, open("./pickle/hog_negatives_test_norm1.p", "wb"))
    pickle.dump(hog_negatives_test_norm2, open("./pickle/hog_negatives_test_norm2.p", "wb"))
    pickle.dump(hog_negatives_test_norm3, open("./pickle/hog_negatives_test_norm3.p", "wb"))

    ###########################################################################
    ####################### PREPARE THE DATA (TRAIN) ##########################
    ###########################################################################

    hog_positives_train_norm0 = pickle.load(open("./pickle/hog_positives_train_norm0.p", "rb"))
    hog_positives_train_norm1 = pickle.load(open("./pickle/hog_positives_train_norm1.p", "rb"))
    hog_positives_train_norm2 = pickle.load(open("./pickle/hog_positives_train_norm2.p", "rb"))
    hog_positives_train_norm3 = pickle.load(open("./pickle/hog_positives_train_norm3.p", "rb"))

    hog_negatives_train_norm0 = pickle.load(open("./pickle/hog_negatives_train_norm0.p", "rb"))
    hog_negatives_train_norm1 = pickle.load(open("./pickle/hog_negatives_train_norm1.p", "rb"))
    hog_negatives_train_norm2 = pickle.load(open("./pickle/hog_negatives_train_norm2.p", "rb"))
    hog_negatives_train_norm3 = pickle.load(open("./pickle/hog_negatives_train_norm3.p", "rb"))

    pos_labels_train = np.ones(len(hog_positives_train_norm0))
    neg_labels_train = np.zeros(len(hog_negatives_train_norm0))
    labels_train = np.append(pos_labels_train, neg_labels_train)

    data_train_norm0 = np.append(hog_positives_train_norm0, hog_negatives_train_norm0, axis=0)
    data_train_norm1 = np.append(hog_positives_train_norm1, hog_negatives_train_norm1, axis=0)
    data_train_norm2 = np.append(hog_positives_train_norm2, hog_negatives_train_norm2, axis=0)
    data_train_norm3 = np.append(hog_positives_train_norm3, hog_negatives_train_norm3, axis=0)

    ###########################################################################
    ####################### PREPARE THE DATA (TEST) ###########################
    ###########################################################################

    hog_positives_test_norm0 = pickle.load(open("./pickle/hog_positives_test_norm0.p", "rb"))
    hog_positives_test_norm1 = pickle.load(open("./pickle/hog_positives_test_norm1.p", "rb"))
    hog_positives_test_norm2 = pickle.load(open("./pickle/hog_positives_test_norm2.p", "rb"))
    hog_positives_test_norm3 = pickle.load(open("./pickle/hog_positives_test_norm3.p", "rb"))

    hog_negatives_test_norm0 = pickle.load(open("./pickle/hog_negatives_test_norm0.p", "rb"))
    hog_negatives_test_norm1 = pickle.load(open("./pickle/hog_negatives_test_norm1.p", "rb"))
    hog_negatives_test_norm2 = pickle.load(open("./pickle/hog_negatives_test_norm2.p", "rb"))
    hog_negatives_test_norm3 = pickle.load(open("./pickle/hog_negatives_test_norm3.p", "rb"))

    pos_labels_test = np.ones(len(hog_positives_test_norm0))
    neg_labels_test = np.zeros(len(hog_negatives_test_norm0))
    labels_test = np.append(pos_labels_test, neg_labels_test)

    data_test_norm0 = np.append(hog_positives_test_norm0, hog_negatives_test_norm0, axis=0)
    data_test_norm1 = np.append(hog_positives_test_norm1, hog_negatives_test_norm1, axis=0)
    data_test_norm2 = np.append(hog_positives_test_norm2, hog_negatives_test_norm2, axis=0)
    data_test_norm3 = np.append(hog_positives_test_norm3, hog_negatives_test_norm3, axis=0)

    ###########################################################################
    ################################# SVM #####################################
    ###########################################################################

    plot_classifier('linear', data_train_norm0, labels_train, data_test_norm0, labels_test, 'Norm: L2.',
                    "linear_svm_norm_L2")
    plot_classifier('linear', data_train_norm1, labels_train, data_test_norm1, labels_test, 'Norm: L2-Hys.',
                    "linear_svm_norm_L2_hys")
    plot_classifier('linear', data_train_norm2, labels_train, data_test_norm2, labels_test, 'Norm: L1.',
                    "linear_svm_norm_L1")
    plot_classifier('linear', data_train_norm3, labels_train, data_test_norm3, labels_test, 'Norm: sqrt(L1).',
                    "linear_svm_norm_sqrt_L1")

    ###########################################################################
    ############################# RECOGNITION #################################
    ###########################################################################

    linear_svm_norm_L2 = pickle.load(open("./pickle_gaussian/linear_svm_norm_L2.p", "rb"))

    img = cv2.imread("./full_data/Test/pos/person_120.png")
    start_time = time.time()
    positive_regions = recognition(img=img, classifier=linear_svm_norm_L2, type_norm=0)
    elapsed_time = time.time() - start_time
    print("{} seconds".format(elapsed_time))
    pickle.dump(positive_regions, open("positive_regions_test_pos_person_120.p", "wb"))

    img = cv2.imread("./full_data/Test/pos/person_198.png")
    start_time = time.time()
    positive_regions = recognition(img=img, classifier=linear_svm_norm_L2, type_norm=0)
    elapsed_time = time.time() - start_time
    print("{} seconds".format(elapsed_time))
    pickle.dump(positive_regions, open("positive_regions_test_pos_person_198.p", "wb"))

    img = cv2.imread("./full_data/Test/pos/person_138.png")
    start_time = time.time()
    positive_regions = recognition(img=img, classifier=linear_svm_norm_L2, type_norm=0)
    elapsed_time = time.time() - start_time
    print("{} seconds".format(elapsed_time))
    pickle.dump(positive_regions, open("positive_regions_test_pos_person_138.p", "wb"))

    img = cv2.imread("./full_data/Test/pos/person_120.png")
    positive_regions_test_pos_person_120 = pickle.load(open("positive_regions_test_pos_person_120.p", "rb"))
    [cv2.rectangle(img, (p[1] * int(1.2 ** p[0]), p[2] * int(1.2 ** p[0])),
                   (p[1] * int(1.2 ** p[0]) + 64 * int(1.2 ** p[0]), p[2] * int(1.2 ** p[0]) + 128 * int(1.2 ** p[0])),
                   (0, 255, 255), 1) for p in positive_regions_test_pos_person_120]
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("person_120")
    plt.savefig("./recog_person_120", dpi=700)
    # plt.show()
    plt.clf()

    img = cv2.imread("./full_data/Test/pos/person_198.png")
    positive_regions_test_pos_person_198 = pickle.load(open("positive_regions_test_pos_person_198.p", "rb"))
    [cv2.rectangle(img, (p[1] * int(1.2 ** p[0]), p[2] * int(1.2 ** p[0])),
                   (p[1] * int(1.2 ** p[0]) + 64 * int(1.2 ** p[0]), p[2] * int(1.2 ** p[0]) + 128 * int(1.2 ** p[0])),
                   (0, 255, 255), 1) for p in positive_regions_test_pos_person_198]
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("person_198")
    plt.savefig("./recog_person_198", dpi=700)
    # plt.show()
    plt.clf()

    img = cv2.imread("./full_data/Test/pos/person_138.png")
    positive_regions_test_pos_person_138 = pickle.load(open("positive_regions_test_pos_person_138.p", "rb"))
    [cv2.rectangle(img, (p[1] * int(1.2 ** p[0]), p[2] * int(1.2 ** p[0])),
                   (p[1] * int(1.2 ** p[0]) + 64 * int(1.2 ** p[0]), p[2] * int(1.2 ** p[0]) + 128 * int(1.2 ** p[0])),
                   (0, 255, 255), 1) for p in positive_regions_test_pos_person_138]
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("person_138")
    plt.savefig("./recog_person_138", dpi=700)
    # plt.show()
    plt.clf()


if __name__ == "__main__":
    main()
