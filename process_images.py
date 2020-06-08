import cv2 as cv

from utils import process_new

if __name__ == '__main__':
    index = 0
    # read lables
    with open('data/JNIST/label_test.csv', 'r') as f:
        lines = f.readlines()

    for i in range(len(lines)):
        split_label = lines[i][:-1].split(',')
        source_filename = 'data/JNIST/test/%05d.jpg' % i
        label_count = len(split_label) - 1
        ret_images = process_new(source_filename)

        if label_count != len(ret_images):
            index += 1
            print('%d: %s' % (index, source_filename))
        else:
            for j in range(label_count):
                label = split_label[j+1]
                filename = 'data/JNIST/test_split_images/%s/%05d.jpg' % (label, i)
                cv.imwrite(filename, ret_images[j])
