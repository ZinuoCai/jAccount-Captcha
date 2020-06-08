import cv2 as cv
import numpy as np

# label source_images size
image_count = 250
# output size
output_size = 32

# read lables
with open('data/MyData/labels.csv', 'r') as f:
    lines = f.readlines()

for i in range(image_count):
    split_label = lines[i][:-1].split(',')
    source_filename = 'data/source_images/%04d.png' % i

    color_img = cv.imread(source_filename)
    gray_img = cv.cvtColor(color_img, cv.COLOR_BGR2GRAY)
    height, width = gray_img.shape

    blurred = cv.medianBlur(gray_img, 3)
    kernel = np.ones((3, 3), np.uint8)
    erosion = cv.erode(blurred, kernel, iterations=1)

    sum_by_col = np.sum(erosion != 255, axis=1)
    sum_by_row = np.sum(erosion != 255, axis=0)

    top, down, left, right = 0, height - 1, 0, width - 1

    while top < height:
        if sum_by_col[top] != 0:
            break
        else:
            top += 1

    while down > 0:
        if sum_by_col[top] != 0:
            break
        else:
            down -= 1

    while left < width:
        if sum_by_row[left] != 0:
            break
        else:
            left += 1

    while right > 0:
        if sum_by_row[right] != 0:
            break
        else:
            right -= 1

    label_count = len(split_label)
    single_width = (right - left) // (label_count - 1)
    vertical_padding = (max(output_size, height) - (down - top)) // 2
    horizontal_padding = (max(output_size, width) - (right - left)) // 2

    for j in range(1, label_count):
        label = split_label[j]
        filename = 'data/gray_split_images/%s/%04d.png' % (label, i)
        # single_img = color_img[top:down, left + single_width * (j - 1):left + single_width * j, :]
        single_img = erosion[top:down, left + single_width * (j - 1):left + single_width * j]
        single_img = cv.copyMakeBorder(single_img, vertical_padding, vertical_padding,
                                       horizontal_padding, horizontal_padding,
                                       cv.BORDER_CONSTANT, value=[255, 255, 255])
        single_img = cv.resize(single_img, (output_size, output_size))
        cv.imwrite(filename, single_img)
