import cv2
import numpy as np
import os

# Đường dẫn thư mục input và output
input_folder = 'input/'  # Thư mục chứa ảnh đầu vào
output_folder = 'output/'  # Thư mục chứa ảnh đầu ra

# Đảm bảo thư mục output tồn tại
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


# Hàm thực hiện các thao tác tăng cường ảnh
def process_image(image, operations, output_name):
    result_images = []

    if 'negative' in operations:
        negative_image = 255 - image
        cv2.imwrite(os.path.join(output_folder, f'negative_{output_name}.jpg'), negative_image)
        result_images.append(negative_image)

    if 'contrast' in operations:
        min_val, max_val = np.min(image), np.max(image)
        contrast_stretched = ((image - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(output_folder, f'contrast_{output_name}.jpg'), contrast_stretched)
        result_images.append(contrast_stretched)

    if 'log' in operations:
        log_transformed = (np.log1p(image) / np.log1p(np.max(image)) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(output_folder, f'log_{output_name}.jpg'), log_transformed)
        result_images.append(log_transformed)

    if 'histogram' in operations:
        equalized_image = cv2.equalizeHist(image)
        cv2.imwrite(os.path.join(output_folder, f'histogram_{output_name}.jpg'), equalized_image)
        result_images.append(equalized_image)


# Đọc và xử lý ảnh phong cảnh
anh_phongcanh = cv2.imread(os.path.join(input_folder, 'anh_phongcanh.jpg'), cv2.IMREAD_GRAYSCALE)
process_image(anh_phongcanh, ['contrast', 'log', 'histogram'], 'anh_phongcanh')

# Đọc và xử lý ảnh người
anh_nguoi = cv2.imread(os.path.join(input_folder, 'anh_nguoi.jpg'), cv2.IMREAD_GRAYSCALE)
process_image(anh_nguoi, ['contrast', 'log'], 'anh_nguoi')

# Đọc và xử lý ảnh y tế
anh_yte = cv2.imread(os.path.join(input_folder, 'anh_yte.jpg'), cv2.IMREAD_GRAYSCALE)
process_image(anh_yte, ['negative', 'histogram'], 'anh_yte')
