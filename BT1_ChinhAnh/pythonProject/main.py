import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Đường dẫn thư mục input và output
input_folder = 'input/'  # Thư mục chứa ảnh đầu vào
output_folder = 'output/'  # Thư mục chứa ảnh đầu ra

# Đảm bảo thư mục output tồn tại
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


# Hàm hiển thị ảnh
def show_images(original_image, processed_images, titles):
    plt.figure(figsize=(12, 6))

    # Hiển thị ảnh gốc đầu tiên (màu RGB)
    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))  # Chuyển ảnh gốc từ BGR sang RGB để hiển thị đúng màu
    plt.title('Original Image')
    plt.axis('off')

    # Hiển thị các ảnh đã qua xử lý
    for i in range(len(processed_images)):
        plt.subplot(2, 3, i + 2)
        if len(processed_images[i].shape) == 3:  # Nếu là ảnh màu
            plt.imshow(cv2.cvtColor(processed_images[i], cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(processed_images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')

    plt.tight_layout()
    plt.show()


# Hàm thực hiện các thao tác tăng cường ảnh
def process_image(image, operations, output_name):
    result_images = []
    titles = []

    if 'negative' in operations:
        negative_image = 255 - cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(os.path.join(output_folder, f'negative_{output_name}.jpg'), negative_image)
        result_images.append(negative_image)
        titles.append(f'Negative {output_name}')

    if 'contrast' in operations:
        # Tăng cường tương phản trên ảnh màu
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)  # Chuyển ảnh sang không gian màu LAB
        l, a, b = cv2.split(lab)  # Tách các kênh L, A, B
        l = cv2.equalizeHist(l)  # Cân bằng histogram cho kênh L
        contrast_stretched = cv2.merge((l, a, b))  # Ghép các kênh lại
        contrast_image = cv2.cvtColor(contrast_stretched, cv2.COLOR_LAB2BGR)  # Chuyển lại ảnh sang không gian màu BGR
        cv2.imwrite(os.path.join(output_folder, f'contrast_{output_name}.jpg'), contrast_image)
        result_images.append(contrast_image)
        titles.append(f'Contrast {output_name}')

    if 'log' in operations:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        log_transformed = (np.log1p(gray_image) / np.log1p(np.max(gray_image)) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(output_folder, f'log_{output_name}.jpg'), log_transformed)
        result_images.append(log_transformed)
        titles.append(f'Log {output_name}')

    if 'histogram' in operations:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        equalized_image = cv2.equalizeHist(gray_image)
        cv2.imwrite(os.path.join(output_folder, f'histogram_{output_name}.jpg'), equalized_image)
        result_images.append(equalized_image)
        titles.append(f'Histogram {output_name}')

    # Hiển thị ảnh gốc cùng với ảnh đã qua xử lý
    show_images(image, result_images, titles)


# Đọc và xử lý ảnh phong cảnh
anh_phongcanh = cv2.imread(os.path.join(input_folder, 'anh_phongcanh.jpg'))  # Đọc ảnh màu
process_image(anh_phongcanh, ['contrast', 'log', 'histogram'], 'anh_phongcanh')

# Đọc và xử lý ảnh người
anh_nguoi = cv2.imread(os.path.join(input_folder, 'anh_nguoi.jpg'))  # Đọc ảnh màu
process_image(anh_nguoi, ['contrast', 'log'], 'anh_nguoi')

# Đọc và xử lý ảnh y tế
anh_yte = cv2.imread(os.path.join(input_folder, 'anh_yte.jpg'))  # Đọc ảnh màu
process_image(anh_yte, ['negative', 'histogram'], 'anh_yte')
