import cv2
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
import matplotlib.pyplot as plt
# image_path = "/root/model_test/dataset/whole/original/low_3_20230417_135529_797.bmp"
# image_path = '/root/model_test/dataset/whole/original/medium_3_20230417_135240_453.bmp'
image_path = '/root/model_test/dataset/whole/original/high_1_20230417_140213_742.bmp'
if "high" in image_path :
    print("Input high image")
elif "low" in image_path :
    print("Input low image")
else :
    print("Input med image")

image = cv2.imread(image_path)
# 이미지를 그레이스케일로 변환
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 엣지 검출 (Canny 알고리즘 사용)
edges = cv2.Canny(gray_image, 50, 150)  # threshold1 = 50, threshold2 = 150

# 허프 변환을 사용하여 직선 검출
lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)  # rho = 1, theta = np.pi / 180, threshold = 100

# Canny 엣지 검출 결과 이미지 저장
canny_output_path = "Canny_output.jpg"
cv2.imwrite(canny_output_path, edges)

# 검출된 직선들을 순회하면서 직사각형의 위와 아래 선분 식별
top_line = None
bottom_line = None

# HoughLines 직선 검출 결과 이미지 생성
lines_image = np.zeros_like(image)
for i, line in enumerate(lines):
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    
    # 기울기가 수직에 가까운 직선 검출
    if np.abs(b) > 0.9:
        if top_line is None or rho < top_line[0]:
            top_line = (rho, theta)
        if bottom_line is None or rho > bottom_line[0]:
            bottom_line = (rho, theta)
            
# 위와 아래 선분 추출
if top_line is not None:
    rho_top, theta_top = top_line
    a_top = np.cos(theta_top)
    b_top = np.sin(theta_top)
    x0_top = a_top * rho_top
    y0_top = b_top * rho_top
    x1_top = int(x0_top + 1000 * (-b_top))
    y1_top = int(y0_top + 1000 * (a_top))
    x2_top = int(x0_top - 1000 * (-b_top))
    y2_top = int(y0_top - 1000 * (a_top))
    top_y = max(y1_top, y2_top)
    cv2.line(lines_image, (x1_top, y1_top), (x2_top, y2_top), (255, 255, 255), 1)  # 위 선분 그리기

if bottom_line is not None:
    rho_bottom, theta_bottom = bottom_line
    a_bottom = np.cos(theta_bottom)
    b_bottom = np.sin(theta_bottom)
    x0_bottom = a_bottom * rho_bottom
    y0_bottom = b_bottom * rho_bottom
    x1_bottom = int(x0_bottom + 1000 * (-b_bottom))
    y1_bottom = int(y0_bottom + 1000 * (a_bottom))
    x2_bottom = int(x0_bottom - 1000 * (-b_bottom))
    y2_bottom = int(y0_bottom - 1000 * (a_bottom))
    bottom_y = min(y1_bottom, y2_bottom)
    cv2.line(lines_image, (x1_bottom, y1_bottom), (x2_bottom, y2_bottom), (255, 255, 255), 1)  # 아래 선분 그리기

# HoughLines 직선 검출 결과 이미지 저장
hough_output_path = "HoughLines_output.jpg"
cv2.imwrite(hough_output_path, lines_image)

lines_image = cv2.cvtColor(lines_image, cv2.COLOR_BGR2GRAY)
rest = edges-lines_image
cv2.imwrite("split_reigion/rest_output.jpg", rest)
# 가운데 영역 추출
if top_y < bottom_y :
    slic_top = top_y + 5
    slic_bottom = bottom_y - 5
else : 
    slic_top = bottom_y + 5
    slic_bottom = top_y - 5
    
top_region = edges[:slic_top, :]
middle_region = edges[slic_top:slic_bottom, :]
bottom_region = edges[slic_bottom:, :]

line_top_region = lines_image[:slic_top, :]
line_middle_region = lines_image[slic_top:slic_bottom, :]
line_bottom_region = lines_image[slic_bottom:, :]

# 픽셀 단위 유사성 비교
# SSNR 비교 
score_top, diff_top = compare_ssim(top_region, line_top_region, full=True)
score_bottom, diff_bottom = compare_ssim(bottom_region, line_bottom_region, full=True)
diff_top = (diff_top * 255).astype('uint8')
diff_bottom = (diff_bottom * 255).astype('uint8')

score = (score_top + score_bottom)/ 2
print(f'SSIM: {score:.6f}')
cv2.imwrite('split_reigion/diff_top.jpg', diff_top)
cv2.imwrite('split_reigion/diff_bottom.jpg', diff_bottom)

# MSE 비교 
diff_top = np.square(top_region.astype("float") - line_top_region.astype("float"))
diff_bottom = np.square(bottom_region.astype("float") - line_bottom_region.astype("float"))

mse_top = diff_top.mean() / np.max(diff_top) # 평균 제곱 오차 계산
mse_bottom = diff_bottom.mean() / np.max(diff_bottom) # 평균 제곱 오차 계산
mse = (mse_bottom + mse_top) / 2
print(f'MSE: {mse:.6f}')

# 픽셀간 유사도 비교
# 이미지 크기 확인
height, width = top_region.shape
# 픽셀 단위 유사성 비교
pixel_similarity_top = np.sum(top_region == line_top_region) / (height * width)
pixel_similarity_bottom = np.sum(bottom_region == line_bottom_region) / (height * width)
print(f'pixel_similarity : {(pixel_similarity_bottom + pixel_similarity_top)/2}')

# PSNR 비교
def calculate_psnr(image1, image2):
    mse = np.mean((image1 - image2) ** 2)  # Calculate mean squared error (MSE)
    max_pixel = 255.0  # Assuming pixel values range from 0 to 255

    if mse == 0:  # If MSE is zero, the images are identical
        return float('inf')

    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))  # Calculate PSNR
    return psnr

# Calculate PSNR
psnr_value_t = calculate_psnr(top_region, line_top_region)
psnr_value_b = calculate_psnr(bottom_region, line_bottom_region)
print("PSNR:", (psnr_value_t + psnr_value_b) / 2)

cv2.imwrite('split_reigion/top.jpg', top_region)
cv2.imwrite('split_reigion/middle.jpg', middle_region)
cv2.imwrite('split_reigion/bottom.jpg', bottom_region)

cv2.imwrite('split_reigion/top_line.jpg', line_top_region)
cv2.imwrite('split_reigion/middle_line.jpg', line_middle_region)
cv2.imwrite('split_reigion/bottom_line.jpg', line_bottom_region)