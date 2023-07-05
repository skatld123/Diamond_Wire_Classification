import cv2
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
import matplotlib.pyplot as plt
import os
import shutil

list_image = os.listdir('/root/dataset/whole/original/')
# mode 0:orginal, 1:top, 2:top&bottom, 3:center, 4:plus
mode = 2
ssim_dic = {}
psnr_dic = {}
mse_dic = {}
pixel_similarity_dic = {}

# PSNR 비교
def calculate_psnr(image1, image2):
    mse = np.mean((image1 - image2) ** 2)  # Calculate mean squared error (MSE)
    max_pixel = 255.0  # Assuming pixel values range from 0 to 255

    if mse == 0:  # If MSE is zero, the images are identical
        return float('inf')

    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))  # Calculate PSNR
    return psnr

def make_sort_list(mse_dic, psnr_dic, ssim_dic, pixel_similarity_dic, mode) :
    mse_dic = sorted(mse_dic.items(), key=lambda x: x[1])
    psnr_dic = sorted(psnr_dic.items(), key=lambda x: x[1])
    ssim_dic = sorted(ssim_dic.items(), key=lambda x: x[1])
    pixel_similarity_dic = sorted(pixel_similarity_dic.items(), key=lambda x: x[1])
    if mode == 0 : mode = "origin"
    elif mode == 1 : mode = "top"
    elif mode == 2 : mode = "top&bottom"
    elif mode == 3 : mode = "middle"
    # 파일 일렬로 저장~
    
    with open('split_reigion/sort_mse_'+ mode +'.txt', 'w') as f:
        for idx, item in enumerate(mse_dic):
            f.write(f"{item}\n")
            # 파일 복사
            print(mode)
            if mode == 'top&bottom' :
                name = os.path.basename(item[0])
                shutil.copyfile(item[0], '/root/split_reigion/mse_top_bottom/{}_{}'.format(idx, name))
    with open('split_reigion/sort_psnr_'+ mode +'.txt', 'w') as f:
        for idx, item in enumerate(psnr_dic):
            f.write(f"{item}\n")
    with open('split_reigion/sort_ssim_'+ mode +'.txt', 'w') as f:
        for item in ssim_dic:
            f.write(f"{item}\n")
    with open('split_reigion/sort_pixel_similarity_'+ mode +'.txt', 'w') as f:
        for item in pixel_similarity_dic:
            f.write(f"{item}\n")


def cal_similar(edges, lines_image, mode) :
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
    
    if mode == 0 : 
        # SSNR 비교 
        score, diff = compare_ssim(edges, lines_image, full=True)
        diff = (diff * 255).astype('uint8')
        cv2.imwrite('split_reigion/diff_all.jpg', diff)
        ssim = score

        # MSE 비교 
        diff = np.square(edges.astype("float") - lines_image.astype("float"))
        mse = diff.mean() / np.max(diff) # 평균 제곱 오차 계산
        mse = mse
        
        # 픽셀 단위 유사성 비교
        height, width = edges.shape
        pixel_similarity = np.sum(edges == lines_image) / (height * width)

        # Calculate PSNR
        psnr = calculate_psnr(edges, lines_image)

    elif mode == 1 : 
        # SSNR 비교 
        score_top, diff_top = compare_ssim(top_region, line_top_region, full=True)
        diff_top = (diff_top * 255).astype('uint8')
        cv2.imwrite('split_reigion/diff_top.jpg', diff_top)
        ssim = score_top

        # MSE 비교 
        diff_top = np.square(top_region.astype("float") - line_top_region.astype("float"))
        mse_top = diff_top.mean() / np.max(diff_top) # 평균 제곱 오차 계산
        mse = mse_top
        
        # 픽셀 단위 유사성 비교
        height, width = top_region.shape
        pixel_similarity = np.sum(top_region == line_top_region) / (height * width)

        # Calculate PSNR
        psnr = calculate_psnr(top_region, line_top_region)
    
    elif mode == 2 :
        # SSNR 비교 
        score_top, diff_top = compare_ssim(top_region, line_top_region, full=True)
        score_bottom, diff_bottom = compare_ssim(bottom_region, line_bottom_region, full=True)
        diff_top = (diff_top * 255).astype('uint8')
        diff_bottom = (diff_bottom * 255).astype('uint8')
        ssim = (score_top + score_bottom)/2

        cv2.imwrite('split_reigion/diff_top.jpg', diff_top)
        cv2.imwrite('split_reigion/diff_bottom.jpg', diff_bottom)

        # MSE 비교 
        diff_top = np.square(top_region.astype("float") - line_top_region.astype("float"))
        diff_bottom = np.square(bottom_region.astype("float") - line_bottom_region.astype("float"))
        mse_top = diff_top.mean() / np.max(diff_top) # 평균 제곱 오차 계산
        mse_bottom = diff_bottom.mean() / np.max(diff_bottom) # 평균 제곱 오차 계산
        mse = (mse_top  + mse_bottom)/2 

        # 픽셀 단위 유사성 비교
        height, width = top_region.shape
        pixel_similarity_top = np.sum(top_region == line_top_region) / (height * width)
        pixel_similarity_bottom = np.sum(bottom_region == line_bottom_region) / (height * width)
        pixel_similarity = (pixel_similarity_top + pixel_similarity_bottom)/2

        # Calculate PSNR
        psnr_t = calculate_psnr(top_region, line_top_region)
        psnr_b = calculate_psnr(bottom_region, line_bottom_region)
        psnr = (psnr_t + psnr_b)/2
        
    elif mode == 3 :
        # SSNR 비교 
        score_middle, diff_middle = compare_ssim(middle_region, line_middle_region, full=True)
        diff_middle = (diff_middle * 255).astype('uint8')
        ssim = score_middle
        cv2.imwrite('split_reigion/diff_middle.jpg', diff_middle)

        # MSE 비교 
        diff_middle = np.square(middle_region.astype("float") - line_middle_region.astype("float"))
        mse_middle = diff_middle.mean() / np.max(diff_middle) # 평균 제곱 오차 계산
        mse = mse_middle

        # 픽셀 단위 유사성 비교
        height, width = top_region.shape
        pixel_similarity = np.sum(middle_region == line_middle_region) / (height * width)
        
        # Calculate PSNR
        psnr = calculate_psnr(middle_region, line_middle_region)
        
    # 결과 출력
    print(f'MSE: {mse:.6f}')
    print(f'SSIM: {ssim:.6f}')
    print(f'pixel_similarity : {pixel_similarity}')
    print("PSNR:", (psnr))
    
    return mse, ssim, pixel_similarity, psnr
    
for index, image_path in enumerate(list_image) :
    if "high" in image_path :
        print("Input high image")
    elif "low" in image_path :
        print("Input low image")
    else :
        print("Input med image")
    image_path = os.path.join('/root/dataset/whole/original/',image_path)
    image = cv2.imread(image_path)
    # 이미지를 그레이스케일로 변환
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 엣지 검출 (Canny 알고리즘 사용)
    edges = cv2.Canny(gray_image, 50, 150)  # threshold1 = 50, threshold2 = 150

    # 허프 변환을 사용하여 직선 검출
    # lines = cv2.HoughLines(edges, 1, np.pi / 180, 80)  # rho = 1, theta = np.pi / 180, threshold = 100
    # lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=80, theta=np.pi/2, srn=0, stn=0)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 70)
    cv2.imwrite('split_reigion/sort/Canny_output_%03d.jpg' % (index), edges)

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
        if np.abs(b) > 0.9 :
        # if np.abs(b) == 1.0:
            if top_line is None or rho < top_line[0]:
                top_line = (rho, theta)
            if bottom_line is None or rho > bottom_line[0]:
                bottom_line = (rho, theta)
        
        
    # 위와 아래 선분 추출
    if top_line is not None:
        if index == 40 : 
            print(top_line)
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

    if bottom_line is not None :
        if index == 39 : 
            print(bottom_line)
            print(lines)
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
    cv2.imwrite('split_reigion/sort/HoughLines_output_%03d.jpg'%(index), lines_image)
    lines_image = cv2.cvtColor(lines_image, cv2.COLOR_BGR2GRAY)
    
    rest = edges - lines_image
    cv2.imwrite("split_reigion/sort/rest_output_%03d.jpg"%(index), rest)

    mse, psnr, ssim, pixel_similarity = cal_similar(lines_image=lines_image, edges=edges, mode=mode)
    
    mse_dic[image_path] = mse
    psnr_dic[image_path] = psnr
    ssim_dic[image_path] = ssim
    pixel_similarity_dic[image_path] = pixel_similarity
    
make_sort_list(mse_dic,psnr_dic,ssim_dic,pixel_similarity_dic,mode)


    # cv2.imwrite('split_reigion/top.jpg', top_region)
    # cv2.imwrite('split_reigion/middle.jpg', middle_region)
    # cv2.imwrite('split_reigion/bottom.jpg', bottom_region)

    # cv2.imwrite('split_reigion/top_line.jpg', line_top_region)
    # cv2.imwrite('split_reigion/middle_line.jpg', line_middle_region)
    # cv2.imwrite('split_reigion/bottom_line.jpg', line_bottom_region)

