import cv2
import numpy as np

# 讀取灰階影像
image_path = 'ultrasound/8/a.png'  # 替換為你的灰階超音波影像路徑
ori_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
x,y = 167,59 #剪裁座標
width = 690 
height = 551

gray = ori_img[y:y+height, x:x+width]

# 影像銳化
kernel_sharpening = np.array([[-1, -1, -1],
                              [-1, 9, -1],
                              [-1, -1, -1]])
sharpened = cv2.filter2D(gray, -1, kernel_sharpening)

# 邊緣檢測
edges = cv2.Canny(sharpened, threshold1=30, threshold2=100)

# 儲存銳化後的影像
cv2.imwrite('ultrasound/8/sharpened_a.png', sharpened)

# 顯示結果
cv2.imshow('Original', gray)
cv2.imshow('Sharpened', sharpened)
cv2.imshow('Edges', edges)

# 等待按鍵後關閉窗口
cv2.waitKey(0)
cv2.destroyAllWindows()
