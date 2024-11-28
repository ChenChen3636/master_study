import os
import re
from datetime import datetime

# 指定資料夾路徑
folder_path = input("請輸入資料夾路徑：")

# 確認資料夾是否存在
if not os.path.isdir(folder_path):
    print("指定的資料夾不存在，請重新確認！")
else:
    # 初始化日期列表
    dates = []

    # 遍歷資料夾中的檔案
    for filename in os.listdir(folder_path):
        # 使用正則表達式提取檔名中的日期部分
        match = re.search(r'_([0-9]{8})_', filename)
        if match:
            # 將提取的日期轉換為 datetime 格式
            dates.append(datetime.strptime(match.group(1), "%Y%m%d"))

    # 檢查是否有提取到日期
    if dates:
        # 找出最早和最晚日期
        earliest_date = min(dates)
        latest_date = max(dates)
        print("最早日期:", earliest_date.strftime("%Y-%m-%d"))
        print("最晚日期:", latest_date.strftime("%Y-%m-%d"))
    else:
        print("資料夾中沒有找到符合格式的日期！")
