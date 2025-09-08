import ddddocr
from PIL import Image
import io
import os
import cv2
import numpy as np

# --- 核心参数 ---
SATURATION_LOWER_BOUND = 50
VALUE_LOWER_BOUND = 40
MIN_CONTOUR_AREA = 400
IOU_THRESHOLD = 0.5  # 新增：IoU阈值，用于判断重叠

# --- 1. 初始化 ddddocr 识别器 ---
try:
    ocr = ddddocr.DdddOcr(ocr=True,
                          import_onnx_path=r"C:\Users\Deen\PycharmProjects\dddd_captcha\projects\tiktok_v9\models\tiktok_v9_1.0_51_5000_2025-09-07-17-32-09.onnx",
                          charsets_path="charsets.json")
    print("成功加载自定义OCR模型。")
except FileNotFoundError:
    print("错误：未找到自定义模型，将使用默认OCR模型。")
    ocr = ddddocr.DdddOcr()

# --- 文件路径设置 ---
images_dir = 'new_raw_images'
output_dir = 'results'
os.makedirs(output_dir, exist_ok=True)


hsv_ranges = {
    'blue': (np.array([80, 83, 223]), np.array([179, 255, 255])),
    'yellow': (np.array([11, 92, 0]), np.array([82, 255, 255])),
    'orange': (np.array([2, 97, 0]), np.array([13, 255, 255])),
    'purple': (np.array([127, 66, 0]), np.array([179, 255, 255])),
    'green': (np.array([36, 31, 14]), np.array([97, 203, 222]))
}

# --- 【新增】计算 IoU (Intersection over Union) 的函数 ---
def calculate_iou(boxA, boxB):
    """
    计算两个边界框的IoU。
    :param boxA: 第一个框，格式为 (x1, y1, x2, y2)
    :param boxB: 第二个框，格式为 (x1, y1, x2, y2)
    :return: IoU值，范围在0.0到1.0之间
    """
    # 确定相交矩形的坐标
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # 计算相交区域的面积
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # 计算两个框的面积
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    # boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # 计算IoU
    iou = interArea / boxAArea
    return iou


# --- 【修改】核心处理函数，增加对已存在检测结果的检查 ---
def process_and_recognize(mask, image_pil_obj, ocr_instance, existing_detections=None):
    """
    根据输入的蒙版处理轮廓并进行OCR识别。
    :param mask: 输入的二值蒙版
    :param image_pil_obj: 原始图片的PIL对象
    :param ocr_instance: ddddocr实例
    :param existing_detections: (可选) 已有的检测结果列表，用于避免重复检测
    :return: 包含识别结果的列表
    """
    if existing_detections is None:
        existing_detections = []

    # 形态学操作
    kernel = np.ones((3, 3), np.uint8)
    mask_opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    # mask_closed = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel, iterations=2)

    mask_closed = mask

    contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > MIN_CONTOUR_AREA]

    img_width, img_height = image_pil_obj.size
    padding = 5
    detections = []

    for cnt in valid_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        current_box = (x, y, x + w, y + h)

        # --- START OF MODIFICATION ---
        # 检查当前轮廓是否与已有的彩色检测结果重叠
        is_overlapping = False
        for existing_det in existing_detections:
            # 使用原始未padding的坐标进行IoU计算，更准确
            if calculate_iou(current_box, existing_det['pos_unpadded']) > IOU_THRESHOLD:
                is_overlapping = True
                break

        # 如果重叠度很高，则跳过这个轮廓
        if is_overlapping:
            # print(f"  - 发现重叠轮廓，已跳过。")
            continue
        # --- END OF MODIFICATION ---

        x1_padded = max(0, x - padding)
        y1_padded = max(0, y - padding)
        x2_padded = min(img_width, x + w + padding)
        y2_padded = min(img_height, y + h + padding)
        pos_padded = (x1_padded, y1_padded, x2_padded, y2_padded)

        cropped_image_color = image_pil_obj.crop(pos_padded)
        cropped_image_grayscale = cropped_image_color.convert('L')

        buffer = io.BytesIO()
        cropped_image_grayscale.save(buffer, format='PNG')
        char = ocr_instance.classification(buffer.getvalue(), png_fix=True)

        # 存储带padding和不带padding的坐标，用于后续不同的逻辑
        detections.append({
            'char': char,
            'pos': pos_padded,  # 用于绘图和显示的坐标
            'pos_unpadded': current_box  # 用于IoU计算的原始坐标
        })

    return mask_closed, detections


# --- 主循环 ---
if not os.path.isdir(images_dir):
    print(f"错误：请先创建名为 '{images_dir}' 的文件夹。")
else:
    for filename in os.listdir(images_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(images_dir, filename)
            print(f"--- 正在处理图片: {image_path} ---")
            image = cv2.imread(image_path)
            if image is None: continue

            img_pil = Image.open(image_path)
            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            all_detections = []

            # 1. 处理高饱和度（彩色）部分
            print(">>> 阶段1：处理彩色图案...")

            for color_name, (lower, upper) in hsv_ranges.items():
                # a. 为当前颜色创建蒙版
                color_mask = cv2.inRange(hsv_image, lower, upper)

                # lower_color = np.array([67, 94, 0])
                # upper_color = np.array([179, 255, 255])
                # color_mask = cv2.inRange(hsv_image, lower_color, upper_color)
                processed_color_mask, color_detections = process_and_recognize(color_mask, img_pil, ocr)
                all_detections.extend(color_detections)
                print(f"通过{color_name}切割找到 {len(color_detections)} 个彩色目标。")

            # 2. 处理低饱和度（灰色、褐色）部分
            print(">>> 阶段2：处理灰色图案...")
            lower_gray = np.array([0, 16, 0])
            upper_gray = np.array([21, 55, 196])
            gray_mask = cv2.inRange(hsv_image, lower_gray, upper_gray)
            # --- 【修改】传入 color_detections 以进行重叠检查 ---
            processed_gray_mask, gray_detections = process_and_recognize(gray_mask, img_pil, ocr,
                                                                         existing_detections=color_detections)
            all_detections.extend(gray_detections)
            print(f"通过颜色切割找到 {len(gray_detections)} 个灰色目标 (已排除与彩色的重叠部分)。")
            print(f"总共识别到 {len(all_detections)} 个目标。")

            # --- 步骤3：基于合并后的结果进行识别与可视化 ---
            draw_img = image.copy()
            results = {}

            for det in all_detections:
                char = det['char']
                pos = det['pos']
                if char not in results:
                    results[char] = []
                results[char].append(pos)

            duplicate_chars = {c: p for c, p in results.items() if len(p) == 2}
            matched_char_set = set(duplicate_chars.keys())
            for detection in all_detections:
                char, (x1, y1, x2, y2) = detection['char'], detection['pos']
                color = (0, 0, 255) if char in matched_char_set else (0, 255, 0)
                cv2.rectangle(draw_img, (x1, y1), (x2, y2), color, 2)
                label_y = y1 - 10 if y1 - 10 > 10 else y1 + 20
                cv2.putText(draw_img, char, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            output_path = os.path.join(output_dir, f"result_{filename}")
            cv2.imwrite(output_path, draw_img)

            # 显示结果
            cv2.imshow(f"Result for {filename}", draw_img)
            cv2.imshow("Color Mask (Processed)", processed_color_mask)
            cv2.imshow("Gray Mask (Processed)", processed_gray_mask)
            print("按任意键关闭窗口并继续...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            if duplicate_chars:
                print("找到匹配的图案:")
                for char, positions in duplicate_chars.items():
                    print(f"  - 字符 '{char}'")
            else:
                print("未找到匹配的图案。")
            print("-" * 40)
