import os
import cv2
import numpy as np

# 用于存储鼠标点击的四个角点
points = []


def select_points(event, x, y, flags, param):
    """
    鼠标回调函数，用于在点击时存储点的坐标。
    """
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"Point selected: {(x, y)}")
        # 在图像上绘制一个小圆圈来标记选中的点
        cv2.circle(param, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Select 4 Points", param)


def image_path_generator(images_dir):
    """
    获取文件夹中的所有图片路径
    """
    # 常见的图片文件扩展名
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
    # 遍历文件夹
    for file in os.listdir(images_dir):
        # 构造文件完整路径
        file_path = os.path.join(images_dir, file)
        # 检查是否为文件（而不是目录）
        if os.path.isfile(file_path):
            # 获取文件的扩展名并转换为小写
            ext = os.path.splitext(file)[1].lower()
            # 如果文件扩展名是图片格式，则生成路径
            if ext in image_extensions:
                yield file_path


# 主程序部分
if __name__ == "__main__":
    # 图像路径
    img_dir = '../raw_images'  # 替换为你的图像文件夹路径

    # 创建文件保存文件夹
    save_dir = os.path.join(img_dir, './ipm')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 读取图像
    img0 = cv2.imread(next(image_path_generator(img_dir)))

    # 检查图像是否成功加载
    if img0 is None:
        print("Error: Could not load image. Check the file path.")
    else:
        # 显示图像并设置鼠标回调
        cv2.namedWindow("Select 4 Points", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Select 4 Points", 600, 800)
        cv2.imshow("Select 4 Points", img0)
        cv2.setMouseCallback("Select 4 Points", select_points, param=img0)

        print("Please select 4 points on the image.")

        # 持续等待，直到选取4个点
        while len(points) < 4:
            cv2.waitKey(1)

        # 关闭图像窗口
        cv2.destroyAllWindows()

        # 将选中的点转换为 numpy 数组
        print("Selected points: ", points)

        # 定义输出图像的尺寸（鸟瞰图的尺寸）
        world_points = np.array([
            [points[3][0], points[3][1] - (points[2][0] - points[3][0])],
            [points[2][0], points[3][1] - (points[2][0] - points[3][0])],
            list(points[2]),
            list(points[3])
        ], dtype=np.float32)

        src_points = np.array(points, dtype=np.float32)
        # 计算单应性矩阵
        H = cv2.getPerspectiveTransform(src_points, world_points)

        h, w = img0.shape[:2]
        # 计算原始图像四个角点经过变换后的新坐标
        corners = np.array([[0, 0], [w-1, 0], [w-1, h], [0, h-1]], dtype=np.float32).reshape(-1, 1, 2)
        transformed_corners = cv2.perspectiveTransform(corners, H).reshape(-1, 2)

        # 找到变换后的角点的最小和最大坐标
        min_x, min_y = np.min(transformed_corners, axis=0)
        max_x, max_y = np.max(transformed_corners, axis=0)

        # 计算输出图像的尺寸
        width = int(np.ceil(max_x - min_x))
        height = int(np.ceil(max_y - min_y))

        # 平移变换矩阵，确保图像显示在正坐标系中
        translation_matrix = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]])

        # 更新单应性矩阵，包含平移变换
        H = np.dot(translation_matrix, H)

        #  对整个图像进行逆透视变换
        # for img_path in image_path_generator(img_dir):
        #     # 读取图像
        #     img = cv2.imread(img_path)
        #     # 提取图片名称，包含后缀
        #     img_name = os.path.basename(img_path)
        #
        #     # 执行透视变换，生成完整的鸟瞰图
        #     result = cv2.warpPerspective(img, H, (width, height))
        #
        #     # 创建保存文件夹
        #     save_path = os.path.join(save_dir, img_name)
        #     cv2.imwrite(save_path, result)
        #     print(f'The ipm result of {img_name.split(".")[0]} has been saved {save_path}.')

        '''
            对轮廓进行逆透视变换后，进行四边形拟合然后再变换到原图像
        '''
        # 读取图像
        image = cv2.imread(os.path.join(img_dir, 'mask_0.png'))

        # 转换为灰度图并进行边缘检测
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # 寻找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 选择一个轮廓（例如最大的轮廓）
        largest_contour = max(contours, key=cv2.contourArea)

        # 确保轮廓点的数据类型为 float32
        contour_float32 = largest_contour.astype(np.float32)

        # 应用逆透视变换仅到轮廓点
        transformed_contour = cv2.perspectiveTransform(contour_float32.reshape(-1, 1, 2), H)

        # 创建一个空白图像用于绘制变换后的轮廓
        output_image = np.zeros_like(image)
        cv2.polylines(output_image, [transformed_contour.astype(np.int32).reshape((-1, 1, 2))], isClosed=True,
                      color=(0, 255, 0), thickness=2)
        # 显示结果
        cv2.imwrite('Transformed.jpg', output_image)
