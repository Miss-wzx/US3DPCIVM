
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

import vtk
import pickle, gzip


# 截图函数
def take_screenshot(render_window ,img_save_name, scale=1):
    # 创建窗口到图像过滤器
    window_to_image_filter = vtk.vtkWindowToImageFilter()
    window_to_image_filter.SetInput(render_window)
    window_to_image_filter.SetScale(scale, scale)  # 可选: 提高图像分辨率，截图为 2 倍大小 X Y 的放大倍数
    # window_to_image_filter.SetInputBufferTypeToRGBA()  # 捕获RGBA (包括alpha)
    window_to_image_filter.SetInputBufferTypeToRGB()  # 捕获RGB (不包括Alpha)
    window_to_image_filter.ReadFrontBufferOff()  # 读取后缓冲区
    window_to_image_filter.Update()

    # 创建PNG写入器
    writer = vtk.vtkPNGWriter()
    writer.SetFileName(img_save_name)
    writer.SetInputConnection(window_to_image_filter.GetOutputPort())
    writer.Write()

    print(f"Screenshot saved at {img_save_name}")



# 相邻两帧之间迭代插值 num_ibf 表示分为几段 相当于插值num_ibf-1次
def interpolate_between_frames(p, pn, wd, wdn, num_ibf):
    pos_list = []
    wave_data_list = []

    for n in range(num_ibf):
        new_p = p + n / num_ibf * (pn - p)
        new_wd = wd + n / num_ibf * (wdn - wd)
        pos_list.append(new_p)
        wave_data_list.append(new_wd)
        print(new_p)
    print("---------------")

    return pos_list, wave_data_list



def reconstruction(us_data_path, threshold, scan_start_stop_and_step, Z_axis_depth_range, img_path, num_ibf=2, perspective=None):

    # 以二进制读模式打开文件
    with gzip.open(us_data_path, 'rb') as f:
        us_data_loaded_list = pickle.load(f)

    print(len(us_data_loaded_list))

    # 数据格式[{'pos': [robot_pos.fX, robot_pos.fY, robot_pos.fZ, robot_pos.fXr, robot_pos.fYr, robot_pos.fZr],
    # 'wave_data': [_tmp.x, _tmp.y, _tmp.z, _tmp.amp, _tmp.x, _tmp.y, _tmp.z, _tmp.amp, ...]}, {}]

    pcd_data = []
    pcd_color = []

    sas = scan_start_stop_and_step

    if sas[1] == -1:
        sas[1] = len(us_data_loaded_list)

    rl = list(range(len(us_data_loaded_list))[sas[0]:sas[1]:sas[2]])
    print(rl)

    # 扫查起始和步进
    for i in rl:

        pos_list = []
        wave_data_list = []

        if i == rl[-1]:
            # 最后一个数据
            pos = us_data_loaded_list[i][0:6]
            wave_data = np.array(us_data_loaded_list[i][6:]).reshape(-1, 4)
            pos_list.append(pos)
            wave_data_list.append(wave_data)
        else:
            # 插值
            pos = us_data_loaded_list[i][0:6]
            wave_data = np.array(us_data_loaded_list[i][6:]).reshape(-1, 4)
            # 下一帧
            pos_next_frame = us_data_loaded_list[i + sas[2]][0:6]
            wave_data_next_frame = np.array(us_data_loaded_list[i + sas[2]][6:]).reshape(-1, 4)
            # 插值次数 num_ibf-1

            # 矫正机械臂位姿数据
            pos[3] =  abs(pos[3])
            pos_next_frame[3] = abs(pos_next_frame[3])

            pos_list, wave_data_list = interpolate_between_frames(pos, pos_next_frame, wave_data, wave_data_next_frame, num_ibf)

        for pos, wave_data in zip(pos_list, wave_data_list):

            # 提取平移向量
            T = np.array(pos[:3])
            # print(T)
            # 计算旋转矩阵 R（假设 fXr, fYr, fZr 是欧拉角，单位：弧度）
            alpha, beta, gamma = pos[3:]  # 偏航、俯仰、滚转角

            # print(alpha, beta, gamma)

            # 构造旋转矩阵（Z-Y-X 欧拉角）
            Rz = np.array([
                [np.cos(gamma), -np.sin(gamma), 0],
                [np.sin(gamma), np.cos(gamma), 0],
                [0, 0, 1]
            ])

            Ry = np.array([
                [np.cos(beta), 0, np.sin(beta)],
                [0, 1, 0],
                [-np.sin(beta), 0, np.cos(beta)]
            ])

            # alpha = 0 if abs(alpha) < 1e-2 else alpha

            Rx = np.array([
                [1, 0, 0],
                [0, np.cos(alpha), -np.sin(alpha)],
                [0, np.sin(alpha), np.cos(alpha)]
            ])

            # 旋转矩阵计算
            # 使用 Z-Y-X 欧拉角构造旋转矩阵，按 偏航 (Z 轴) → 俯仰 (Y 轴) → 滚转 (X 轴) 顺序旋转。

            # 固定轴旋转 滚转 (X 轴) → 俯仰 (Y 轴) → 偏航 (Z 轴)  顺序旋转
            R = Rz @ Ry @ Rx  # 组合旋转矩阵

            # 根据阈值取出第四列满足条件的行
            filtered_data = wave_data[wave_data[:, 3] > threshold]

            filtered_data = filtered_data[
                (filtered_data[:, 2] >= Z_axis_depth_range[0]) &
                (filtered_data[:, 2] <= Z_axis_depth_range[1])
                ]

            # print(filtered_data)

            # 提取前三列 (n,3) 的点云
            points_xyz = filtered_data[:, :3]
            points_color = filtered_data[:, 3]

            # print(points_color)

            # 应用变换：R @ p + T
            transformed_points = (R @ points_xyz.T).T + T

            # print(transformed_points)

            pcd_data.extend(transformed_points.tolist())
            pcd_color.extend(points_color.tolist())


    amp_norm = np.array(pcd_color) / 100  # 范围0-1
    # 自适应标准化范围
    # 归一化

    points = np.asarray(pcd_data)  # 确保是 NumPy 数组

    colormap = plt.get_cmap('jet')  # 使用 matplotlib 的 jet colormap （蓝→红）
    colors = colormap(amp_norm)[:, :3]  # 获取 RGB，不要 alpha 通道

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    # 保存为 PLY 格式
    # o3d.io.write_point_cloud(f"./pcd/pcd{num_ibf}_{threshold}.ply", pcd)

    # 使用 colormap 映射颜色
    # colormap = plt.get_cmap('viridis')
    colormap = plt.get_cmap('jet')
    colors_rgba = colormap(amp_norm)  # 包含 alpha，范围是0~1

    # 创建 vtkPoints
    vtk_points = vtk.vtkPoints()
    for pt in points:
        vtk_points.InsertNextPoint(pt)

    # 创建 RGBA 颜色数组（使用 unsigned char）
    vtk_colors = vtk.vtkUnsignedCharArray()
    vtk_colors.SetNumberOfComponents(4)
    vtk_colors.SetName("Colors")

    amp_norm = amp_norm * 255  # 透明度

    for i, rgba in enumerate(colors_rgba):
        r, g, b = (rgba[:3] * 255).astype(np.uint8)  # 转换为 0-255
        # vtk_colors.InsertNextTuple4(r, g, b, int(amp_norm[i]))
        vtk_colors.InsertNextTuple4(r, g, b, 255)  # 255不透明

    # 创建 polydata 和 cell array（每个点一个 vertex）
    poly_data = vtk.vtkPolyData()
    poly_data.SetPoints(vtk_points)

    # 声明单个点进行渲染
    vertices = vtk.vtkCellArray()
    for i in range(len(points)):
        vertices.InsertNextCell(1)
        vertices.InsertCellPoint(i)
    poly_data.SetVerts(vertices)

    poly_data.GetPointData().SetScalars(vtk_colors)

    # 创建 mapper
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(poly_data)

    # 创建 actor
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetPointSize(3)  # 点大小

    # 创建 renderer 和 window，开启透明度支持
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(1.0, 1.0, 1.0)

    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(1600, 1200)

    # 开启透明度支持
    render_window.SetAlphaBitPlanes(1)  # 启用 alpha 通道支持（关键）

    # 添加 actor
    renderer.AddActor(actor)

    # 创建交互器
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)

    # 添加坐标轴 actor
    axes = vtk.vtkAxesActor()
    axes.SetTotalLength(100, 100, 100)  # 增加坐标轴的长度
    axes.SetCylinderRadius(0.05)  # 增加坐标轴的粗细

    axes_widget = vtk.vtkOrientationMarkerWidget()
    axes_widget.SetOrientationMarker(axes)
    axes_widget.SetInteractor(interactor)
    axes_widget.SetViewport(0.0, 0.0, 0.3, 0.3)

    axes_widget.SetEnabled(1)
    axes_widget.InteractiveOff()
    # axes_widget.InteractiveOn()

    renderer.ResetCamera()

    # 启动可视化
    render_window.SetMultiSamples(8)  # 抗锯齿采样数，可为 4、8、16
    render_window.Render()
    # interactor.Start()  # 启动之后后续不能运行

    # 设置主视图（前视图）
    renderer.GetActiveCamera().SetPosition(0, -1, 0)  # 摄像机位于 Y 轴正方向
    renderer.GetActiveCamera().SetFocalPoint(0, 0, 0)  # 观察点是原点
    renderer.GetActiveCamera().SetViewUp(0, 0, 1)  # 上方向为 Z 轴负方向
    renderer.ResetCamera()  # 重置摄像机
    # 获取当前窗口
    render_window = interactor.GetRenderWindow()
    take_screenshot(render_window, img_path+f'C{num_ibf}_T{threshold}_{Z_axis_depth_range}_主视图.png')  # 截图

    # 设置俯视图（顶视图）
    renderer.GetActiveCamera().SetPosition(0, 0, 1)  # 摄像机位于 Z 轴正方向
    renderer.GetActiveCamera().SetFocalPoint(0, 0, 0)  # 观察点是原点
    renderer.GetActiveCamera().SetViewUp(0, 1, 0)  # 上方向为 Y 轴正方
    renderer.ResetCamera()  # 重置摄像机
    # 获取当前窗口
    render_window = interactor.GetRenderWindow()
    take_screenshot(render_window, img_path+f'C{num_ibf}_T{threshold}_{Z_axis_depth_range}_俯视图.png')  # 截图

    # 设置侧视图（右视图）
    renderer.GetActiveCamera().SetPosition(1, 0, 0)  # 摄像机位于 X 轴正方向
    renderer.GetActiveCamera().SetFocalPoint(0, 0, 0)  # 观察点是原点
    renderer.GetActiveCamera().SetViewUp(0, 0, 1)  # 上方向为 Z 轴正方向
    renderer.ResetCamera()  # 重置摄像机
    # 获取当前窗口
    render_window = interactor.GetRenderWindow()
    take_screenshot(render_window, img_path+f'C{num_ibf}_T{threshold}_{Z_axis_depth_range}_测视图.png')  # 截图

    # 自由视角
    renderer.GetActiveCamera().SetPosition(-1, -1, 1)  # 摄像机位
    renderer.GetActiveCamera().SetFocalPoint(0, 0, 0)  # 观察点是原点
    renderer.GetActiveCamera().SetViewUp(0, 0, 1)
    renderer.ResetCamera()  # 重置摄像机
    # 获取当前窗口
    render_window = interactor.GetRenderWindow()
    take_screenshot(render_window, img_path+f'C{num_ibf}_T{threshold}_{Z_axis_depth_range}_自由视图.png')  # 截图

    # interactor.Start()


if __name__ == "__main__":
    us_data_path = "./robot_scan_data.pkl.gz"  # pcd文件
    threshold = -0.1  # 范围0-100 百分比 0 需要设置为-0.1保证计算精度
    scan_start_stop_and_step = [0, -1, 1]  # 完整数据
    num_ibf = 3  # 插值分段数
    img_path = f'./img/cz{num_ibf}/'  # 截图命名
    # Z轴深度范围
    Z_axis_depth_range = [-0.1, 20]  # 测试数据范围0-20

    import os
    if not os.path.exists(img_path):
        os.makedirs(img_path)

    reconstruction(us_data_path, threshold, scan_start_stop_and_step, Z_axis_depth_range, img_path, num_ibf)
