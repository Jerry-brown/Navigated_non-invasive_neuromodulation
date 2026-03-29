import os

import mcubes
import numpy as np
import vedo
from vedo import Plotter


def obj2ply(input_path, output_path):
    for file in os.listdir(input_path):
        a = vedo.read(os.path.join(input_path, file))
        points = a.points()
        mins = np.min(points, axis=0)
        maxs = np.max(points, axis=0)

        # 线性归一化到 [-1, 1] 的范围
        # scaled_points = 2 * (points - mins) / (maxs - mins) - 1
        scaled_points = 2 * (points - np.array([0, 0, 0])) / (np.array([512, 512, 512]) - np.array([0, 0, 0])) - 1

        # 将归一化后的点坐标更新到 mesh 中
        a.points(scaled_points)
        a.write(os.path.join(output_path, file.replace('obj', 'ply')))
        print(file)


def exact_boundary(source_array):
    bound_array = np.zeros_like(source_array)
    # for x in range(source_array.shape[0]):
    #     for z in range(source_array.shape[2]):
    #         y_indices = np.where(source_array[x, :, z] >= 1)[0]  # 找到满足条件的位置索引
    #         if len(y_indices) > 0:
    #             y = y_indices[-1]  # 找到最大的y
    #             bound_array[x, y:min(y + 3, source_array.shape[1]), z] = 1  # 修改范围

    # 优化版操作2: 沿y轴反方向寻找 >= 1 的位置
    for x in range(source_array.shape[0]):
        for z in range(source_array.shape[2]):
            y_indices = np.where(source_array[x, :, z] >= 1)[0]
            if len(y_indices) > 0:
                y = y_indices[0]  # 找到最小的y
                bound_array[x, max(y - 3, 0):y, z] = 1

    # 优化版操作3: 沿z轴方向寻找 >= 1 的位置
    for x in range(source_array.shape[0]):
        for y in range(source_array.shape[1]):
            z_indices = np.where(source_array[x, y, :] >= 1)[0]
            if len(z_indices) > 0:
                z = z_indices[-1]
                bound_array[x, y, z:min(z + 3, source_array.shape[2])] = 1

    # 优化版操作4: 沿z轴反方向寻找 >= 1 的位置
    for x in range(source_array.shape[0]):
        for y in range(source_array.shape[1]):
            z_indices = np.where(source_array[x, y, :] >= 1)[0]
            if len(z_indices) > 0:
                z = z_indices[0]
                bound_array[x, y, max(z - 3, 0):z] = 1

    # 优化版操作5: 沿x轴方向寻找 >= 1 的位置
    for y in range(source_array.shape[1]):
        for z in range(source_array.shape[2]):
            x_indices = np.where(source_array[:, y, z] >= 1)[0]
            if len(x_indices) > 0:
                x = x_indices[-1]
                bound_array[x:min(x + 3, source_array.shape[0]), y, z] = 1

    # 优化版操作6: 沿x轴反方向寻找 >= 1 的位置
    for y in range(source_array.shape[1]):
        for z in range(source_array.shape[2]):
            x_indices = np.where(source_array[:, y, z] >= 1)[0]
            if len(x_indices) > 0:
                x = x_indices[0]
                bound_array[max(x - 3, 0):x, y, z] = 1
    return bound_array


def resize_and_pad_to_360(model_array, target_shape=(512, 512, 512)):
    """
    将一个三维模型数组调整为给定的目标尺寸，并在四周填充 0 使其居中。

    参数:
    - model_array: 输入的三维数组 (模型数据)。
    - target_shape: 目标尺寸，默认为 (360, 360, 360)。

    返回:
    - padded_model: 填充后的三维数组，尺寸为目标大小。
    """
    # 获取模型当前的尺寸
    current_shape = model_array.shape

    # 计算每个维度需要填充的数量
    pad_x = (target_shape[0] - current_shape[0]) // 2
    pad_y = (target_shape[1] - current_shape[1]) // 2
    pad_z = (target_shape[2] - current_shape[2]) // 2

    # 计算填充在每个维度的前后长度
    padding = ((pad_x, target_shape[0] - current_shape[0] - pad_x),
               (pad_y, target_shape[1] - current_shape[1] - pad_y),
               (pad_z, target_shape[2] - current_shape[2] - pad_z))

    # 在模型四周填充 0
    padded_model = np.pad(model_array, padding, mode='constant', constant_values=0)

    return padded_model


def npy2obj(input_dir, out_dir):
    i = 0
    for file in os.listdir(input_dir):
        if os.path.exists(os.path.join(out_dir, file.replace('npy', 'obj'))):
            continue
        try:
            head = np.load(os.path.join(input_dir, file))
        except ValueError as e:
            print(e)
            print(file)
            continue
        head = resize_and_pad_to_360(head)
        vertices, traingles = mcubes.marching_cubes(head, 0.5)
        mcubes.export_obj(vertices, traingles, os.path.join(out_dir, file.replace('npy', 'obj')))
        print(head.shape)
        i += 1
    print()


npy2obj(r'/share/home/202321044005/XCube/data/pred/npy', r'/share/home/202321044005/XCube/data/pred/obj')
obj2ply(r'/share/home/202321044005/XCube/data/pred/obj', r'/share/home/202321044005/XCube/data/pred/ply')

# xyz = np.load('a.xyz.npy')
# p = vedo.Points(xyz)
# plt = Plotter()
# plt.add(p)
# plt.show(interactive=True)

# npy2obj(r'E:\code\DiffComplete-main\data\head', r'D:\dataset\xcube\ours_test\obj')
# obj2ply(r'D:\dataset\xcube\ours_test\obj', r'D:\dataset\xcube\ours_test\ply')

# path = r'D:\dataset\xcube\shapenet\1510685'
# for file in os.listdir(path):
#     print()
#     mesh = vedo.read(os.path.join(path, file))
#     head = np.load(r'E:\code\DiffComplete-main\data\head\Ai Xiu Li_head.npy')
#     vo = vedo.Volume(head)
#     vo.tomesh()
#     vertices, traingles = mcubes.marching_cubes(head, 0.5)
#     mcubes.export_obj(vertices, traingles, 'out_file.obj')
#     points = mesh.points()
#     mesh.write(os.path.join(path, file.replace('.obj', '.ply')))
#     plt = Plotter(axes=1)
#     plt.add(mesh)
#     plt.show(interactive=True)

#


# path = r'D:\dataset\xcube\1118\face\pkl\128'
# # path = r'G:\dataset\thy_ct\pkl\128\03001627'
# with open(os.path.join(path, 'train.lst'), 'w') as f:
#     i = 0
#     for file in os.listdir(path):
#         if i < 680:
#             if file.endswith('lst'):
#                 continue
#             f.write(file.replace('.pkl', '') + '\n')
#         i += 1
# with open(os.path.join(path, 'val.lst'), 'w') as f:
#     j = 0
#     for file in os.listdir(path):
#         if j > 680:
#             if file.endswith('lst'):
#                 continue
#             f.write(file.replace('.pkl', '') + '\n')
#         j += 1
