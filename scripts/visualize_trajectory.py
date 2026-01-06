import sys
import os
import click
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

from umi.common.orb_slam_util import load_csv_trajectory

@click.command()
@click.option('-i', '--input', required=True, help='Path to camera_trajectory.csv')
@click.option('--step', default=10, type=int, help='Interval to draw coordinate axes')
@click.option('--axis_size', default=0.05, type=float, help='Size of the coordinate axes')
def main(input, step, axis_size):
    # 使用 codebase 现有的工具加载 CSV
    print(f"Loading trajectory from {input}...")
    result = load_csv_trajectory(input)
    
    if 'pose' not in result:
        print("Error: No valid tracked poses found in the CSV.")
        return
        
    poses = result['pose']  # shape: (N, 4, 4)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 1. 提取位置 (x, y, z) 并绘制轨迹线
    # 变换矩阵中 [0:3, 3] 是平移部分
    x = poses[:, 0, 3]
    y = poses[:, 1, 3]
    z = poses[:, 2, 3]
    
    ax.plot(x, y, z, label='Camera Path', color='blue', alpha=0.7)
    
    # 2. 绘制各个点的坐标轴 (Orientation)
    # 这里参考了 k3d_util.py 的逻辑，用红/绿/蓝表示 X/Y/Z 轴
    for i in range(0, len(poses), step):
        pos = poses[i, :3, 3]
        rot = poses[i, :3, :3]
        
        # X 轴 (红色)
        ax.quiver(pos[0], pos[1], pos[2], rot[0,0], rot[1,0], rot[2,0], 
                  length=axis_size, color='red', alpha=0.5)
        # Y 轴 (绿色)
        ax.quiver(pos[0], pos[1], pos[2], rot[0,1], rot[1,1], rot[2,1], 
                  length=axis_size, color='green', alpha=0.5)
        # Z 轴 (蓝色)
        ax.quiver(pos[0], pos[1], pos[2], rot[0,2], rot[1,2], rot[2,2], 
                  length=axis_size, color='blue', alpha=0.5)

    # 绘制起点和终点
    ax.scatter(x[0], y[0], z[0], color='green', s=100, label='Start')
    ax.scatter(x[-1], y[-1], z[-1], color='red', s=100, label='End')

    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Z (meters)')
    ax.set_title('SLAM Camera Trajectory Visualization')
    ax.legend()
    
    # 保持坐标轴比例一致
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.show()

if __name__ == "__main__":
    main()