import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import csv

VarMaxx = 100
VarMaxy = 100
Rc = 10
Rs = 10

small_radius=1

image = Image.open('C1_v2.png')
image_resized = image.resize((VarMaxx+1, VarMaxy+1))
image_L = image_resized.convert('L')
Area1 = np.zeros((VarMaxx+1,VarMaxy+1))
image_matrix = np.array(image_L)

image_1 = Image.open('C1_real.png')

for i in range(VarMaxx+1):
    for j in range(VarMaxy+1):
        if image_matrix[i,j]  > 1:
            Area1[i,j] = 255
        else:
            Area1[i,j] = 1

ban_position_list = np.argwhere(Area1 == 1)
ban_position = [(x, y) for y, x in ban_position_list]


def draw_circle(ax, center, radius, small_radius):
    # Đường bao  phủ của cảm biến
    outline_circle = plt.Circle(center, radius, fill=False, ec='black', lw=0.8, alpha=1)  
    ax.add_artist(outline_circle)
    
    # Phạm vi cảm biến
    large_circle = plt.Circle(center, radius, color='cyan', alpha=0.2)  
    ax.add_artist(large_circle)

    # Hình tròn nhỏ bên trong
    small_circle = plt.Circle(center, small_radius, fill=False, ec='red', lw=1, alpha=0.7)  
    ax.add_artist(small_circle)
    
def plot_sensor(sensor_nodes, fitness):
    
    fig, (ax,ay) = plt.subplots(1,2)
    ax.set_xlim(0, VarMaxx)
    ax.set_ylim(0, VarMaxy)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks(np.arange(0, 101, 10))
    ax.set_yticks(np.arange(0, 101, 10))
    ax.grid(True, linewidth=0.5)
    ax.invert_yaxis()
    
    ay.set_xlim(0, VarMaxx)
    ay.set_ylim(0, VarMaxy)
    ay.set_aspect('equal', adjustable='box')
    ay.set_xticks(np.arange(0, 101, 10))
    ay.set_yticks(np.arange(0, 101, 10))
    ay.grid(True, linewidth=0.5)
    ay.invert_yaxis()

    # Hình 1: Vẽ các điểm trong chèn nền

    for i, node in enumerate(sensor_nodes):
        draw_circle(ax,node,Rs,small_radius)
    ax.imshow(image_1, extent=[0, VarMaxx, VarMaxy, 0])
    
    # Hình 2: Vẽ các điểm và Vẽ lại các vật cản

    for i, node in enumerate(sensor_nodes):
        draw_circle(ay,node,Rs,small_radius)
    for bp in ban_position:
        ay.plot(bp[0], bp[1], 'ko')

    plt.grid(True)
    caculator = len(ban_position)/((VarMaxy+1)*(VarMaxx+1))
    ax.set_title(f"Coverage percent:{round(fitness/(1-caculator),4) * 100 :.2f} %")
    ay.set_title(f"Coverage percent:{round(fitness/(1-caculator),4) * 100 :.2f} %")
    plt.show()

with open('FOA.CSV', mode ='r', newline='') as file:
    csv_reader = csv.reader(file)
    csv_header = next(csv_reader)
    fitness = float(csv_header[0])
    sensor_nodes =[]
    for i in csv_reader:
        sensor_nodes.append((float(i[0]),float(i[1])))
    plot_sensor(sensor_nodes, fitness)
