import pandas as pd
import numpy as np
import scipy as sc

def move(sw,su,h,y,r,p,pos):
    """
    Given a three-dimensional point pos, return the position of the point when rotated and translated
    accordingly.
    :param sw: sway/movement x-axis (float)
    :param su: surge/movement y-axis (float)
    :param h: heave/movement upward (float)
    :param y: yaw/rotation of z-axis (float)
    :param r: roll/rotation of x-axis (float)
    :param p: pitch/rotation of y-axis (float)
    :param pos: position (numpy array of length 3)
    :return: the translated position (numpy array of length 3)
    """
    rot_x = np.matrix([[1, 0, 0],
                       [0, np.cos(r), -np.sin(r)],
                       [0, np.sin(r), np.cos(r)]])
    rot_y = np.matrix([[np.cos(p), 0, np.sin(p)],
                       [0, 1, 0],
                       [-np.sin(p), 0, np.cos(p)]])
    rot_z = np.matrix([[np.cos(y), -np.sin(y), 0],
                       [np.sin(y), np.cos(y), 0],
                       [0, 0, 1]])
    rot = np.matmul(np.matmul(rot_x, rot_y), rot_z)
    rotRes = np.matmul(rot, pos)
    rotRes = np.array([rotRes[0, p] for p in range(3)])
    return np.add(rotRes,np.array([sw,su,h]))

#Get the dataset:
df = pd.read_csv('Modellenpracticum/clean_data.csv', header = [0,1])
print(df.head())


#Introduce the main data variables:
heave = np.array(df['z_wf'])
sway = np.array(df['y_wf'])
surge = np.array(df['x_wf'])
yaw = np.array(df['psi_wf'])
roll = np.array(df['phi_wf'])
pitch = np.array(df['theta_wf'])
dt = np.array(df['Delta_t'])
time = np.array(df['t'])
N = len(time) #Number of time steps
#Position of centre of helicopter deck relative to centre mass point of ship:
l = 3
H = [-50,0,-7.5]
#two different points to define plane of helideck
P_1 = [-50 + l, 0, -7.5]
P_2 = [-50, l, -7.5]



def heli_incl(heave, sway, surge, yaw, roll, pitch, time, pos_helideck, radius_helideck):
    N = len(time)
    H = pos_helideck
    P_1 = np.add(pos_helideck, np.array([radius_helideck,0,0]))
    P_2 = np.add(pos_helideck, np.array([0,radius_helideck,0]))
    H_pos = [np.array([]) for _ in range(N)]
    P_1_pos = [np.array([]) for _ in range(N)]
    P_2_pos = [np.array([]) for _ in range(N)]
    for i in range(N):
        H_pos[i] =  move(sway[i,0],surge[i,0],heave[i,0],yaw[i,0],roll[i,0],pitch[i,0],H)
        P_1_pos[i] =  move(sway[i,0],surge[i,0],heave[i,0],yaw[i,0],roll[i,0],pitch[i,0],P_1)
        P_2_pos[i] =  move(sway[i,0],surge[i,0],heave[i,0],yaw[i,0],roll[i,0],pitch[i,0],P_2)

    #define plane while moving
    normal = []
    for i in range(N):
        normal += [np.cross(np.add(P_1_pos[i], -1*H_pos[i]), np.add(P_2_pos[i], -1*H_pos[i]))]

    angle = []
    for k in range(N):
        angle += [np.arccos((normal[k][2])/np.linalg.norm(normal[k]))]
    return angle

df = df.assign(heli_incl=heli_incl(heave, sway, surge, yaw, roll, pitch, dt, time, H, l))
print(df.head(10))
#Determine heave speed
# Heave_Speed = [0]
# for j in range(len(H_pos) - 1):
#     Heave_Speed.append((H_pos[j + 1][2] - H_pos[j][2])/dt[j])
# print(Heave_Speed)