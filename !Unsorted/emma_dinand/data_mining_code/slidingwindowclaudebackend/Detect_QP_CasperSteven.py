import pandas as pd
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import special
from extras import load_processed_data, save_processed_data

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

def heave_speed(sw,su,h,y,r,p,pos,N, dt):
    H_pos = [np.array([]) for _ in range(N)]
    H_pos[0] = move(sw[0,0],su[0,0],h[0,0],y[0,0],r[0,0],p[0,0],pos)
    Heave_Speed = [0]
    for i in range(1,N):
        H_pos[i] = move(sw[i,0],su[i,0],h[i,0],y[i,0],r[i,0],p[i,0],pos)
        Heave_Speed.append(((H_pos[i][2] - H_pos[i - 1][2])/dt[i]).item())
    return Heave_Speed

#function that returns an array of the helideck inclination with input 3d array of postition of helideck relative to center 0f gravity
def heli_incl(heave, sway, surge, yaw, roll, pitch, time, pos_helideck):
    N = len(time)
    H = pos_helideck
    P_1, P_2 = np.add(pos_helideck, np.array([1,0,0])), np.add(pos_helideck, np.array([0,1,0]))
    H_pos, P_1_pos, P_2_pos = [np.array([]) for _ in range(N)], [np.array([]) for _ in range(N)], [np.array([]) for _ in range(N)]
    normal = []
    angle = []
    for i in range(N):
        H_pos[i] =  move(sway[i,0],surge[i,0],heave[i,0],yaw[i,0],roll[i,0],pitch[i,0],H)
        P_1_pos[i] =  move(sway[i,0],surge[i,0],heave[i,0],yaw[i,0],roll[i,0],pitch[i,0],P_1)
        P_2_pos[i] =  move(sway[i,0],surge[i,0],heave[i,0],yaw[i,0],roll[i,0],pitch[i,0],P_2)

    for i in range(N):
        normal += [np.cross(np.add(P_1_pos[i], -1*H_pos[i]), np.add(P_2_pos[i], -1*H_pos[i]))]

    for k in range(N):
        angle += [np.arccos((normal[k][2])/np.linalg.norm(normal[k]))]
    return angle

#Get the dataset: (use copy as path)
#file_path = "C:/Users/caspe/OneDrive/Documents/Programming/Modellenpracticum/QPtest/out_clean_wavespreading_36000s.csv"
# df = pd.read_csv(file_path, header=[0,1])df_temp

# oude
# timeThres = 30
# HRThres = 0.5
# rollThres = sc.special.radian(1.0,0,0)
# pitchThres = sc.special.radian(1.0,0,0)
# inclThres = sc.special.radian(1.5,0,0)



# zwaarder
timeThres = 30
HRThres = 0.3
rollThres = sc.special.radian(0.7,0,0)
pitchThres = sc.special.radian(0.7,0,0)
inclThres = sc.special.radian(1.1,0,0)

# lichter
# timeThres = 30
# HRThres = 0.7
# rollThres = sc.special.radian(1.2,0,0)
# pitchThres = sc.special.radian(1.2,0,0)
# inclThres = sc.special.radian(1.8,0,0)


def mark_QP(df,name="QP",new = False):
    try:
        if new:
            raise FileNotFoundError
        
        
        QP = load_processed_data(f"slidingwindowclaudebackend/pickle_saves/vectors/{name}.pkl")
        print("Loaded QP from pickle.")

    except FileNotFoundError:
        # df = pd.read_csv('Modellenpracticum/clean_data.csv', header=[0,1])
        # df_temp = df.copy()
        #Main data variables:
        heave, sway, surge, yaw, roll, pitch = np.array(df['z_wf']), np.array(df['y_wf']),np.array(df['x_wf']),np.array(df['psi_wf']),np.array(df['phi_wf']),np.array(df['theta_wf'])
        dt, time = np.array(df['Delta_t']), np.array(df['t'])
        N = len(time) #Number of time steps
        H = [-50,0,-7.5] #Position helicopter deck relative to COM of ship

        #adding the heave speed and heli incl to dataframe
        Heave_Speed = heave_speed(sway,surge,heave,yaw,roll,pitch,H,N, dt)
        # df_temp = df_temp.assign(z_velocity=Heave_Speed)

        Heli_Incl = heli_incl(heave, sway, surge, yaw, roll, pitch, time, H)
        # df_temp = df_temp.assign(heli_incl=Heli_Incl)


        QP = np.array([])
        QPstart = []
        QPend = []


        #tresholds for QPs (see document Ed)



        i = 0
        while(i < len(time)):
            if(abs(Heave_Speed[i]) < HRThres and abs(roll[i]) < rollThres and abs(pitch[i]) < pitchThres) and Heli_Incl[i] < inclThres:
                j = 1
                while(abs(Heave_Speed[i + j - 1]) < HRThres and abs(roll[i + j - 1]) < rollThres and abs(pitch[i + j - 1]) < pitchThres  and Heli_Incl[i] < inclThres and i + j + 1 < len(Heave_Speed)):
                    j = j + 1
                if(time[i + j - 1] - time[i] >= timeThres):
                    QP= np.concatenate((QP, np.repeat(True, j)))
                    QPstart.append(time[i])
                    QPend.append(time[i+j])
                elif(time[i + j - 1] - time[i] < timeThres):
                    QP = np.concatenate((QP, np.repeat(False, j)))
                i = i+j
            else:
                QP = np.append(QP, False)
                i = i + 1
        #adding 
        # df_temp = df_temp.assign(QP=QP)

        #check if data is correct
        # print(df.head())
        # print(QPstart)
        save_processed_data(QP, f"slidingwindowclaudebackend/pickle_saves/vectors/{name}.pkl")
        print("Saved QP data to pickle.")

        # fig, ax = plt.subplots(1, 1, figsize=(12, 5))
        # for i in range((len(QPstart))):
        #     ax.axvspan(QPstart[i].item(), QPend[i].item(), facecolor='green', alpha=0.2)
        # ax.plot(time, Heave_Speed)
        # ax.set_xlabel("Time (s)")
        # ax.set_ylabel("Heave speed (m/s)")
        # ax.set_title("QPs")
        # plt.show()

    return QP

#exporting data (choose your own path)
# df.to_csv (r"C:\Users\steve\Downloads\CleanQP_data_36000.csv", index = False, header=True) 

#making plots
