import pandas as pd
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import special
from emma_dinand.extras import load_processed_data, save_processed_data
from constants import pos_helideck, timeThres, HRThres, rollThres, pitchThres, inclThres

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

def heave_speed(sw,su,h,y,r,p,pos,N, dt = None, new=False, name="heave_speed"):
    dt = np.repeat(0.2, N)
    # make it so this function also first tries to load the data from a pickle file
    try:
        if new:
            raise FileNotFoundError
        Heave_Speed = load_processed_data(f"emma_dinand/pickle_saves/vectors/heave_speed_{name}.pkl")
        print("Loaded heave speed from pickle.")

    except FileNotFoundError:
        print("No heave speed pickle found, calculating it.")

        H_pos = [np.array([]) for _ in range(N)]
        H_pos[0] = move(sw[0],su[0],h[0],y[0],r[0],p[0],pos)
        Heave_Speed = [0]
        for i in range(1,N):
            H_pos[i] = move(sw[i],su[i],h[i],y[i],r[i],p[i],pos)
            Heave_Speed.append(((H_pos[i][2] - H_pos[i - 1][2])/dt[i]).item())

        
        save_processed_data(np.array(Heave_Speed), f"emma_dinand/pickle_saves/vectors/heave_speed_{name}.pkl")
        print("saved heave speed to pickle.")

    return Heave_Speed

#function that returns an array of the helideck inclination with input 3d array of postition of helideck relative to center 0f gravity
def heli_incl(heave, sway, surge, yaw, roll, pitch, time, pos_helideck, new = False, name="heli_incl"):	
    try:
        if new:
            raise FileNotFoundError
        angle = load_processed_data(f"emma_dinand/pickle_saves/vectors/heli_incl_{name}.pkl")
        print(f"Loaded heli_incl {name} from pickle.")  
             
    except FileNotFoundError:
 

        N = len(time)
        H = pos_helideck
        P_1, P_2 = np.add(pos_helideck, np.array([1,0,0])), np.add(pos_helideck, np.array([0,1,0]))
        H_pos, P_1_pos, P_2_pos = [np.array([]) for _ in range(N)], [np.array([]) for _ in range(N)], [np.array([]) for _ in range(N)]
        normal = []
        angle = []
        for i in range(N):
            H_pos[i] =  move(sway[i],surge[i],heave[i],yaw[i],roll[i],pitch[i],H)
            P_1_pos[i] =  move(sway[i],surge[i],heave[i],yaw[i],roll[i],pitch[i],P_1)
            P_2_pos[i] =  move(sway[i],surge[i],heave[i],yaw[i],roll[i],pitch[i],P_2)

        for i in range(N):
            normal += [np.cross(np.add(P_1_pos[i], -1*H_pos[i]), np.add(P_2_pos[i], -1*H_pos[i]))]

        for k in range(N):
            angle += [np.arccos((normal[k][2])/np.linalg.norm(normal[k]))]
        
        save_processed_data(np.array(angle), f"emma_dinand/pickle_saves/vectors/heli_incl_{name}.pkl")
        print(f"Saved heli_incl {name} to pickle.")

    return np.array(angle)
#Get the dataset: (use copy as path)
#file_path = "C:/Users/caspe/OneDrive/Documents/Programming/Modellenpracticum/QPtest/out_clean_wavespreading_36000s.csv"
# df = pd.read_csv(file_path, header=[0,1])df_temp



def mark_QP(df, name="QP", new = False):
    try:
        if new:
            raise FileNotFoundError
        
        
        QP = load_processed_data(f"emma_dinand/pickle_saves/vectors/{name}.pkl")
        print("Loaded QP from pickle.")

    except FileNotFoundError:
        # df = pd.read_csv('Modellenpracticum/clean_data.csv', header=[0,1])
        # df_temp = df.copy()
        #Main data variables:
        heave, sway, surge, yaw, roll, pitch = np.array(df['z_wf']).flatten(), np.array(df['y_wf']).flatten(),np.array(df['x_wf']).flatten(),np.array(df['psi_wf']).flatten(),np.array(df['phi_wf']).flatten(),np.array(df['theta_wf']).flatten()
        
        
        time = np.array(df['t']).flatten()
        dt = np.repeat(0.2, len(time)) #Time step (in seconds)
                        
        N = len(time) #Number of time steps
        H = pos_helideck

        #adding the heave speed and heli incl to dataframe
        Heave_Speed = heave_speed(sway,surge,heave,yaw,roll,pitch,H,N, dt,new=new, name=name)
        # df_temp = df_temp.assign(z_velocity=Heave_Speed)

        Heli_Incl = heli_incl(heave, sway, surge, yaw, roll, pitch, time, H,new=new, name=name)
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
        save_processed_data(QP, f"emma_dinand/pickle_saves/vectors/{name}.pkl")
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
