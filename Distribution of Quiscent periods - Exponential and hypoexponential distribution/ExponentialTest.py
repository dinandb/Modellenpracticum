import pandas as pd
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as patches
from scipy import special
from pathlib import Path

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'font.size' : 8,
    'pgf.rcfonts': False,
    'text.usetex': True,
    'axes.titlesize': 14,
    'axes.labelsize': 11,
    'lines.linewidth' : 0.5,
     'lines.markersize'  : 5,
    'xtick.labelsize' : 8,
    'ytick.labelsize': 8})


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

def heave_speed(sw,su,h,y,r,p,pos,N):
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
file_path = "C:\\Users\\caspe\\OneDrive\\Documents\\GitHub\\Modellenpracticum\\casper_jort\\Data\\5415M_Hs=5m_Tp=10s_10h_clean.csv"
df = pd.read_csv(file_path, header=[0,1])
#df = pd.read_csv('Modellenpracticum/clean_data.csv', header=[0,1])

#Main data variables:
heave, sway, surge, yaw, roll, pitch = np.array(df['z_wf']), np.array(df['y_wf']),np.array(df['x_wf']),np.array(df['psi_wf']),np.array(df['phi_wf']),np.array(df['theta_wf'])
dt, time = [0.2]*180001, np.array(df['t'])
N = len(time) #Number of time steps
H = [-50,0,-7.5] #Position helicopter deck relative to COM of ship

#adding the heave speed and heli incl to dataframe
Heave_Speed = heave_speed(sway,surge,heave,yaw,roll,pitch,H,N)
#df = df.assign(z_velocity=Heave_Speed)

Heli_Incl = heli_incl(heave, sway, surge, yaw, roll, pitch, time, H)
#df = df.assign(heli_incl=Heli_Incl)


QP = np.array([])
QPstart = []
QPend = []


#tresholds for QPs (see document Ed)
timeThres = 30
rollThres = sc.special.radian(2.0,0,0)
pitchThres = rollThres
inclThres = sc.special.radian(2.5,0,0)
HRThres = 1.0



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
#df = df.assign(QP=QP)

#check if data is correct
print(df.head())
print(QPstart)

#exporting data (choose your own path)
#df.to_csv (r"C:\Users\steve\Downloads\CleanQP_data_36000.csv", index = False, header=True)

#making plots
#fig, ax = plt.subplots(1, 1, figsize=(12, 5))
#for i in range(len(QPstart)):
 #   ax.axvspan(QPstart[i].item(), QPend[i].item(), facecolor='green', alpha=0.2)
#ax.plot(time, Heave_Speed)
#ax.set_xlabel("Time (s)")
#ax.set_ylabel("Heave speed (m/s)")
#ax.set_title("QPs")
#plt.show()

QPstart,QPend = np.array(QPstart), np.array(QPend)
QPstart = np.array([QPstart[r,0] for r in range(len(QPstart))])
QPend = np.array([QPend[u,0] for u in range(len(QPstart))])
QPduration = QPend-QPstart
QPduration_n = QPduration - timeThres*np.ones(len(QPduration))
QPend_n = np.append(np.array([0]),QPend)
QPstart_n = np.append(QPstart, np.array([max(time)]))
QPbetweentimes = QPstart_n-QPend_n
if QPbetweentimes[len(QPbetweentimes)-1] == 0: #the last element can be 0 if the simulation ends with a QP active
    QPbetweentimes = np.delete(QPbetweentimes,len(QPbetweentimes)-1)
print(f"QPduration_n: {QPduration_n}, has length {len(QPduration_n)} and mean {np.mean(QPduration_n)}")
print(f"QPbetweentimes: {QPbetweentimes}, has length {len(QPbetweentimes)} and mean {np.mean(QPbetweentimes)}")

#Histogram:
figure, axis = plt.subplots(1,2)

x = np.arange(0,np.max(QPduration_n),0.5)
y = np.arange(0,np.max(QPbetweentimes),0.5)

lambda_1 = 1/np.mean(QPduration_n)
axis[0].hist(QPduration_n,density=True,bins=30,label='Histogram')
axis[0].plot(x,lambda_1*np.exp(-lambda_1*x),label='MLE scaled exp pdf')
axis[0].set_title("Normalized QP duration")
axis[0].set_xlabel('time (s)')
axis[0].set_ylabel('density')
axis[0].legend()
#axis[0].set_ylim([0,0.050])

lambda_2 = 1/np.mean(QPbetweentimes)
axis[1].hist(QPbetweentimes,density=True,bins=30,label='Histogram')
axis[1].plot(y,lambda_2*np.exp(-lambda_2*y),label='MLE scaled exp pdf')
axis[1].set_title("Times between QP")
axis[1].set_xlabel('time (s)')
axis[1].set_ylabel('density')
axis[1].legend()
#axis[1].set_ylim([0,0.010])


#Do the KS test:
print(sc.stats.kstest(QPduration_n,'expon',args=(0,np.mean(QPduration_n))))
print(sc.stats.kstest(QPbetweentimes,'expon',args=(0,np.mean(QPbetweentimes))))

#print(sc.stats.kstest(QPduration_n,'norm',args=(np.mean(QPduration_n), np.std(QPduration_n,ddof=0))))
#print(sc.stats.kstest(QPbetweentimes,'norm',args=(np.mean(QPbetweentimes), np.std(QPbetweentimes,ddof=0))))

# plt.show()



figure.tight_layout()
plt.subplots_adjust(bottom=0.15, right=1, top=0.85)
figure.set_size_inches(w=5.5, h=3.5)
plt.savefig("C:\\Users\\caspe\\OneDrive\\Documents\\GitHub\\Modellenpracticum\\casper_jort\\Output\\" + Path(__file__).stem + ".pgf")

#Correlation test:
#QPbet_n = np.delete(QPbetweentimes,0)
b = min(len(QPduration),len(QPbetweentimes))
C = np.corrcoef(QPduration_n[0:b-1],QPbetweentimes[0:b-1])
print(f"correlation coeff: {C}")


