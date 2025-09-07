import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from numba import jit 
import time
from celia_MPM_T1_VG_psiall_3_qa import ModelRun,thetafun
from scipy.ndimage import gaussian_filter1d
from numba import types
from numba.typed import Dict
from scipy.interpolate import interp1d

def plotset():#xlabel,x1,x2,ylabel,y1,y2,
  # 显示中文
  plt.rcParams['font.sans-serif'] = ['SimHei']
  plt.rcParams['axes.unicode_minus'] = False
  # 画图
  #plt.figure(figsize=(8, 7.2))
  plt.tick_params(bottom=True, top=True, left=True, right=True, direction='in', width=1.5)  # 设置边框线宽
  plt.gca().spines['top'].set_linewidth(1.5)  # 设置边框线宽
  plt.gca().spines['bottom'].set_linewidth(1.5)  # 设置边框线宽
  plt.gca().spines['right'].set_linewidth(1.5)  # 设置边框线宽
  plt.gca().spines['left'].set_linewidth(1.5)  # 设置边框线宽
  #plt.xlim(x1, x2)  # 设置y轴范围
  #plt.ylim(y1, y2)  # 设置x轴范围
  plt.xticks(size=25, family='Arial', weight='normal')  # 坐标数据设置
  plt.yticks(size=25, family='Arial', weight='normal')  # 坐标数据设置
  #plt.xlabel(xlabel, family="Arial", weight='normal', size=25)#坐标题注
  #plt.ylabel(ylabel, family="Arial", weight='normal', size=25)#坐标题注

def MakeDictFloat():
    d=Dict.empty(
    key_type=types.unicode_type,
    value_type=types.float64,)
    return d

def setpars():
    pars=MakeDictFloat()
    pars['thetaR']=0.04
    pars['thetaS']=0.41
    pars['thetaI']=0.2
    pars['alpha']=0.019
    pars['n']=1.31
    pars['m']=1-1/pars['n']
    pars['neta']=0.5
    pars['a0']=0.43
    pars['Ss']=0.000001
    pars['Ks']=8e-3/60*1.3 #(cm/s)
    #Top boundary
    pars['inf']=2 #(cm)积水入渗边界条件
    pars['finalH']=-100 #(cm)
    pars['t1']=45*60 #(s)第一次积水入渗阶段，水位变低至零
    pars['t2']=(45+260)*60#(s)#第二次积水入渗阶段，水位变低至零
    pars['t3']=(45+260+260)*60  # (s)第三次积水入渗阶段，水位变低至零
    pars['t4']=86400#(s)上层水头为最终阶段
    return pars



def TopBfun(psiT,pars,nt,t):
    inf=pars['inf']
    finalH=pars['finalH']
    t1=pars['t1']
    t2=pars['t2']
    t3=pars['t3']
    for i in range(nt):
        if t[i]<=t1:
            #psiT[i]=inf*(1-t[i]/t1) #积水入渗阶段，水位变低至零
            psiT[i]=inf-(2 * 0.022028 * t[i]**0.5-0.00010708 * t[i])
        elif t[i]<=t2:
            #psiT[i]=inf*(1-(t[i]-t1)/(t2-t1)) #积水入渗阶段，水位变低至零
            psiT[i]=4-(2 * 0.022028 * t[i]**0.5-0.00010708 * t[i])
        elif t[i]<=t3:
            psiT[i]=inf*(1-(t[i]-t2)/(t3-t2)) #积水入渗阶段，水位变低至零
            #psiT[i]=6-(2 * 0.022028 * t[i]**0.5-0.00010708 * t[i])
        else:
            psiT[i]=0 #入渗流量为零阶段
    return psiT

def Ksfun(n,pars):
    Ks0=np.ones(n)
    Ks=np.ones(n)
    Ks[:11]=Ks0[:11]*pars['Ks']
    Ks[11:21]=Ks0[11:21] * pars['Ks']*0.36
    Ks[21:51]=Ks0[21:51] * pars['Ks']*0.33
    Ks[51:]=Ks0[51:] * pars['Ks']*0.33
    return Ks
def Qfun(qa,t1,t2,dt):
    Q=sum(qa[t1:t2,0])*dt
    return Q
dt=10
def setup(dt):
    # Set parameters:
    pars=setpars()

    # Grid:
    dz=1
    zN=100.

    tN=pars['t4']
    t_1=round(pars['t1'] / dt)
    t_2=round(pars['t2'] / dt)
    t_3=round(pars['t3'] / dt)
    th_q=round(pars['t3']/dt)
    tq_h=round(pars['t4'] / dt)

    z=np.arange(0,zN+dz,dz)
    n=len(z)

    t=np.arange(0,tN+dt,dt)
    nt=len(t)

    # Initialize array:
    psi=np.zeros((nt,n))
    fmax=np.zeros((nt,n))
    qa=np.zeros((nt,n))
    #Ks=gaussian_filter1d(Ksfun(n,pars), sigma=1)
    Ks=pars['Ks']
    # ICs:
    data=pd.read_excel('D:\data\pythonProject\Baker_and_Hillel(1990)\Sheng2014.xlsx', sheet_name='P2含水量')
    Pzi=data.loc[:, 'zi']
    Pthetai=data.loc[:, 'thetai']
    linear_θ=interp1d(Pzi, Pthetai, kind='linear')
    θ_linear=linear_θ(z)
    θi=gaussian_filter1d(θ_linear, sigma=4)
    psi[0,:]=-1/pars['alpha']*(((θi-pars['thetaR'])/(pars['thetaS']-pars['thetaR']))**(-1/pars['m'])-1)**(1/pars['n'])
    fmax[0,:]=np.ones(n)*0.00001
    # BCs:
    psiT=np.ones(nt)#*pars['Ks']*0.8
    psiT=TopBfun(psiT, pars, nt, t)
    psiB=np.array([psi[0,-1]])
    #储存上边界条件到数组中
    boundary=np.zeros((2,nt))
    boundary[0]=t/60/60
    boundary[1]=psiT

    return z,t,tN,t_1,t_2,t_3,th_q,tq_h,dz,n,nt,zN,psi,fmax,qa,pars,dt,psiT,psiB,Ks,θi,boundary

z,t,tN,t_1,t_2,t_3,th_q,tq_h,dz,n,nt,zN,psi,fmax,qa,pars,dt,psiT,psiB,Ks,θi,boundary=setup(dt)
#plot('time (s)',0,tN,'head (cm)',-150,5,)
#plt.plot(Ks,z)
# 创建一个 Figure 对象和两个子图对象
fig, ax1 = plt.subplots(figsize=(6,5))
# 在第一个子图上绘制数据
plt.xlim(-0.5, tN)
#plt.ylim(-1, 5)  # 设置y轴范围
ax1.plot(t[:th_q+1], psiT[:th_q+1], 'k-',label="h")
ax1.plot(t[tq_h+1:], psiT[tq_h+1:], 'k-')
ax1.set_ylabel('head (cm)', color='k')
#plt.legend(labels=["h"],loc=1,edgecolor='k')
# 创建第二个子图对象，并共享 x 轴
ax2 = ax1.twinx()
plt.ylim(-1*pars['Ks'], 5*pars['Ks'])  # 设置y轴范围
ax2.plot(t[th_q+1:tq_h+1], psiT[th_q+1:tq_h+1], 'r-',label="q$_s$")
ax2.set_ylabel('Water flux (cm/s)', color='k')
ax1.set_xlabel('time (s)')
ax1.legend(bbox_to_anchor=(0.9, 0.9))
ax2.legend(bbox_to_anchor=(0.91, 0.82))
fig.subplots_adjust(left=0.15, right=0.8, top=0.85, bottom=0.15)
#plt.legend(labels=["qs"],loc=1,edgecolor='k')#,prop=font)
plt.show()

tic=time.time()
psi,fmax,qa=ModelRun(dt,dz,z,nt,th_q,tq_h,psi,fmax,qa,psiB,psiT,pars,Ks,θi)#,QIN,QOUT,S,err
runtime=time.time()-tic
print('Celia solution, dt=%.4f, runtime = %.2f seconds'%(dt,runtime))
Q1=Qfun(qa,0,t_1,dt)
print("第一阶段总入渗水量%.2f(cm)"%(Q1))
Q2=Qfun(qa,t_1,t_2,dt)
print("第一阶段总入渗水量%.2f(cm)"%(Q2))
Q3=Qfun(qa,t_2,t_3,dt)
print("第一阶段总入渗水量%.2f(cm)"%(Q3))
#plot('qa ',-0.5*pars['Ks'],30*pars['Ks'],'z (m)',zN, -0.02)
data=pd.read_excel('D:\data\pythonProject\Baker_and_Hillel(1990)\Sheng2014.xlsx', sheet_name='P2含水量')
Pzi=data.loc[:, 'zi']
Pthetai=data.loc[:, 'thetai']
data=pd.read_excel('D:\data\pythonProject\Baker_and_Hillel(1990)\Sheng2014.xlsx', sheet_name='P2含水量')
P2z=data.loc[:, 'z']
P2theta=data.loc[:, 'theta']
data=pd.read_excel('D:\data\pythonProject\Baker_and_Hillel(1990)\Sheng2014.xlsx', sheet_name='f_P2')
P2zf=data.loc[:, 'z']
P2f=data.loc[:, 'f']
result=np.zeros((21,n));k=1#储存数据到数组中
fig, axs = plt.subplots(1, 4,figsize=(11, 4))
for i in [0,1080,2160,4320,8640]:#range(0,nt,500):
    axs[0].plot(qa[i,:],z ,linestyle='-', linewidth=1)
    axs[1].plot(fmax[i,:],z ,linestyle='-', linewidth=1)
    axs[2].plot(psi[i,:],z ,linestyle='-', linewidth=1)
    axs[3].plot(thetafun(psi[i, :], pars), z, linestyle='-', linewidth=1)
    result[k]=qa[i, :]
    result[k+5]=fmax[i, :]
    result[k+10]=psi[i, :]
    result[k+15]=thetafun(psi[i, :], pars)
    k=k+1
#axs[1].scatter(P1f,P1zf, s = 20,c='r',label="P1")
axs[1].scatter(P2f,P2zf, s = 20,c='k',label="Sheng2014_P2f")
axs[3].scatter(Pthetai,Pzi, s = 20,c='r',label="Sheng2014_P2i")
axs[3].scatter(P2theta,P2z, s = 20,c='k',label="Sheng2014_P2a")
axs[0].set_ylabel('z (cm)')
axs[0].set_xlabel('qa (cm/s)')
axs[1].set_xlabel('f')
axs[2].set_xlabel('h (cm)')
axs[3].set_xlabel('θ')
axs[1].legend()
axs[3].legend()
for ax in axs:
    ax.xaxis.set_ticks_position('top')  # 将横坐标位置设置为顶部
    ax.xaxis.set_label_position('top')  # 将横坐标标签位置设置为顶部
    ax.set_ylim(100, 0)  # 设置纵坐标刻度范围为0到10
plt.show()

#储存数据到Excel

result[0]=z
result_T=result.T
boundary_T=boundary.T
# 创建一个DataFrame
df1 = pd.DataFrame(boundary_T,columns=['t','psiT'])
#df2 = pd.DataFrame(result_T,columns=['z','qa0','qa3','qa6','qa12','qa24','f0','f3','f6','f12','f24','h0','h3','h6','h12','h24','theta0','theta3','theta6','theta12','theta24'])

# 以追加模式打开现有的Excel文件
with pd.ExcelWriter('计算数据(new).xlsx', mode='a') as writer:
    # 将数据追加到现有的Excel表中
    df1.to_excel(writer, sheet_name='Sheng2014_P2_boundary_Sa', index=False)
    #df2.to_excel(writer, sheet_name='Sheng2014_P2_Sa', index=False)