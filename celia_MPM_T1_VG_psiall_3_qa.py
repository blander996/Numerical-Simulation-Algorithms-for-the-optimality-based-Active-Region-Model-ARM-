import numpy as np
from numba import jit
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
#@jit(nopython=True)
def qfun(psiiter,K,Kmid,dz):
    qa=-K[1:-1]*((psiiter[2:]-psiiter[:-2])/(2*dz)-1)
    qa0=np.array([-Kmid[0]*((psiiter[1]-psiiter[0])/dz-1)])
    qaN=np.array([-Kmid[-1]*((psiiter[-1]-psiiter[-2])/dz-1)])
    qaN_0=np.hstack((qa0,qa,qaN))
    #qaN_0[qaN_0<=0]=0.00001
    qa_smooth=gaussian_filter1d(qaN_0, sigma=3)
    #plt.plot(qa_smooth)
    #plt.show()
    return qa_smooth

@jit(nopython=True)
def fSafun(pars,Sa):
    f=Sa**(pars['a0']/(1-pars['a0']))
    return  f

#@jit(nopython=True)
def fqafun(pars,qa,Ks):
    #f=(qa/(1*Ks))**(pars['a0']/(1-pars['a0']))
    f=abs(qa / (1 * Ks))**(pars['a0'] / (1-pars['a0']))
    f[qa>Ks]=1
    f[qa<0]=0
    f_smooth=gaussian_filter1d(f, sigma=3)
    return  f_smooth

@jit(nopython=True)
def Safun(psi,pars):
    Sa=(1+(psi*-pars['alpha'])**pars['n'])**(-pars['m'])
    Sa[psi>0.]=1.0
    return Sa

@jit(nopython=True)
def thetafun(psi,pars):
    Sa=Safun(psi,pars)
    return pars['thetaR']+(pars['thetaS']-pars['thetaR'])*Sa

@jit(nopython=True)
def Cfun(psi,pars):
    Sa=Safun(psi,pars)
    theta=pars['thetaR']+(pars['thetaS']-pars['thetaR'])*Sa
    dSedh=pars['alpha']*pars['m']/(1-pars['m'])*Sa**(1/pars['m'])*(1-Sa**(1/pars['m']))**pars['m']
    return (theta/pars['thetaS']*pars['Ss']+(pars['thetaS']-pars['thetaR'])*dSedh)

@jit(nopython=True)
def Kfun(psi,pars,Ks):
    Sa=Safun(psi,pars)
    return Ks*Sa**pars['neta']*(1-(1-Sa**(1/pars['m']))**pars['m'])**2

@jit(nopython=True)
#求解三对角矩阵方程，a、b、c、d为同的长度的数组，分别对应下对角、主对角、上对角和常数项。
def thomas_algorithm(a,b,c,d):
    n=len(d)
    c_=np.zeros(n-1)
    d_=np.zeros(n)
    x=np.zeros(n)

    c_[0]=c[0]/b[0]
    #print(c_[0])
    d_[0]=d[0]/b[0]

    for i in range(1,n-1):
        #print(c_[i-1])
        m=1/(b[i]-a[i]*c_[i-1])#有0值
        c_[i]=c[i]*m
        d_[i]=(d[i]-a[i]*d_[i-1])*m

    d_[n-1]=(d[n-1]-a[n-1]*d_[n-2])/(b[n-1]-a[n-1]*c_[n-2])
    x[n-1]=d_[n-1]
    for i in range(n-2,-1,-1):
        x[i]=d_[i]-c_[i]*x[i+1]
    return x

@jit(nopython=True)
def abchfun(C,Kmid,dt,dz,f,fmid):
    # Construct matrix
    a1=-fmid[:-1]*Kmid[:-1]/dz**2#长度为（n-2）
    b1=f[1:-1]*C[1:-1]/dt+(fmid[:-1]*Kmid[:-1]+fmid[1:]*Kmid[1:])/dz**2#长度为（n-2）
    c1=-fmid[1:]*Kmid[1:]/dz**2#长度为（n-2）
    an=np.array([0])#下对角底部边界条件(h)
    a=np.hstack((np.array([0]),a1,an))
    b0=np.array([1])#主对角顶部边界条件(h)
    #b0=f[0]*C[0]/dt+2*fmid[0]*Kmid[0]/dz**2#主对角顶部边界条件(qs)
    bn=np.array([1])#主对角底部边界条件(h)
    b=np.hstack((b0,b1,bn))
    c0=np.array([0])#上对角顶部边界条件(h)
    #c0=-2*fmid[0]*Kmid[0]/dz**2#主对角顶部边界条件(qs)
    c=np.hstack((c0,c1,np.array([0])))
    return a,b,c

@jit(nopython=True)
#常数项计算函数（h）
def Rhfun(psiiter,psiin,psiT,psiB,pars,Kmid,dtheta,dt,dz,f,fmid,df,C,θi):
    # This solves the Picard residual term:
    theta=thetafun(psiin, pars)
    x1=-f[1:-1]/dt*(dtheta[1:-1]-C[1:-1]*psiiter[1:-1])
    x2=-(fmid[1:]*Kmid[1:]-fmid[:-1]*Kmid[:-1])/dz
    x3=-(theta[1:-1]-θi[1:-1])*df[1:-1]/dt
    R1=x1+x2+x3
    R0=psiT#常数项顶部边界条件(h)
    Rn=psiB#常数项顶部边界条件(h)
    R=np.hstack((R0,R1,Rn))
    return R

@jit(nopython=True)
def abcqfun(C,Kmid,K,dt,dz,f,fmid):
    # Construct matrix
    a1=-fmid[:-1]*Kmid[:-1]/dz**2#长度为（n-2）
    b1=f[1:-1]*C[1:-1]/dt+(fmid[:-1]*Kmid[:-1]+fmid[1:]*Kmid[1:])/dz**2#长度为（n-2）
    c1=-fmid[1:]*Kmid[1:]/dz**2#长度为（n-2）
    an=np.array([0])#下对角底部边界条件(h)
    a=np.hstack((np.array([0]),a1,an))
    b0=f[0]*C[0]/dt+2*fmid[0]*Kmid[0]/dz**2#主对角顶部边界条件(qs)
    #b0=f[0]*C[0]/dt+(fmid[0]*Kmid[0]+f[0]*K[0])/dz**2#预设边界有-1
    bn=np.array([1])#主对角底部边界条件(h)
    b=np.hstack((np.array([b0]),b1,bn))
    c0=-2*fmid[0]*Kmid[0]/dz**2#主对角顶部边界条件(qs)
    #c0=-(fmid[0]*Kmid[0]+f[0]*K[0])/dz**2#预设边界有-1
    c=np.hstack((np.array([c0]),c1,np.array([0])))
    return a,b,c


@jit(nopython=True)
#常数项计算函数（qs）
def Rqfun(psiiter,psiin,psiT,psiB,pars,Kmid,K,dtheta,dt,dz,f,fmid,df,C,θi):
    # This solves the Picard residual term:
    theta=thetafun(psiin, pars)
    x1=-f[1:-1]/dt*(dtheta[1:-1]-C[1:-1]*psiiter[1:-1])
    #print((dtheta[1:-1]-C[1:-1] * psiiter[1:-1]))
    x2=-(fmid[1:]*Kmid[1:]-fmid[:-1]*Kmid[:-1])/dz
    x3=-(theta[1:-1]-θi[1:-1])*df[1:-1]/dt
    R1=x1+x2+x3
    R0=-f[0]/dt*(dtheta[0]-C[0]*psiiter[0])-2*fmid[0]*Kmid[0]/dz+2*psiT/dz-(theta[0]-θi[0])*df[0]/dt#常数项顶部边界条件(qs)
    #R0=-f[0] / dt * (dtheta[0]-C[0] * psiiter[0])-fmid[0] * Kmid[0] / dz-f[0]*K[0]/dz+2*psiT/dz-(theta[0]-pars['thetaI']) * df[0] / dt
    Rn=psiB#常数项顶部边界条件(h)
    R=np.hstack((R0,R1,Rn))
    return R



#@jit(nopython=True)
def iterhfun(psiin,fin,finin,pars,Ks,psiT,psiB,dt,dz,z,θi):
    # psiin = psi^n
    # psiiter = psi^n+1,m
    # psiout = psi^n+1,m+1

    tolerance=1e-10
    maxcount=500
    Rmax=1.

    # Initialize arrays
    psiiter=np.zeros(len(psiin))
    #psiout=np.zeros(len(psiin))

    # Initial guess: psi_n+1^1 = psi_n
    psiiter[:]=psiin[:]

    count=0.
    while count <= 1 or (Rmax >= tolerance and count<= maxcount):
        # Get C,K:
        C=Cfun(psiiter, pars)
        K=Kfun(psiiter, pars,Ks)
        Kmid=(K[1:]+K[:-1]) / 2.
        #Sa=Safun(psiiter,pars)
        #fiter=fSafun(pars, Sa)
        qa=qfun(psiiter, K, Kmid, dz)
        fiter=fqafun(pars,qa,Ks)
        fmax=np.maximum(fiter, fin)
        #fmax=finin
        fmid=(fmax[1:]+fmax[:-1])/2.
        dtheta=thetafun(psiiter, pars)-thetafun(psiin, pars)
        df=fin-finin
        # Get a,b,c,
        a,b,c=abchfun(C,Kmid,dt,dz,fmax,fmid)
        # Get R
        R=Rhfun(psiiter,psiin,psiT,psiB,pars,Kmid,dtheta,dt,dz,fmax,fmid,df,C,θi)
        # Solve for del
        dell=thomas_algorithm(a,b,c,R)
        #A=np.diag(a[1:],-1)+np.diag(b,0)+np.diag(c[:-1],1)
        # Solve:
        #dell = np.linalg.solve(A, R)
        # Update psi estimates at different iteration levels
        Rmax = np.abs(np.max(dell-psiiter))
        psiiter[:]=dell[:]
        #psiout[:]=psiiter[:]

        count+=1

    print(count)
    #print(Rmax)
    #print('')
    #print('Iteration count = %d'%(count-1))
    return psiiter,fmax,qa

#@jit(nopython=True)
def iterqfun(psiin,fin,finin,pars,Ks,psiT,psiB,dt,dz,z,θi):
    # psiin = psi^n
    # psiiter = psi^n+1,m
    # psiout = psi^n+1,m+1

    tolerance=1e-10
    maxcount=500
    Rmax=1.

    # Initialize arrays
    psiiter=np.zeros(len(psiin))
    #psiout=np.zeros(len(psiin))

    # Initial guess: psi_n+1^1 = psi_n
    psiiter[:]=psiin[:]

    count=0.
    while count <= 1 or (Rmax >= tolerance and count<= maxcount):
        # Get C,K:
        C=Cfun(psiiter, pars)
        K=Kfun(psiiter, pars,Ks)
        Kmid=(K[1:]+K[:-1]) / 2.
        #Sa=Safun(psiiter,pars)
        #fiter=fSafun(pars, Sa)
        qa=qfun(psiiter, K, Kmid, dz)
        fiter=fqafun(pars, qa,Ks)
        fmax=np.maximum(fiter, fin)
        #fmax=finin
        fmid=(fmax[1:]+fmax[:-1])/2.
        dtheta=thetafun(psiiter, pars)-thetafun(psiin, pars)
        df=fin-finin
        # Get a,b,c,
        a,b,c=abcqfun(C,Kmid,K,dt,dz,fmax,fmid)
        # Get R
        R=Rqfun(psiiter,psiin,psiT,psiB,pars,Kmid,K,dtheta,dt,dz,fmax,fmid,df,C,θi)
        # Solve for del
        dell=thomas_algorithm(a,b,c,R)
        #A=np.diag(a[1:],-1)+np.diag(b,0)+np.diag(c[:-1],1)
        # Solve:
        #dell = np.linalg.solve(A, R)
        # Update psi estimates at different iteration levels
        Rmax = np.abs(np.max(dell-psiiter))
        psiiter[:]=dell[:]
        #psiout[:]=psiiter[:]
        count+=1



    print(count)
    #print(Rmax)
    #print('')
    #print('Iteration count = %d'%(count-1))
    return psiiter,fmax,qa

@jit(nopython=True)
def massbal(psi,psiT,psiB,pars,n,dt,dz):

    # Initial storage:
    theta=thetafun(psi.reshape(-1),pars)
    theta=np.reshape(theta,psi.shape)
    S=np.sum(theta*dz,1)
    S0=S[0]
    SN=S[-1]

    # Inflow:
    Kin=(Kfun(psiB,pars)+Kfun(psi[:,0],pars))/2.
    QIN=-Kin*((psi[:,0]-psiB)/dz+1.)
    QIN[0]=0.
    QINsum=np.sum(QIN)*dt

    # Outflow:
    Kout=(Kfun(psi[:,-1],pars)+Kfun(psiT,pars))/2.
    QOUT=-Kout*((psiT-psi[:,-1])/dz+1.)
    QOUT[0]=0.
    QOUTsum=np.sum(QOUT)*dt

    # Balance:
    dS=SN-S0
    dQ=QINsum-QOUTsum
    err=dS/dQ
    
#    print('Delta storage = %.3f'%dS)
#    print('Flow at base (+ve upwards) = %.3f'%QINsum)
#    print('Flow at surface (+ve upwards) = %.3f'%QOUTsum)
#    print('Delta flow = %.3f'%dQ)
#    print('Error metric = %.3f'%err)
    return QIN,QOUT,S,err

#@jit(nopython=True)
def ModelRun(dt,dz,z,nt,th_q,tq_h,psi,fmax,qa,psiB,psiTop,pars,Ks,θi):
    # Solve:
    for j in range(1,nt):
        #print(j)
        psiT=np.array([psiTop[j]])
        if j==1:
            #Sa=Safun(psi[j,:], pars)
            #finin=fSafun(pars, Sa)
            finin = fmax[0,:]
            psi[j,:],fmax[j,:],qa[j,:]=iterhfun(psi[j-1,:],fmax[j-1,:],finin,pars,Ks,psiT,psiB,dt,dz,z,θi)
        elif j<=th_q:
            psi[j,:],fmax[j,:],qa[j,:]=iterhfun(psi[j-1,:],fmax[j-1,:],fmax[j-2,:],pars,Ks,psiT,psiB,dt,dz,z,θi)
        elif j<=tq_h:
            psi[j,:],fmax[j,:],qa[j,:]=iterqfun(psi[j-1,:],fmax[j-1,:],fmax[j-2,:],pars,Ks,psiT,psiB,dt,dz,z,θi)
        else:
            psi[j,:],fmax[j,:],qa[j,:]=iterhfun(psi[j-1,:],fmax[j-1,:],fmax[j-2,:],pars,Ks,psiT,psiB,dt,dz,z,θi)
    #QIN,QOUT,S,err=massbal(psi,psiT,psiB,pars,n,dt,dz)

    #QIN=0.
    #QOUT=0.
    #S=0.
    #err=0.

    return psi,fmax,qa#,QIN,QOUT,S,err
