import math,ray,os,time
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.special import erf 
from scipy.spatial import distance


def cos_exp_square_nd(x):
    """
    f(x) = -cos(2*pi*||x||_2)*exp(-||x||_2)
        x: [N x d]
    """
    x_rownorm = np.linalg.norm(x,axis=1).reshape((-1,1)) # [N x 1]
    y = -np.cos(2*np.pi*x_rownorm)*np.exp(-x_rownorm**1) # [N x 1]
    return y

def x_sampler(n_sample,x_minmax):
    """
    Sample x as a list from the input domain 
    """
    x_samples = []
    for _ in range(n_sample):
        x_sample = x_minmax[:,0]+(x_minmax[:,1]-x_minmax[:,0])*np.random.rand(1,x_minmax.shape[0])
        x_samples.append(x_sample)
    return x_samples # list 

def plot_line(
    x,y,fmt='-',lc='k',lw=2,label=None,
    x2=None,y2=None,fmt2='-',lc2='k',lw2=2,ms2=12,mfc2='none',mew2=2,label2=None,
    x3=None,y3=None,fmt3='-',lc3='k',lw3=2,ms3=12,mfc3='none',mew3=2,label3=None,
    x4=None,y4=None,fmt4='-',lc4='k',lw4=2,ms4=12,mfc4='none',mew4=2,label4=None,
    x5=None,y5=None,fmt5='-',lc5='k',lw5=2,ms5=12,mfc5='none',mew5=2,label5=None,
    x6=None,y6=None,fmt6='-',lc6='k',lw6=2,ms6=12,mfc6='none',mew6=2,label6=None,
    x_fb=None,y_fb_low=None,y_fb_high=None,fba=0.1,fbc='g',labelfb=None,
    figsize=(10,5),
    xstr='',xfs=12,ystr='',yfs=12,
    tstr='',tfs=15,
    ylim=None,
    lfs=15,lloc='lower right'):
    """
    Plot a line
    """
    plt.figure(figsize=figsize)
    plt.plot(x,y,fmt,color=lc,linewidth=lw,label=label)
    if (x2 is not None):
        plt.plot(x2,y2,fmt2,color=lc2,linewidth=lw2,ms=ms2,mfc=mfc2,mew=mew2,label=label2)
    if (x3 is not None):
        plt.plot(x3,y3,fmt3,color=lc3,linewidth=lw3,ms=ms3,mfc=mfc3,mew=mew3,label=label3)
    if (x4 is not None):
        plt.plot(x4,y4,fmt4,color=lc4,linewidth=lw4,ms=ms4,mfc=mfc4,mew=mew4,label=label4)
    if (x5 is not None):
        plt.plot(x5,y5,fmt5,color=lc5,linewidth=lw5,ms=ms5,mfc=mfc5,mew=mew5,label=label5)
    if (x6 is not None):
        plt.plot(x6,y6,fmt6,color=lc6,linewidth=lw6,ms=ms6,mfc=mfc6,mew=mew6,label=label6)


    if (x_fb is not None):
        plt.fill_between(x_fb.reshape(-1),
                        (y_fb_low).reshape(-1),
                        (y_fb_high).reshape(-1),
                        alpha=fba,color=fbc,label=labelfb)

    plt.xlabel(xstr,fontsize=xfs)
    plt.ylabel(ystr,fontsize=yfs)
    plt.title(tstr,fontsize=tfs)

    if ylim is not None:
        plt.ylim(ylim[0],ylim[1])

    plt.legend(fontsize=lfs,loc=lloc)
    plt.show()

def r_sq(x1,x2,x_range=1.0,invlen=5.0):
    """
    Scaled pairwise dists 
    """
    x1_scaled,x2_scaled = invlen*x1/x_range,invlen*x2/x_range
    D_sq = distance.cdist(x1_scaled,x2_scaled,'sqeuclidean') 
    return D_sq
    
def k_m52(x1,x2,x_range=1.0,gain=1.0,invlen=5.0):
    """
    Automatic relevance determination (ARD) Matern 5/2 kernel
    """
    R_sq = r_sq(x1,x2,x_range=x_range,invlen=invlen)
    K = gain*(1+np.sqrt(5*R_sq)+(5.0/3.0)*R_sq)*np.exp(-np.sqrt(5*R_sq))
    return K

def gp_m52(x,y,x_test,gain=1.0,invlen=5.0,eps=1e-8):
    """
    Gaussian process with ARD Matern 5/2 Kernel
    """
    x_range = np.max(x,axis=0)-np.min(x,axis=0)
    k_test = k_m52(x_test,x,x_range=x_range,gain=gain,invlen=invlen)
    K = k_m52(x,x,x_range=x_range,gain=gain,invlen=invlen)
    n = x.shape[0]
    inv_K = np.linalg.inv(K+eps*np.eye(n))
    mu_y = np.mean(y)
    mu_test = np.matmul(np.matmul(k_test,inv_K),y-mu_y)+mu_y
    var_test = (gain-np.diag(np.matmul(np.matmul(k_test,inv_K),k_test.T))).reshape((-1,1))
    return mu_test,var_test

def Phi(x):
    """
    CDF of Gaussian
    """
    return (1.0 + erf(x / math.sqrt(2.0))) / 2.0

def acquisition_function(x_bo,y_bo,x_test,SCALE_Y=True,gain=1.0,invlen=5.0,eps=1e-6):
    """
    Acquisition function of Bayesian Optimization with Expected Improvement
    """
    if SCALE_Y:
        y_bo_scaled = np.copy(y_bo)
        y_bo_mean = np.mean(y_bo_scaled)
        y_bo_scaled = y_bo_scaled - y_bo_mean
        y_min,y_max = np.min(y_bo_scaled), np.max(y_bo_scaled)
        y_range = y_max - y_min
        y_bo_scaled = 2.0 * y_bo_scaled / y_range
    else:
        y_bo_scaled = np.copy(y_bo)
    
    mu_test,var_test = gp_m52(x_bo,y_bo_scaled,x_test,gain=gain,invlen=invlen,eps=eps)
    gamma = (np.min(y_bo_scaled) - mu_test)/np.sqrt(var_test)
    a_ei = 2.0 * np.sqrt(var_test) * (gamma*Phi(gamma) + norm.pdf(mu_test,0,1))
    
    if SCALE_Y:
        mu_test = 0.5 * y_range * mu_test + y_bo_mean
    
    return a_ei,mu_test,var_test

def scale_to_match_range(x_to_change,y_to_refer):
    """
    Scale the values of 'x_to_change' to match the range of 'y_to_refer'
    """
    x_to_change_scale = np.copy(x_to_change)
    xmin,xmax = np.min(x_to_change_scale),np.max(x_to_change_scale)
    ymin,ymax = np.min(y_to_refer),np.max(y_to_refer)
    x_to_change_scale = (ymax-ymin)*(x_to_change_scale-xmin)/(xmax-xmin)+ymin
    return x_to_change_scale

def get_best_xy(x_data,y_data):
    """
    Get the current best solution
    """
    min_idx = np.argmin(y_data)
    return x_data[min_idx,:].reshape((1,-1)),y_data[min_idx,:].reshape((1,-1))

def sample_from_best_voronoi_cell(x_data,y_data,x_minmax,n_sample,
                                  max_try_sbv=5000):
    """
    Sample from the Best Voronoi Cell for Voronoi Optimistic Optimization (VOO)
    """
    x_dim = x_minmax.shape[0]
    idx_min_voo = np.argmin(y_data) # index of the best x
    x_evals = []
    for _ in range(n_sample):
        n_try,x_tried,d_tried = 0,np.zeros((max_try_sbv,x_dim)),np.zeros((max_try_sbv,1))
        x_sol,_ = get_best_xy(x_data,y_data)
        while True:
            # if n_try < (max_try_sbv/2):
            if np.random.rand() < 0.5:
                x_sel = x_sampler(n_sample=1,x_minmax=x_minmax)[0] # random sample
            else:
                # Gaussian sampling centered at x_sel
                x_sel = x_sol + 0.1*np.random.randn(*x_sol.shape)*np.sqrt(1e-6+x_minmax[:,1]-x_minmax[:,0].reshape((1,-1)))
                
            dist_sel = r_sq(x_data,x_sel)
            idx_min_sel = np.argmin(dist_sel)
            if idx_min_sel == idx_min_voo: 
                break
            # Sampling the best vcell might took a lot of time 
            x_tried[n_try,:] = x_sel
            d_tried[n_try,:] = r_sq(x_data[idx_min_voo,:].reshape((1,-1)),x_sel)
            n_try += 1 # increase tick
            if n_try >= max_try_sbv:
                idx_min_tried = np.argmin(d_tried) # find the closest one 
                x_sel = x_tried[idx_min_tried,:].reshape((1,-1))
                break
        x_evals.append(x_sel) # append 
    return x_evals

def run_bavoo(
    func_eval,x_minmax,USE_RAY=True,
    n_random=1,n_bo=1,n_voo=1,n_cd=1,
    n_data_max=100,n_worker=10,seed=0,
    n_sample_for_bo=2000,max_try_sbv=5000,
    save_folder='',VERBOSE=True):
    
    """
    Run BAyesian-VOO
    """
    if USE_RAY:
        @ray.remote
        def func_eval_ray(x):
            """
            Eval with Ray
            """
            y = func_eval(x)
            return y
    
    np.random.seed(seed=seed) # fix seed 
    # First start
    x_dim = x_minmax.shape[0]
    x_evals = x_sampler(n_sample=n_worker,x_minmax=x_minmax)
    if USE_RAY:
        evals = [func_eval_ray.remote(x=x_eval) for x_eval in x_evals]
        y_evals = ray.get(evals)
    else:
        y_evals = [func_eval(x=x_eval) for x_eval in x_evals] 
    x_data,y_data = np.asarray(x_evals)[:,0,:],np.asarray(y_evals)[:,0,:]
    iclk_total = time.time()
    
    if VERBOSE:
        print ( "\nStart Bayesian VOO with [%d] Workers."%(n_worker) )
        print ( " x_dim:[%d] n_random:[%d] n_bo:[%d] n_cd:[%d]."%(x_dim,n_random,n_bo,n_cd) )
        print ( " seed:[%d] n_sample_for_bo:[%d]."%(seed,n_sample_for_bo) )
        if save_folder:
            print ( " Optimization results will be saved to [%s]."%(save_folder) )
        print ( "" )
    
    while True:
        # Random sample
        iclk_random = time.time()
        for _ in range(n_random):
            x_evals = x_sampler(n_sample=n_worker,x_minmax=x_minmax)
            if USE_RAY:
                evals = [func_eval_ray.remote(x=x_eval) for x_eval in x_evals] # EVALUATE
                y_evals = ray.get(evals)
            else:
                y_evals = [func_eval(x=x_eval) for x_eval in x_evals] 
            x_random,y_random = np.asarray(x_evals)[:,0,:],np.asarray(y_evals)[:,0,:]
            x_data = np.concatenate((x_data,x_random),axis=0)
            y_data = np.concatenate((y_data,y_random),axis=0)
        esec_random = time.time() - iclk_random
        esec_total = time.time() - iclk_total
        # Plot Random samples
        if n_random > 0:
            x_sol,y_sol = get_best_xy(x_data,y_data)
            if VERBOSE:
                print("[%.1f]sec [%d/%d] RS took [%.1f]sec. Current best x:%s best y:[%.3f]"%
                    (esec_total,x_data.shape[0],n_data_max,esec_random,x_sol,y_sol))
        # Terminate Condition
        if x_data.shape[0] >= n_data_max: break

        # Bayesian Optimization
        iclk_bo = time.time()
        for _ in range(n_bo):
            # Constant liar model for parallelizing BO
            x_evals,x_data_copy,y_data_copy = [],np.copy(x_data),np.copy(y_data)
            for _ in range(n_worker):
                x_checks = np.asarray(x_sampler(n_sample_for_bo,x_minmax=x_minmax))[:,0,:]
                a_ei,mu_checks,_ = acquisition_function(
                    x_data_copy,y_data_copy,x_checks,gain=1.0,invlen=5.0,eps=1e-6) # get the acquisition values 
                max_idx = np.argmax(a_ei) # select the one with the highested value 
                # As we cannot get the actual y_eval from the real evaluation, we use the constant liar model
                # that uses the GP mean to approximate the actual evaluation value. 
                x_liar,y_liar = x_checks[max_idx,:].reshape((1,-1)),mu_checks[max_idx].reshape((1,-1))
                # Append
                x_data_copy = np.concatenate((x_data_copy,x_liar),axis=0)
                y_data_copy = np.concatenate((y_data_copy,y_liar),axis=0)
                x_evals.append(x_liar) # append the inputs to evaluate 
                
            # Evaluate k candidates in one scoop 
            if USE_RAY:
                evals = [func_eval_ray.remote(x=x_eval) for x_eval in x_evals] # EVALUATE
                y_evals = ray.get(evals)
            else:
                y_evals = [func_eval(x=x_eval) for x_eval in x_evals] 
            x_bo = np.asarray(x_evals)[:,0,:]
            y_bo = np.asarray(y_evals)[:,0,:]
            # Concatenate BO results
            x_data = np.concatenate((x_data,x_bo),axis=0)
            y_data = np.concatenate((y_data,y_bo),axis=0)
        esec_bo = time.time() - iclk_bo
        esec_total = time.time() - iclk_total
        # Plot BO
        if n_bo > 0:
            x_sol,y_sol = get_best_xy(x_data,y_data)
            if VERBOSE:
                print("[%.1f]sec [%d/%d] BO took [%.1f]sec. Current best x:%s best y:[%.3f]"%
                    (esec_total,x_data.shape[0],n_data_max,esec_bo,x_sol,y_sol))
        # Terminate Condition
        if x_data.shape[0] >= n_data_max: break
        
        # Voronoi Optimistic Optimization
        iclk_voo = time.time()
        for _ in range(n_voo):
            # Get input points to eval from sampling the best Voronoi cell 
            x_evals = sample_from_best_voronoi_cell(
                x_data,y_data,x_minmax,n_sample=n_worker,max_try_sbv=max_try_sbv)
            # Evaluate
            if USE_RAY:
                evals = [func_eval_ray.remote(x=x_eval) for x_eval in x_evals] # EVALUATE
                y_evals = ray.get(evals)
            else:
                y_evals = [func_eval(x=x_eval) for x_eval in x_evals] 
            x_sbv,y_sbv = np.asarray(x_evals)[:,0,:],np.asarray(y_evals)[:,0,:]
            x_data = np.concatenate((x_data,x_sbv),axis=0)
            y_data = np.concatenate((y_data,y_sbv),axis=0)
        esec_voo = time.time() - iclk_voo
        esec_total = time.time() - iclk_total
        # Plot VOO
        if n_voo > 0:
            x_sol,y_sol = get_best_xy(x_data,y_data)
            if VERBOSE:
                print("[%.1f]sec [%d/%d] VOO took [%.1f]sec. Current best x:%s best y:[%.3f]"%
                    (esec_total,x_data.shape[0],n_data_max,esec_voo,x_sol,y_sol))
        

        # Coordinate Descent 
        iclk_cd = time.time()
        for _ in range(n_cd):
            x_sol,y_sol = get_best_xy(x_data,y_data)
            # for d_idx in range(x_dim): # for each dim
            if True:
                d_idx = np.random.permutation(x_dim)[0] # select one coordinate at a time 
                x_minmax_d = x_minmax[d_idx,:]
                x_sample_d = x_minmax_d[0]+(x_minmax_d[1]-x_minmax_d[0])*np.random.rand(n_worker)
                x_sample_d[0] = x_sol[0,d_idx]
                x_temp,x_evals = x_sol,[]
                for i_idx in range(n_worker):
                    x_temp[0,d_idx] = x_sample_d[i_idx]
                    x_evals.append(np.copy(x_temp.reshape((1,-1))))
                # Evaluate k candidates in one scoop
                if USE_RAY:
                    evals = [func_eval_ray.remote(x=x_eval) for x_eval in x_evals] # EVALUATE
                    y_evals = ray.get(evals)
                else:
                    y_evals = [func_eval(x=x_eval) for x_eval in x_evals] 
                # Update the current coordinate
                min_idx = np.argmin(np.asarray(y_evals)[:,0,0])
                x_sol[0,d_idx] = x_sample_d[min_idx]
                # Concatenate CD results
                x_cd,y_cd = np.asarray(x_evals)[:,0,:],np.asarray(y_evals)[:,0,:]
                x_data = np.concatenate((x_data,x_cd),axis=0)
                y_data = np.concatenate((y_data,y_cd),axis=0)
        esec_cd = time.time() - iclk_cd
        esec_total = time.time() - iclk_total
        # Plot CD
        if n_cd > 0:
            x_sol,y_sol = get_best_xy(x_data,y_data)
            if VERBOSE:
                print("[%.1f]sec [%d/%d] CD took [%.1f]sec. Current best x:%s best y:[%.3f]"%
                    (esec_total,x_data.shape[0],n_data_max,esec_cd,x_sol,y_sol))
        
        # Save intermediate resutls
        if save_folder:
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
                print ( "[%s] created."%(save_folder) )
            # Save
            npz_path = os.path.join(save_folder,'bavoo_result.npz')
            np.savez(npz_path, x_data=x_data,y_data=y_data,
                     x_minmax=x_minmax,n_random=n_random,n_bo=n_bo,n_cd=n_cd,
                     n_data_max=n_data_max,n_worker=n_worker,seed=seed,
                     n_sample_for_bo=n_sample_for_bo)
            print ( "[%s] saved."%(npz_path) )
            
        # Terminate Condition
        if x_data.shape[0] >= n_data_max: break
            
    return x_data,y_data
    
    