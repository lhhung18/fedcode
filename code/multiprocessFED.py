import numpy as np
import math
import time
import multiprocessing
import queue

from PIL import Image as IMG
from IPython.display import display
from skimage.metrics import peak_signal_noise_ratio as PSNR
from sklearn.metrics import mean_absolute_error as MAE
from skimage.metrics import structural_similarity as SSIM

from Image import Image

sigma = 20
maxT = sigma

def load_image(path):
    with IMG.open(path) as img:
        img.load()
        img = img.resize((251,251))
        img = np.array(img)
    return img

def add_gaussian_noise(image, mean=0, stddev=30):
    rn = np.random.RandomState(42)
    noise = rn.normal(mean, stddev, image.shape)
    noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
    return noisy_image

def best_res(image,_psnr,_mae,_ssim,_time,_stop,_m):
    best_res_image = image
    best_res_psnr = _psnr
    best_res_mae = _mae
    best_res_ssim = _ssim
    best_res_time = _time
    best_res_T = _stop
    best_res_M = _m
    # IMG.fromarray(image.astype(np.uint8)).save("./result_images/fed/"+_stop+'_'+_m+'.png')
    return best_res_image, best_res_psnr, best_res_mae, best_res_ssim, best_res_time, best_res_T, best_res_M

def print_res(_psnr,_mae,_ssim,_time,_T,_M):
    print('-------------------------------')
    print('T = ', _T)
    print('M = ', _M)
    print('Sigma = ', sigma)
    print('PSNR: ',_psnr)
    print('MAE: ', _mae)
    print('SSIM: ', _ssim)
    print('Time: ', _time)

def FED(img, noisy_image, stoptime):
    max_psnr = 0
    max_psnr_T = 0
    max_psnr_M = 0
    max_psnr_time = 0
    max_psnr_SSIM = 0
    max_psnr_MAE = 0

    hx=10
    hy=10
    image = Image()
    noisy=image.normalize(noisy_image)
    minL=min(hx,hy)
    hx=hx/minL
    hy=hy/minL
    max_val = np.max(noisy)
    min_val = np.min(noisy)
    noisy2 = np.copy(noisy)
    noisy2 = noisy2 - min_val # Shift the minimum to 0
    if(max_val != 0):
       noisy2 = noisy2/max_val # Normalize to range [0,1]

    for m in range(2,2*sigma+1):
        start_time = time.time()
        res = image.multiscale_fourth_order_anisotropic_diffusion_filter(\
                noisy, np.copy(noisy), hx, hy,\
                [0.2,0.3,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0,7.5,8.0,8.5,9.0],\
                sigma, 0.5, var_lambda=0.01,\
                T = stoptime, M = m, c_vars = [], betta_var = 0.5, theta_var = 0.13,\
                crease_type = 'r', auto_stop = False, EULER_EX_STEP = False)
        end_time = time.time()
        compute_fed = end_time - start_time
        res = res*255
        psnr = PSNR(img.astype(np.uint8), res.astype(np.uint8))
        mae = MAE(img, res)
        ssim = SSIM(img.astype(np.uint8),res.astype(np.uint8))
        if (psnr > max_psnr):
            best_img,\
            max_psnr,\
            max_psnr_MAE,\
            max_psnr_SSIM,\
            max_psnr_time,\
            max_psnr_T,\
            max_psnr_M = best_res(res,psnr,mae,ssim,compute_fed,stoptime,m)
            IMG.fromarray(best_img.astype(np.uint8)).save("./result_images/fed/barbara/sigma20/"+str(stoptime)+'_'+str(m)+'.png')
            print_res(psnr,mae,ssim,compute_fed,stoptime,m)
    return best_img,max_psnr,max_psnr_MAE,max_psnr_SSIM,max_psnr_time,max_psnr_T,max_psnr_M

def FED_filter(fedqueue):
    while True:
        try:
            '''
                try to get task from the queue. get_nowait() function will 
                raise queue.Empty exception if the queue is empty. 
                queue(False) function would do the same task also.
            '''
            task = fedqueue.get_nowait()
        except queue.Empty:

            break
        else:
            '''
                if no exception has been raised, add the task completion 
                message to task_that_are_done queue
            '''
            FED(task[0],task[1],task[2])
            # feddone.put(task + ' is done by ' + multiprocessing.current_process().name)
            # time.sleep(.5)
    return True

def main():
    path = './original_images/Barbara.png'
    img = load_image(path)
    noisy_image = add_gaussian_noise(img, stddev=sigma)
    

    numofprocesses = 10
    numoftasks = maxT
    fedqueue = multiprocessing.Queue()
    feddone = multiprocessing.Queue()
    processes = []

    for i in range(numoftasks):
        args = []
        args.append(img)
        args.append(noisy_image)
        args.append(i+1)
        fedqueue.put(args)

    for i in range(numofprocesses):
        p = multiprocessing.Process(target=FED_filter, args=(fedqueue,))
        processes.append(p)
        p.start()

    for process in processes:
        process.join()

    return True

if __name__=='__main__':
    main()