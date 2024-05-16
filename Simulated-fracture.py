
import numpy as np
import os
import nibabel as nib
import math
import random
import scipy
def f(x,y,rand1,rand2,shape):
    return shape[2]/2 + shape[2]/2*math.sin(2*(x+rand1)*math.pi/shape[0])*math.sin(2*(y+rand2)*math.pi/shape[1]) + shape[2]/200*math.sin(80*(x+rand1)*math.pi/shape[0])*math.sin(80*(y+rand2)*math.pi/shape[1])

def roat(i,j,k, tx,ty,tz, alp, beta, gamma,shape, heart):
    cod = [i,j,k,1]
    para = [tx,ty,tz, alp, beta, gamma]
    para[3] = para[3]*math.pi/180
    para[4] = para[4]*math.pi/180
    para[5] = para[5]*math.pi/180
    rx = np.eye(4)
    ry = np.eye(4)
    rz = np.eye(4)
    t = np.eye(4)
    
    t_to_zero = np.eye(4)
    t_to_zero[3] = [-heart[0], -heart[1], -heart[2], 1]
    t_to_oring = np.eye(4)
    t_to_oring[3] = [heart[0], heart[1], heart[2], 1]
    
    rx[1, 1] = rx[2, 2] = math.cos(para[3])
    rx[1, 2] = -1*math.sin(para[3])
    rx[2, 1] = math.sin(para[3])

    ry[0, 0] = ry[2, 2] = math.cos(para[4])
    ry[0, 2] = math.sin(para[4])
    ry[2, 0] = -1*math.sin(para[4])

    rz[0, 0] = rz[1, 1] = math.cos(para[5])
    rz[0, 1] = -1*math.sin(para[5])
    rz[1, 0] = math.sin(para[5])

    t[0,3] = para[0]
    t[1,3] = para[1]
    t[2,3] = para[2]

    new_cod = np.dot(np.dot(np.dot(np.dot(np.dot(cod,t_to_zero),rx),ry),rz),t_to_oring)
    new_cod = np.trunc(new_cod).astype(int)

    return new_cod[0],new_cod[1], new_cod[2]

def get_heart(data):
    nonzero_coords = np.transpose(np.nonzero(data))
    total = np.shape(nonzero_coords)[0]
    mean_nonzero = np.mean(nonzero_coords,axis=0)
    return mean_nonzero,total+1

def trans(data):
    shape = np.shape(data)
    flag =0
    times = 0
    
    while(flag == 0):

            new_data = np.zeros_like(data)
            part1 = np.zeros_like(data)
            part2 = np.zeros_like(data)
            
            rand1 = random.randint(0,2*shape[0])
            rand2 = random.randint(0,2*shape[1])

            tx = random.randint(-10,10)*shape[0]/512
            ty = random.randint(-10,10)*shape[1]/512
            tz = random.randint(-10,10)*shape[2]/512

            alp = random.randrange(-8,8)
            beta = random.randrange(-8,8)
            gamma = random.randrange(-8,8)



            for i in range(shape[0]):
                for j in range(shape[1]):
                    if(np.sum(data[i,j,:] >0)):
                        fz = f(i,j,rand1, rand2,np.shape(data))
                        for k in range(shape[2]):
                            if(data[i,j,k] >0):
                                if(k>fz):
                                    part1[i,j,k] = 1
                                else:
                                    part2[i,j,k] = 1
            
            
            
            heart1, sum_1 = get_heart(part1)
            heart2, sum_2 = get_heart(part2)
            
            if(sum_1>4*sum_2 and sum_1 < 15*sum_2):
                    for i in range(shape[0]):
                        for j in range(shape[1]):
                            if(np.sum(part2[i,j,:] >0)):
                                for k in range(shape[2]):
                                    if(part2[i,j,k] > 0 ):
                                            i2,j2,k2 = roat(i,j,k, tx,ty,tz, alp, beta, gamma, np.shape(data), heart2)
                                            if i2 >=1 and i2 <shape[0]-1 and j2>=1 and j2<shape[1]-1 and k2<shape[2]-1 and k2>=1:         
                                                new_data[i2,j2,k2] = 1 
                                                new_data[i2+1,j2,k2] = 1       
                                                new_data[i2-1,j2,k2] = 1 
                                                new_data[i2,j2+1,k2] = 1 
                                                new_data[i2,j2-1,k2] = 1 
                                                new_data[i2,j2,k2+1] = 1 
                                                new_data[i2,j2,k2-1] = 1
                    
                    part3 = new_data + part1
                    part3 = np.where(part3>1, 1.0, 0)
                    heart3, sum_3 = get_heart(part3)
                    if(sum_3<sum_1/10 and sum_3<sum_2/10):
                        flag = 1
                        data = np.where(new_data+part1>0, 1.0, 0)
                        mask = part1
                                    
            elif(sum_2>4*sum_1 and sum_2 < 15*sum_1):
                    for i in range(shape[0]):
                        for j in range(shape[1]):
                            if(np.sum(part1[i,j,:] >0)):
                                for k in range(shape[2]):
                                    if(part1[i,j,k] > 0 ):
                                            i2,j2,k2 = roat(i,j,k, tx,ty,tz, alp, beta, gamma, np.shape(data), heart1)
                                            if i2 >=1 and i2 <shape[0]-1 and j2>=1 and j2<shape[1]-1 and k2<shape[2]-1 and k2>=1:         
                                                new_data[i2,j2,k2] = 1 
                                                new_data[i2+1,j2,k2] = 1       
                                                new_data[i2-1,j2,k2] = 1 
                                                new_data[i2,j2+1,k2] = 1 
                                                new_data[i2,j2-1,k2] = 1 
                                                new_data[i2,j2,k2+1] = 1 
                                                new_data[i2,j2,k2-1] = 1
                    part3 = new_data + part2
                    part3 = np.where(part3>1, 1.0, 0)
                    heart3, sum_3 = get_heart(part3)
                    if(sum_3<sum_1/10 and sum_3<sum_2/10):
                        flag = 1
                        data = np.where(new_data+part2>0, 1.0, 0)
                        mask = part2
                        
            else:
                    print('trying....' + str(times+1))
                    times +=1
                    if(times>100):
                        flag = 1
                        mask = data.copy()



    return data, mask


path = '/data/user/temp/segment/data/dataset5_train/'
data_all = os.listdir(path)


out = "/data/user/temp/segment/data/train/"
os.makedirs(out, exist_ok=True)
        
        
        
        
        

for item in data_all:
    if(item[-3:] == '.gz'):
        
        data = nib.load(path + item)
        affine = data.affine.copy()
        data = data.get_fdata()
        
        print(item)
        print(np.shape(data))
        data = np.where(data==4,0,data)
        data = np.where(data>0, 1.0, 0)
        
        for ii in range(2):
            res, part= trans(data)
            
            pelvic_ban = nib.Nifti1Image(res, affine)
            nib.save(pelvic_ban, out+"data" +item[:-7]+"_"+str(ii)+'_'+".nii.gz")
            pelvic_ban = nib.Nifti1Image(part, affine)
            nib.save(pelvic_ban,out + "mask" +item[:-7]+"_"+str(ii)+'_'+".nii.gz")

            print(item)
