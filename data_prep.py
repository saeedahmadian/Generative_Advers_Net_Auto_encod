from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

des = np.load('desired_data.npy')
reg = np.load('regular_data.npy')
scale= StandardScaler((0,1))
des_scale= scale.fit(des).transform(des)
reg_scale= scale.fit(reg).transform(reg)
x_train_des, x_test_des, _,_ = train_test_split(des_scale,np.ones((des_scale.shape[0],1)),test_size=.3)
x_train_reg, x_test_reg, _,_ = train_test_split(reg_scale,np.ones((reg_scale.shape[0],1)),test_size=.3)

np.save('x_train_des.npy',x_train_des)
np.save('x_test_des.npy',x_test_des)
np.save('x_train_reg.npy',x_train_reg)
np.save('x_test_reg.npy',x_test_reg)