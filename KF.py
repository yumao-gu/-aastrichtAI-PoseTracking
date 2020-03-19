import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

R = np.diag([
	0.05,  			# variance of location on x-axis
    	0.05,  			# variance of location on y-axis
    	np.deg2rad(0.5),  	# variance of yaw angle
]) ** 2  			# predict state covariance

R_sqrt = np.diag([
    	0.05,  			
    	0.05,  		
    	np.deg2rad(0.5),  	
])

Q = np.diag([
	0.1, 
	0.1,
	np.deg2rad(1.0),
]) ** 2  			# Observation x,y position covariance

Q_sqrt = np.diag([
	0.1, 
	0.1,
	np.deg2rad(1.0),
]) 

DT = 0.1  # time tick [s]
SIM_TIME = 50.0  # simulation time [s]
v = 0.0  # [m/s]
yawrate = 0.0  # [rad/s]
esc_flag = False

show_animation = True

def calc_input():
    	#v = 1.0  # [m/s]
    	#yawrate = 0.1  # [rad/s]
    	u = np.array([[v], [yawrate]])
    	return u
    
def observation_model(x):
    	H = np.array([
        	[1, 0, 0],
        	[0, 1, 0],
        	[0, 0, 1],
    	])
    	z = H @ x
    	return z

def motion_model(x, u):
    	F = np.array([[1.0, 0, 0],
		[0, 1.0, 0],
                [0, 0, 1.0]])
    	B = np.array([[DT * math.cos(x[2, 0]), 0],
        	[DT * math.sin(x[2, 0]), 0],
                [0.0, DT]])
    	x = F @ x + B @ u
    	return x
    
def observation(xTrue, xd, u):
    	xTrue = motion_model(xTrue, u)
    	# add noise to input
    	xd = motion_model(xd, u) + R_sqrt @ np.random.normal(0,1,(3,1))
    	# add noise to gps x-y-theta
    	n = 0
    	z = observation_model(xTrue) + 1.0 / (n+1) * Q_sqrt @ np.random.normal(0,1,(3,1))
    	return xTrue, z, xd, n

def kf_estimation(xEst, PEst, z, u, n):
	#Q_t = 1.0 / (n+1)**2 * Q
	#R_t = abs(u[0,0]) * R
    	#  Predict
	xPred = motion_model(xEst, u)
	PPred = PEst + R
    	#  Update
	if n == 0:
    		xEst = xPred + R_sqrt @ np.random.normal(0,1,(3,1))
    		PEst = PPred
	else:
		K = PPred @ np.linalg.inv(PPred + Q)
		xEst = xPred + K @ (z - xPred)
		PEst = (np.eye(len(xEst)) - K) @ PPred
	return xEst, PEst

def plot_covariance_ellipse(xEst, PEst):  # pragma: no cover
	Pxy = PEst[0:2, 0:2]
	eigval, eigvec = np.linalg.eig(Pxy)

	if eigval[0] >= eigval[1]:
		bigind = 0
		smallind = 1
	else:
		bigind = 1
		smallind = 0

	t = np.arange(0, 2 * math.pi + 0.1, 0.1)
	a = math.sqrt(eigval[bigind])
	b = math.sqrt(eigval[smallind])
	x = [a * math.cos(it) for it in t]
	y = [b * math.sin(it) for it in t]
	angle = math.atan2(eigvec[bigind, 1], eigvec[bigind, 0])
	rot = np.array([[math.cos(angle), math.sin(angle)],
                    [-math.sin(angle), math.cos(angle)]])
	fx = rot @ (np.array([x, y]))
	px = np.array(fx[0, :] + xEst[0, 0]).flatten()
	py = np.array(fx[1, :] + xEst[1, 0]).flatten()
	plt.plot(px, py, "--r")
	
	t2 = np.arange(0, 2 * math.pi + 0.1, 0.1)
	x2 = [math.cos(it2) for it2 in t2]
	y2 = [math.sin(it2) for it2 in t2]
	fx2 = np.array([x2, y2])
	px2 = np.array(fx2[0, :] + xEst[0, 0]).flatten()
	py2 = np.array(fx2[1, :] + xEst[1, 0]).flatten()
	plt.plot(px2, py2, "--b")
	
	t3 = np.arange(0, 2 * math.pi + 0.1, 0.1)
	x3 = [0.1 * math.cos(it3) for it3 in t3]
	y3 = [0.1 * math.sin(it3) for it3 in t3]
	fx3 = np.array([x3, y3])
	px3 = np.array(fx3[0, :] + xEst[0, 0]).flatten()
	py3 = np.array(fx3[1, :] + xEst[1, 0]).flatten()
	plt.fill(px3,py3,'y')


def key_event(event):
	global v,yawrate,esc_flag
	if event.key == 'W':
		v += 0.1
	elif event.key == 'S':
		v -= 0.1
	elif event.key == 'A':
		yawrate += 0.1
	elif event.key == 'D':
		yawrate -= 0.1
	elif event.key == 'escape':
		esc_flag = True
	print(event.key)

def main():
	print(__file__ + " start!!")
	time = 0.0
	
    	# State Vector [x y theta]'
	xEst = np.zeros((3, 1))
	xTrue = np.zeros((3, 1))
	#PEst = np.eye(3)
	PEst = np.zeros((3, 1))
	xDR = np.zeros((3, 1))  # Dead reckoning
    
	# history
	hxEst = xEst
	hxTrue = xTrue
	hxDR = xTrue
	hz = np.zeros((3, 1))
    
	while SIM_TIME >= time:
		time += DT
		u = calc_input()
		xTrue, z, xDR, n = observation(xTrue, xDR, u)

		xEst, PEst = kf_estimation(xEst, PEst, z, u, n)
		print("time: " + str(time))
		'''print("PEst")
		print(PEst)
		print("xEst")
		print(xEst)'''

        	# store data history
		hxEst = np.hstack((hxEst, xEst))
		hxDR = np.hstack((hxDR, xDR))
		hxTrue = np.hstack((hxTrue, xTrue))
		hz = np.hstack((hz, z))
        
		if show_animation and not esc_flag:
			plt.cla()
            		# for stopping simulation with the esc key.
			plt.gcf().canvas.mpl_connect('key_release_event',key_event)
			plt.plot(hz[0, :], hz[1, :], ".g")
			plt.plot(hxTrue[0, :].flatten(),hxTrue[1, :].flatten(), "-b")
			plt.plot(hxDR[0, :].flatten(),hxDR[1, :].flatten(), "-k")
			plt.plot(hxEst[0, :].flatten(),hxEst[1, :].flatten(), "-r")
			plot_covariance_ellipse(xEst, PEst)
			plt.axis("equal")
			plt.grid(True)
			plt.pause(0.1)
    
	print("END!")

if __name__ == '__main__':
	main()
