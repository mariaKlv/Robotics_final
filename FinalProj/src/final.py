import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Range

from tf.transformations import euler_from_quaternion, quaternion_from_euler
from std_srvs.srv import Empty as EmptyServiceCall
from math import pow, atan2, sqrt

import random

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from numpy.random import uniform
import numpy as np
import math
import cv2

INFRONT_TRESHOLD = 0.09 #Distance between obstacle and robot
MAX_SENSOR_READING = 0.11
SENSOR_MAGNIFICATION = 100
MT_TO_CM = 100
VELOCITY = 0.1
TURN_MULTIPY = 4
PLOT_ARROW_SIZE = 100.0
class ThymioControl:
    def __init__ (self):
        #rospy.init_node('thymioController')
        rospy.init_node('thymioController')
        self.rate = rospy.Rate(10)

        #publishers
        self.vel_publisher = rospy.Publisher('/thymio10/cmd_vel', Twist, queue_size = 10)
        self.vel_message = Twist()

        #subscriptions
        self.odom_suscriber = rospy.Subscriber('/thymio10/odom', Odometry, self.update_odom)
        self.center_sensor_suscriber = rospy.Subscriber('/thymio10/proximity/center', Range, self.update_Csensors)
        self.center_sensor_suscriber = rospy.Subscriber('/thymio10/proximity/center_left', Range, self.update_CLsensors)
        self.center_sensor_suscriber = rospy.Subscriber('/thymio10/proximity/center_right', Range, self.update_CRsensors)
        self.right_sensor_suscriber = rospy.Subscriber('/thymio10/proximity/right', Range, self.update_Rsensors)
        self.left_sensor_suscriber = rospy.Subscriber('/thymio10/proximity/left', Range, self.update_Lsensors)
        self.left_sensor_suscriber = rospy.Subscriber('/thymio10/proximity/rear_left', Range, self.update_rearLsensors)
        self.left_sensor_suscriber = rospy.Subscriber('/thymio10/proximity/rear_right', Range, self.update_rearRsensors)



        #self.odometry = Odometry()
        self.pose_position = Odometry().pose.pose.position
        self.pose_orientation = Odometry().pose.pose.orientation
        
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0

        #sensors
        self.center_sensor = Range()
        self.center_left_sensor = Range()
        self.center_right_sensor = Range()
        self.right_sensor = Range()
        self.left_sensor = Range()
        self.rearL_sensor = Range()
        self.rearR_sensor = Range()


        #flags and variables

        self.timer = 0
        self.direction = 0

        self.crashC_detected = False
        self.crashCL_detected = False
        self.crashCR_detected = False
        self.crashL_detected = False
        self.crashR_detected = False
        self.crash_detected = False


    def update_Csensors(self,sen):
        self.center_sensor = round(sen.range,4)
        #print("\ncenter: " + str(self.center_sensor))
        if(self.center_sensor < INFRONT_TRESHOLD):
            self.crashC_detected = True
            self.crash_detected = True
            #self.stopRobot()
        else:
            self.crashC_detected = False

    def update_CLsensors(self,sen):
        self.center_left_sensor = round(sen.range,4)
        #print("\ncenter: " + str(self.center_sensor))
        if(self.center_left_sensor < INFRONT_TRESHOLD):
            self.crashCL_detected = True
            self.crash_detected = True
            #self.stopRobot()
        else:
            self.crashCL_detected = False

    def update_CRsensors(self,sen):
        self.center_right_sensor = round(sen.range,4)
        #print("\ncenter: " + str(self.center_sensor))
        if(self.center_right_sensor < INFRONT_TRESHOLD):
            self.crashCR_detected = True
            self.crash_detected = True
            #self.stopRobot()
        else:
            self.crashCR_detected = False

    def update_Rsensors(self,sen):
        self.right_sensor = round(sen.range,4)
        #print("right: " + str(self.right_sensor))
        if(self.right_sensor < INFRONT_TRESHOLD):
            self.crashR_detected = True
            self.crash_detected = True
            #self.stopRobot()
        else:
            self.crashR_detected = False

    def update_Lsensors(self,sen):
        self.left_sensor = round(sen.range,4)
        #print("left: " + str(self.left_sensor))
        if(self.left_sensor < INFRONT_TRESHOLD):
            self.crashL_detected = True
            self.crash_detected = True
            #self.stopRobot()
        else:
            self.crashL_detected = False

    def update_rearLsensors(self,sen):
        self.rearL_sensor = round(sen.range,4)
        if(self.rearL_sensor < INFRONT_TRESHOLD):
            pass

    def update_rearRsensors(self,sen):
        self.rearR_sensor = round(sen.range,4)
        if(self.rearR_sensor < INFRONT_TRESHOLD):
            pass


    def update_odom(self, od):
        #odometry = od
        self.pose_position = od.pose.pose.position
        self.pose_orientation = od.pose.pose.orientation

        orientation_list = [self.pose_orientation.x, self.pose_orientation.y, self.pose_orientation.z, self.pose_orientation.w]
        self.roll, self.pitch, self.yaw = euler_from_quaternion(orientation_list)


    def get_yaw(self):
        return self.yaw

    def get_pose_position(self):
        return self.pose_position

    def get_pose_orientation(self):
        return self.pose_orientation

    def reset_crash(self):
        self.crash_detected = False

    def get_crash(self):
        return self.crash_detected 

    def get_sensors_distances(self):
        return [self.right_sensor*SENSOR_MAGNIFICATION,self.center_right_sensor*SENSOR_MAGNIFICATION, \
        self.center_sensor*SENSOR_MAGNIFICATION,self.center_left_sensor*SENSOR_MAGNIFICATION,self.left_sensor*SENSOR_MAGNIFICATION]
        
    def get_center_sensor(self):
        return self.center_sensor*SENSOR_MAGNIFICATION

    def sleep(self,k):
        for i in range(k):
            self.rate.sleep()

    def stopRobot(self):
        self.vel_message.angular.z = 0
        self.vel_message.linear.x = 0
        self.vel_publisher.publish(self.vel_message)

    def look_for_not_crashing_RESPALDO(self):
        print("REVERSE")
        while self.crash_detected:
            self.vel_message.linear.x = -0.1
            self.vel_publisher.publish(self.vel_message)
            self.sleep(1)
            self.vel_message.angular.z = 0.7
            self.vel_publisher.publish(self.vel_message)
            self.sleep(1)

            if not (self.crashC_detected or self.crashCR_detected or self.crashCL_detected or self.crashR_detected or self.crashL_detected):
                print("DONE")
                self.sleep(10)
                self.vel_message.linear.x = 0.0
                self.vel_message.angular.z = 0.0
                self.vel_publisher.publish(self.vel_message)
                self.reset_crash()
                break

    def look_for_not_crashing(self):
        print("GET IN PARALLEL")

        if(self.crashR_detected or self.crashCR_detected):
            self.vel_message.angular.z = -0.7
        elif (self.crashCL_detected or self.crashL_detected):
            self.vel_message.angular.z = 0.7

        while not self.crashC_detected:
            self.vel_publisher.publish(self.vel_message)
            self.sleep(1)

        self.stopRobot()
        self.reset_crash()

    def move_back(self):
        self.vel_message.angular.z = 0.9
        self.vel_message.linear.x = -0.1
        self.vel_publisher.publish(self.vel_message)
        self.sleep(15)
        self.stopRobot()
        self.reset_crash()

    def rotate(self, target):
        kp = 0.5
        target_rad = target #target * math.pi/180
        rot = kp * (target_rad - self.yaw)
        print("Rotating: "+str(rot))
        while rot > 0.06 and (not self.crash_detected):
            rot = kp * (target_rad - self.yaw)
            self.vel_message.angular.z = rot
            self.vel_publisher.publish(self.vel_message)
            self.sleep(1)
        
        self.vel_message.angular.z = 0
        self.vel_publisher.publish(self.vel_message)

    def move(self, startPose,endPose):
        #Move
        #Robot movement difference
        sxr, syr, sthr = startPose[0], startPose[1], startPose[2]
        fxr, fyr, fthr = endPose[0], endPose[1], endPose[2]
        dthr = np.arctan2((fyr-syr), (fxr-sxr))


        euc_dist = math.sqrt((sxr - fxr) **2 + (syr - fyr)**2) 

        #rotate robot
        if( dthr != 0):
            self.rotate(dthr)

        #move robot
        if (euc_dist != 0):
            initpose_x = self.pose_position.x
            initpose_y = self.pose_position.y
            diff = 0.0
            while diff < euc_dist and (not self.crash_detected):
                self.vel_message.linear.x = 0.1
                self.vel_publisher.publish(self.vel_message)
                self.sleep(2)

                mx = self.pose_position.x
                my = self.pose_position.y
                diff = math.sqrt((initpose_x - mx) **2 + (initpose_y - my)**2) 


            self.vel_message.linear.x = 0
            self.vel_publisher.publish(self.vel_message)
            self.look_for_not_crashing()
            #self.move_back()

class Particle:
    def __init__(self,i,position, w = 0):
        self.id = i
        self.pose = [position[0],position[1],position[2]]    #x,y,theta
        self.weight = w
    def getId(self):
        return self.id
    def getPose(self):
        return self.pose
    def getWeight(self):
        return self.weight
    

class Program:
    def __init__(self):
        self.fig = None
        self.ax = None
        self.im = None
        self.n_grids = 10
        self.particle_number = 1000
        self.x_grid_size = None
        self.y_grid_size = None
        self.im_xsize = None
        self.im_ysize = None
        self.particles = []
        self.weights = []
        self.move_error_prob = 0.25
        self.sensor_noise = 0.25
        self.rndmNoise = 0.15

        self.SIM_robot = [] #!!! for simulating the robot. Store x & y
        self.obstacles = [[(0,0),(0,300)],[(0,0),(300,0)],[(0,300),(300,300)],[(300,0),(300,300)],\
            [(60,240),(240,240)],[(120,180),(300,180)],[(60,120),(180,120)],[(240,120),(300,120)],\
                [(60,60),(120,60)],\
                    [(240,240),(240,300)],[(60,180),(60,240)],[(60,0),(60,60)],[(240,0),(240,60)]] ####This is one line on the map
        self.world_boundaries = [[(0,0),(0,300)],[(0,0),(300,0)],[(0,300),(300,300)],[(300,0),(300,300)]]
                    

        self.buildWorld()
        self.initParticles(self.particle_number)
        self.SIM_SPAWN_ROBOT()
        self.plotParticles()
        print("Grid size: "+str(self.x_grid_size) + " x "+str(self.y_grid_size))

    def buildWorld(self,img_source='maze.png'):
        plt.close('all') 
        plt.clf()
        plt.cla() 
        dpi = 100

        im = Image.open(img_source)
        im = im.transpose(Image.FLIP_TOP_BOTTOM)

        self.im_xsize = im.size[0]
        self.im_ysize = im.size[1]
        fig = plt.figure(figsize=(float(self.im_xsize)/dpi,float(self.im_ysize)/dpi),dpi=dpi)
        ax = fig.add_subplot(111)

        # Remove whitespace from around the image
        fig.subplots_adjust(left=0,right=1,bottom=0,top=1)

        ax.set_xlim(0, self.im_xsize)
        ax.set_ylim(0, self.im_ysize)

        self.x_grid_size = self.im_xsize/self.n_grids
        self.y_grid_size = self.im_ysize/self.n_grids
        ax.xaxis.set_major_locator(plticker.MultipleLocator(self.x_grid_size))
        ax.yaxis.set_major_locator(plticker.MultipleLocator(self.y_grid_size))

        ax.grid(which='major', linestyle='-')

        ax.imshow(im, origin='lower')

        fig.savefig('TEST_maze.png')
        im.close()

    def initParticles(self,N):
        for i in range(N):  
            n = uniform(0, self.im_xsize)
            ny = uniform(0, self.im_ysize)
            theta = 0
            self.particles.append(Particle(i,[n,ny,theta]))

    def SIM_SPAWN_ROBOT(self):
        """For simulation only"""
        rx, ry, theta = 180, 150, 0 #240,90,1 #270,270,1 #180, 150, 1.5
        self.SIM_robot = [rx,ry,theta]

    def euclidean_dist(self,start,end):
        return math.sqrt((start[0] - end[0]) **2 + (start[1] - end[1])**2)   
    def ROBOT_GET_POSE(self):
        return self.SIM_robot

    def plotParticles(self):
        
        self.buildWorld() #clear image

        xi = []
        yi = []
        xf = []
        yf = []
        for x in self.particles:
            xii = x.pose[0]
            xi.append(xii)
            xf.append(np.cos(x.pose[2])/PLOT_ARROW_SIZE)
        for y in self.particles:
            yii = y.pose[1]
            yi.append(yii)
            yf.append(np.sin(y.pose[2])/PLOT_ARROW_SIZE)
        
        plt.quiver(xi, yi, xf, yf, angles='xy', scale_units='xy', color = 'blue')

        
        plt.savefig('TEST_maze.png')
        #plt.clf() #!!!!!!!!

        img = cv2.imread('TEST_maze.png')
        cv2.imshow('image',img)
        cv2.waitKey(1)  #!!!wait for key
    
    

    def find_nearest(self, array, value):
        """find nearest element to value in array"""
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]


    def check_for_obstacles(self,sPose, fPose):
        """See if particle can go from s to f
            Returns False if not possible to move
            Returns True if possible to move 
        """
        for obs in self.world_boundaries:
            sobs, fobs = obs[0], obs[1] #start and final points of obstacle
            sobsx, sobsy = sobs #x and y starting of obstacle
            fobsx, fobsy = fobs #x and y finish of obstacle

            
            corr = 999.0
            if(sobsy == fobsy): #Horizontal obstacle
                if(sobsx <= sPose[0] and fobsx >= sPose[0] ):
                    if(sPose[1] <= sobsy and fPose[1] >= sobsy): #robot cross the line top
                        # print("Evited by ---- TOP HORIZONTAL ----")
                        # print(obs)
                        return [sPose[0],sobsy + corr]
                    elif(sPose[1] >= sobsy and fPose[1] <= sobsy):
                        #print("Evited by ---- BTM HORIZONTAL ----")
                        corr *= -1.0
                        #print(obs)
                        return [sPose[0],sobsy + corr]

            if(sobsx == fobsx): #Verticl obstacle
                if(sobsy <= sPose[1] and fobsy >= sPose[1] ):
                    if(sPose[0] <= sobsx and fPose[0] >= sobsx): #robot cross the line
                        # print("Evited by ||| RIGHT VERTICAL |||")
                        # print(obs)
                        return [sPose[0],sobsy + corr]
                    elif (sPose[0] >= sobsx and fPose[0] <= sobsx):
                        #print("Evited by ||| LEFT VERTICAL |||")
                        corr *= -1.0
                        #print(obs)
                        return [sobsx+corr, sPose[1]]

        #you can move
        return [fPose[0], fPose[1]]

    

    def get_distance_to_all_obstacles(self, point):
        """Divide each obstacle into n_grid point
        keep the smalles distance to the given point
        return the distance for all obstacles (walls included)
        """
        if(point[0] >= self.im_xsize or point[0] <= 0 or point[1] >= self.im_ysize or point[1] <= 0):
            return 999  #Penalize outside the box points

        th = point[2]
        distances = []
        for obs in self.obstacles:
            sobs, fobs = obs[0], obs[1] #start and final points of obstacle
            sobsx, sobsy = sobs #x and y starting of obstacle
            fobsx, fobsy = fobs #x and y finish of obstacle
            
            step_walls = self.n_grids * 2
            xstep = (np.abs(fobsx-sobsx))/step_walls
            ystep = (np.abs(fobsy-sobsy))/step_walls
            xs = np.zeros(step_walls)
            ys = np.zeros(step_walls)
            if(xstep != 0):
                xs = np.array(np.arange(sobsx,fobsx,xstep))
            if(ystep != 0):
                ys = np.array(np.arange(sobsy,fobsy,ystep))

            ds = np.sqrt((point[0] - xs) **2 + (point[1] - ys)**2).tolist()
            #an = np.abs(np.arctan2((point[1] - ys),(point[0] - xs))-point[2]).tolist()
            #an = (np.abs(np.arctan2((point[1] - ys),(point[0] - xs)))-np.abs(point[2])).tolist()
            an = (np.arctan2((point[1] - ys),(point[0] - xs))-point[2]).tolist()

            an = []
            ann = (np.arctan2((point[1] - ys),(point[0] - xs))).tolist()

            for a in ann:
                if (a < 0):
                   a = a+ (2*np.pi)

                an.append(a-th)


            for i in range(2):
                m = np.min(an)
                index = an.index(m)    #get smallest angle diff
                an.remove(m)
                d = ds[index]
                distances.append(d)
            
        return distances

    def model_proposal(self, startPose, endPose):
        #Move
        self.SIM_robot = endPose

        #Robot movement difference
        sxr, syr, sthr = startPose[0], startPose[1], startPose[2]
        fxr, fyr, fthr = endPose[0], endPose[1], endPose[2]
        dxr, dyr = (fxr - sxr),(fyr - syr)
        dthr = np.arctan2((fyr-syr), (fxr-sxr))


        print("from: ("+str(sxr)+" , "+str(syr)+")")
        print("to: ("+str(fxr)+" , "+str(fyr)+")")
        print("angle: "+str(dthr))
        

        new_particles = []
        for i, p in enumerate(self.particles):
            spx, spy, spth = p.getPose()[0], p.getPose()[1], p.getPose()[2]

            normald_x = np.cos(dthr)* (5*np.random.randn())
            normald_y = np.sin(dthr)* (5*np.random.randn())
            normald_th = dthr*(5*np.random.randn())
            
            fpx = spx + (normald_x)
            fpy = spy + (normald_y)

            #Do not exceed boundaries
            fpx, fpy = self.check_for_obstacles([spx,spy],[fpx,fpy])
            

            fpth = normald_th
            
            new_particles.append(Particle(i,[fpx,fpy,fpth],p.getWeight()))

            
        
        self.particles = new_particles


    def model_correction(self, sensorDistance):
        #Observe
        new_particles = []
        ws = []
        error_sense = 0.0
        d = 0.0
        for i, p in enumerate(self.particles):
            #find all distances to obs
            distances = self.get_distance_to_all_obstacles(p.getPose())
            #find most similar distance to sensor data
            d = np.min(distances ) 
            error_sense = abs(sensorDistance - d)
            

            #sample from normal curve
            exponent = (-0.5*(error_sense)**2)/(self.sensor_noise)
            noise = (1/np.sqrt(6.28*self.sensor_noise**2))* np.exp(exponent)
            w = (self.rndmNoise * noise) + ((1-self.rndmNoise) * (1/sensorDistance))

            new_particles.append(Particle(i,p.getPose(),w))
            ws.append(w)
            

        self.weights = ws
        self.particles = new_particles

    def model_resample(self):
        new_particles = []
        n_weights = (self.weights/np.sum(self.weights))

        if(np.isnan(n_weights).any()):
            print("!!!!!!!!!!!!!!! N_WIGHTS NAN !!!!!!!!!!!!!!!!!!!")
            print(self.weights)


        arrg = np.arange(0,self.particle_number)
        idx = [np.random.choice(arrg,p=n_weights) \
                for i in range(self.particle_number)]

        for i in idx:
            new_particles.append(self.particles[i])
        self.particles = new_particles

if __name__ == "__main__":
    try:
        robot = ThymioControl()
        p = Program()
        print("Press any key to continue")
        cv2.waitKey(0)

        
        SIM_steps = np.arange(-1,1,0.1)
        moves = ['U','D','L','R']
        tempsxr = 0.0
        tempsyr = 0.0
        
        while not rospy.is_shutdown():
            print("++++++++++++++++++++++++")


            ROBOT_startingPose = robot.get_pose_position()
            sxr, syr = ROBOT_startingPose.x, ROBOT_startingPose.y
            

            #randomly move the robot
            # dx,dy = np.random.choice(SIM_steps,2)
            # dx *= (10*np.random.randn())
            # dy *= (10*np.random.randn())

            #Pick a direction to move
            r = np.random.choice(moves,1)    
            if(r == 'U'):
                dx = 0.0
                dy = 3.0
            elif(r == 'D'):
                dx = 0.0
                dy = -3.0
            elif (r == 'L'):
                dx = -3.0
                dy = 0.0
            else:
                dx = 3.0
                dy = 0.0
            print("Moving: "+str(r))
            endPose = [sxr+(dx),syr+(dy),1]

    
            #else:
            robot.move([sxr,syr,1],endPose) 
            #convert init pos
            sPose = [sxr*MT_TO_CM, syr*MT_TO_CM, 0]
            #get final pos
            rPosePos = robot.get_pose_position()
            ePose = [rPosePos.x*MT_TO_CM, rPosePos.y*MT_TO_CM, 0]

            print("sx: "+str(sxr)+" sy: "+str(syr))

            #model proposal
            p.model_proposal(sPose,ePose)

            #Sense
            sensorReading =  robot.get_center_sensor()
            
            if(sensorReading >= MAX_SENSOR_READING*SENSOR_MAGNIFICATION):
                sensorReading = MAX_SENSOR_READING*SENSOR_MAGNIFICATION #big penalty
            print(sensorReading)
            p.model_correction(sensorReading)

            #Resample
            p.model_resample()

            #refresh view
            p.plotParticles()

            if(robot.get_crash()):
                robot.move_back()

            #print("Waiting key")
            cv2.waitKey(1)

            

            


    except:
        print("Crashed")
        cv2.waitKey(0)
    finally:
        cv2.destroyAllWindows()
        print("Ended")


"""
References:
https://github.com/yz9/Monte-Carlo-Localization/blob/master/python/monte_carlo_localization_v2.py
https://github.com/nitinnat/Monte-Carlo-Localization/blob/master/Problem3_4.py
http://ais.informatik.uni-freiburg.de/teaching/ws12/practicalB/02-mcl.pdf
https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
http://ros-developer.com/2019/04/10/parcticle-filter-explained-with-python-code-from-scratch/
https://github.com/jamesjackson/ev3-localization/blob/da033fa3a3efd499581f85ab1e16b9aaf46ae93d/ev3_wall_trace_localize.py#L330

"""
