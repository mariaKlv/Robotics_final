from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from numpy.random import uniform
import numpy as np
import math
import cv2

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
        self.particle_number = 200
        self.x_grid_size = None
        self.y_grid_size = None
        self.im_xsize = None
        self.im_ysize = None
        self.particles = []
        self.weights = []
        self.move_error_prob = 0.1
        self.sensor_noise = 0.1
        self.rndmNoise = 0.1
        #self.obstacles = [(135,180),(165,180),(195,180),(225,180),(255,180),(285,180)] #assuming robot always look up (1dim test)
        self.SIM_robot = [] #!!! for simulating the robot. Store x & y
        self.obstacles = [[(0,0),(0,300)],[(0,0),(300,0)],[(0,300),(300,300)],[(300,0),(300,300)],\
            [(60,240),(240,240)],[(120,180),(300,180)],[(60,120),(180,120)],[(240,120),(300,120)],\
                [(60,60),(120,60)],\
                    [(240,240),(240,300)],[(60,180),(60,240)],[(60,0),(60,60)],[(240,0),(240,60)]] ####This is one line on the map

        self.buildWorld()
        self.initParticles(self.particle_number)
        self.SIM_SPAWN_ROBOT()
        self.plotParticles()
        print("Grid size: "+str(self.x_grid_size) + " x "+str(self.y_grid_size))

    def buildWorld(self,img_source='maze.png'):
        plt.clf()
        dpi = 100
        #n_grids = 5

        im = Image.open(img_source)
        #im = plt.imread(img_source)
        im = im.transpose(Image.FLIP_TOP_BOTTOM)

        self.im_xsize = im.size[0]
        self.im_ysize = im.size[1]
        fig = plt.figure(figsize=(float(self.im_xsize)/dpi,float(self.im_ysize)/dpi),dpi=dpi)
        ax = fig.add_subplot(111)

        # Remove whitespace from around the image
        fig.subplots_adjust(left=0,right=1,bottom=0,top=1)

        ax.set_xlim(0, self.im_xsize)
        ax.set_ylim(0, self.im_ysize)
        print("SIZE: "+str(self.im_xsize)+" : "+str(self.im_ysize))

        self.x_grid_size = self.im_xsize/self.n_grids
        self.y_grid_size = self.im_ysize/self.n_grids
        ax.xaxis.set_major_locator(plticker.MultipleLocator(self.x_grid_size))
        ax.yaxis.set_major_locator(plticker.MultipleLocator(self.y_grid_size))

        ax.grid(which='major', linestyle='-')

        ax.imshow(im, origin='lower')

        fig.savefig('TEST_maze.png')
        #im.show()
        im.close()
    def initParticles(self,N):
        for i in range(N):  
            #n = np.random.random()
            #n = n*300               #!!!!!!!!!!!!REMOVE RESTRICTION
            n = uniform(0, self.im_xsize)
            #ny = np.random.random()
            #ny = (ny*30)+150        #!!!!!!!!!!!!
            ny = uniform(0, self.im_ysize)
            theta = 1.57
            self.particles.append(Particle(i,[n,ny,theta]))

    def SIM_SPAWN_ROBOT(self):
        """For simulation only"""
        # #spawn robot
        # rx = np.random.random()
        # rx = rx*300
        # ry = np.random.random()
        # ry = (ry*30)+150
        # theta = 1.57 #rads for 90 deg
        rx, ry, theta = 100, 140, 1 #280, 150, 1.5
        self.SIM_robot = [rx,ry,theta]
    def euclidean_dist(self,start,end):
        return math.sqrt((start[0] - end[0]) **2 + (start[1] - end[1])**2)   
    def ROBOT_GET_POSE(self):
        return self.SIM_robot
    def plotParticles(self):
        
        self.buildWorld() #clear image

        xi = []
        yi = []
        for x in self.particles:
            xi.append(x.pose[0])
        for y in self.particles:
            yi.append(y.pose[1])
        
        plt.scatter(xi,yi, c='blue', s=4)
        plt.scatter(self.SIM_robot[0],self.SIM_robot[1], c='red',s=8)
        
        
        plt.savefig('TEST_maze.png')

        img = cv2.imread('TEST_maze.png')
        cv2.imshow('image',img)
        cv2.waitKey(1)  #!!!wait for key
        #plt.show()

        print("SIM_ROBOT coord: "+str(self.SIM_robot[0])+" , "+str(self.SIM_robot[1])+" Theta: "+str(self.SIM_robot[2]))
    
    def point_to_map(self):
        """map a particle to a fixed grid square"""
        #get mapping for x and y
        mapx = []
        for p in self.particles:
            ix = p.getPose()[0]
            iy = p.getPose()[1]

            m = math.floor((float(self.im_xsize) - ix)/self.x_grid_size)
            m = (self.n_grids-1) - m
            
            m1 = math.floor((float(self.im_ysize) - iy)/self.y_grid_size)
            m1 = (self.n_grids-1) - m1

            mapx.append((m,m1))

        return mapx

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
        for obs in self.obstacles:
            sobs, fobs = obs[0], obs[1] #start and final points of obstacle
            sobsx, sobsy = sobs #x and y starting of obstacle
            fobsx, fobsy = fobs #x and y finish of obstacle

            # eu_Pose = self.euclidean_dist([sPose[0],sPose[1]],[fPose[0],fPose[1]])
            # eu_robot_obsI = self.euclidean_dist([sPose[0],sPose[1]],[sobsx,sobsy])
            # eu_robot_obsF = self.euclidean_dist([sPose[0],sPose[1]],[fobsx,fobsy])
            # if(eu_Pose >= eu_robot_obsI) or (eu_Pose >= eu_robot_obsF):
            #     print("Evited by")
            #     print(obs)
            #     return
            
            if(sobsy == fobsy): #Horizontal obstacle
                if(sobsx <= sPose[0] and fobsx >= sPose[0] )and\
                    (sPose[1] <= sobsy and fPose[1] >= sobsy): #robot cross the line
                    print("Evited by HORIZONTAL")
                    print(obs)
                    return False
            if(sobsx == fobsx): #Verticl obstacle
                if(sobsy <= sPose[1] and fobsy >= sPose[1] )and\
                    (sPose[0] <= sobsx and fPose[0] >= sobsx): #robot cross the line
                    print("Evited by VERTICAL")
                    print(obs)
                    return False

        #you can move
        return True

        #!!! TESTING
        #self.SIM_robot = fPose
    def get_distance_to_all_obstacles(self, point):
        """Divide each obstacle into n_grid point
        keep the smalles distance to the given point
        return the distance for all obstacles (walls included)
        """
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

            ds = np.sqrt((point[0] - xs) **2 + (point[1] - ys)**2)
            for d in ds:
                distances.append(d)
        return distances
            #d = self.find_nearest(distances,)

    def model_proposal(self, startPose, endPose):
        #Move
        self.SIM_robot = endPose
        #Robot movement difference
        sxr, syr, sthr = startPose[0], startPose[1], startPose[2]
        fxr, fyr, fthr = endPose[0], endPose[1], endPose[2]
        dxr, dyr = (fxr - sxr),(fyr - syr)
        dthr = np.arctan2((fyr-syr), (fxr-sxr))

        new_particles = []
        for i, p in enumerate(self.particles):
            spx, spy, spth = p.getPose()[0], p.getPose()[1], p.getPose()[2]
            fpx = spx + np.cos(dthr)* (10*np.random.randn())#dxr #try to move as the robot moved
            fpy = spy + np.cos(dthr)* (10*np.random.randn())#dyr
            fpth = spth #!!! Change theta for a not fixed one

            #if not (self.check_for_obstacles([spx,spy,spth],[fpx,fpy,fpth])):
            #   fpx,fpy, fpth = spx, spy, spth
            
            new_particles.append(Particle(i,[fpx,fpy,fpth],p.getWeight()))
        
        self.particles = new_particles
        #Plot new particles
        #self.plotParticles()

    def model_correction(self, sensorDistance):
        #Observe
        new_particles = []
        ws = []
        for i, p in enumerate(self.particles):
            #find all distances to obs
            distances = self.get_distance_to_all_obstacles(p.getPose())
            #find most similar distance to sensor data
            d = np.min(distances ) #!!!TESTING
            #d = self.find_nearest(distances,sensorDistance)

            error_sense = abs(sensorDistance - d)
            # w = (np.exp(- (error_sense ** 2) / (self.sensor_noise ** 2) / 2.0) /
            #           sqrt(2.0 * pi * (self.sensor_noise ** 2)))

            #Normal dist of error
            # w = (np.exp(-(error_sense ** 2)/(2.0*self.sensor_noise ** 2)))/ \
            #     (self.sensor_noise*np.sqrt(6.28))

            exponent = (-0.5*(error_sense)**2)/(self.sensor_noise**2)
            noise = (1/np.sqrt(6.28*self.sensor_noise**2))* np.exp(exponent)
            w = self.rndmNoise * noise + (1-self.rndmNoise) * (1/sensorDistance)

            #update particles
            new_particles.append(Particle(i,p.getPose(),w))
            ws.append(w)
        self.weights = ws
        self.particles = new_particles

    def model_resample(self):
        new_particles = []
        n_weights = self.weights/np.sum(self.weights)

        idx = [np.random.choice(np.arange(0,self.particle_number),p=n_weights) \
                for i in range(self.particle_number)]

        for i in idx:
            new_particles.append(self.particles[i])
        self.particles = new_particles

if __name__ == "__main__":
    
    p = Program()
    
    SIM_steps = np.arange(-1,1,0.1)
    cv2.waitKey(0)
    for x in range(300):
        #get coordinates for particles
        #print("\n Particle on map")
        #print(p.point_to_map())

        #Movement
        startPose = p.ROBOT_GET_POSE()
        #Move robot #!!! For testing user input 
        print("Your pose is: ")
        print(startPose)
        #endPose =   list(map(float,input("\nMove robot to coordinate: ").strip().split()))[:3] 
        #endPose = [startPose[0]-1,170,1.5]

        endPose = startPose
        dx,dy = np.random.choice(SIM_steps,2)
        dx *= (8*np.random.randn())
        dy *= (8*np.random.randn())
        
        tPose = [startPose[0]+(dx),startPose[1]+(dy),1.5]
        if(p.check_for_obstacles(startPose,tPose)): #try to move at random
            endPose = tPose

        p.model_proposal(startPose,endPose)
        #p.check_for_obstacles(startPose,endPose)

        #Sense
        #sensorReading = float(input("\nInsert sensor reading: "))
        d = p.get_distance_to_all_obstacles(endPose)
        sensorReading = np.min(d)
        print(sensorReading)
        p.model_correction(sensorReading)

        #Resample
        p.model_resample()

        #refresh view
        p.plotParticles()
        


    print("OK")
    cv2.waitKey(0)


"""
References:
https://github.com/yz9/Monte-Carlo-Localization/blob/master/python/monte_carlo_localization_v2.py
https://github.com/nitinnat/Monte-Carlo-Localization/blob/master/Problem3_4.py
http://ais.informatik.uni-freiburg.de/teaching/ws12/practicalB/02-mcl.pdf
https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
http://ros-developer.com/2019/04/10/parcticle-filter-explained-with-python-code-from-scratch/
https://github.com/jamesjackson/ev3-localization/blob/da033fa3a3efd499581f85ab1e16b9aaf46ae93d/ev3_wall_trace_localize.py#L330

"""
