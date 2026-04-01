#!/usr/bin/env python
from __future__ import print_function
from lib2to3.pytree import Node
import sys
import math
from tokenize import Double
import numpy as np
import time

from  numpy import array, dot
from quadprog import solve_qp
#ROS Imports
import rospy
from sensor_msgs.msg import Image, LaserScan
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from nav_msgs.msg import Odometry

from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point


class GapBarrier:
    def __init__(self):
        #Topics & Subs, Pubs
        # Read the algorithm parameter paramters form params.yaml
        self.drive_topic = rospy.get_param("~drive_topic")
        self.scan_topic = rospy.get_param("~scan_topic")
        self.odom_topic = rospy.get_param("~odom_topic")

        self.safe_distance = rospy.get_param("~safe_distance")


        #From line following
        self.L = rospy.get_param("~wheelbase")
        self.k_p = rospy.get_param("~k_p")
        self.k_d = rospy.get_param("~k_d")
        self.v_d = rospy.get_param("~vehicle_velocity")
        self.d_stop = rospy.get_param("~stop_distance")
        self.d_tau = rospy.get_param("~stop_distance_decay")
        self.delta_max = rospy.get_param("~max_steering_angle")
        self.max_speed = rospy.get_param("~max_speed") 
        self.angle_bl = rospy.get_param("~angle_bl")
        self.angle_al = rospy.get_param("~angle_al")
        self.angle_br = rospy.get_param("~angle_br")
        self.angle_ar = rospy.get_param("~angle_ar")

        # Add your subscribers for LiDAR scan and Odomotery here
        # ...
        rospy.Subscriber(self.scan_topic, LaserScan, self.lidar_callback, queue_size=1)
        rospy.Subscriber(self.odom_topic, Odometry, self.odom_callback, queue_size=1)

        # Add your publisher for Drive topic here
        #...
        self.drive_pub = rospy.Publisher(self.drive_topic, AckermannDriveStamped, queue_size=1)

        # Initialize varables as needed 
        #...
        self.fov = math.radians(70)
        self.delta_theta = math.radians(11.25)

        self.angle_increment = 0
        self.angle_min = 0
        self.angle_max = 0 #

        self.range_min = 0
        self.range_max = 0 #

        self.yaw = 0

        self.position_x = 0 
        self.position_y = 0
        self.position_z = 0

        self.lower_index = 0 #

        self.vel = 0 
        self.left_obstacles = np.zeros((20,2), dtype = np.double) #creating an array for x and y 
        self.right_obstacles = np.zeros((20,2), dtype = np.double) #creating an array for x and y 

    def wrap_to_pi(self, angle):
        return (angle + math.pi) % (2 * math.pi) 

    # Optional function to pre-process LiDAR by considering only those LiDAR returns within a FOV in front of the vehicle;    

    def preprocess_lidar(self, ranges):
        #need to consider all the lidar beams that are + pi degrees turned
        # I want to take all the ranges between +- 70 deg + pi relative to the baselink (forward direction) compared to lidar (180 deg to baselink, yaw is in terms of baselink to odometry frame)
        lower_angle = (math.pi - self.fov)
        upper_angle = (math.pi + self.fov)

        lower_index = int(self.wrap_to_pi((lower_angle - self.angle_min))/ self.angle_increment)
        upper_index = int(self.wrap_to_pi((upper_angle - self.angle_min))/ self.angle_increment)

        self.lower_index = lower_index

        fov_ranges =  list(ranges[lower_index:upper_index]) #Might be a tuple

        for index, val in enumerate(fov_ranges):
             if math.isnan(val) or math.isinf(val) or val < self.safe_distance:
                  fov_ranges[index] = 0

        return fov_ranges #this should output the lidar beams from  140/angle_increment          
             
    # Optional function to find the the maximum gap in front the vehicle 
    def find_max_gap(self, proc_ranges):

         min_index = 0 
         max_index = 0
         max_gap = 0 

         # 0 0 0 0 1 1 1 0 0 
         # 1 1 1 1 1 0 1 1 1 1 0 0 0 

         curr_min_index = 0
         curr_length = 0 

         in_gap = False #to find a sequence of 1's
         
         for index, val in enumerate(proc_ranges):
             
             if val == 0:
                  in_gap = False
                  curr_length = 0
             else: #non-zero value
                  if in_gap == False: #sees the first non-zero value
                       curr_min_index = index
                       in_gap = True
                  curr_length += 1

                  if curr_length > max_gap:
                       min_index = curr_min_index
                       max_index = index
                       max_gap = curr_length
                       
         return min_index, max_index
        
    #Optional function to find the best direction of travel
    # start_i & end_i are the start and end indices of max-gap range, respectively
    # Returns index of best (furthest point) in ranges
    def find_best_point(self, start_i, end_i, proc_ranges):
        theta = 0
        num = 0 
        denom = 0 

        for index in range(start_i, end_i + 1):
             theta = (self.lower_index + index)*self.angle_increment
             num += proc_ranges[index]*theta
             denom += proc_ranges[index]

        if denom == 0:
             return self.lower_index + ((start_i + end_i)*self.angle_increment)/2 #goes stragiht if denom = 0 but should have no probelm there 

        theta_des = num/denom
        return theta_des
         
    # ...    

    # Optional function to set up and solve the optimization problem for parallel virtual barriers 
    def getWalls(self, left_obstacles, right_obstacles, wl0, wr0, alpha):

         right_obstacles = np.vstack([right_obstacles, np.ones((1,20))])   # (21,2)
         left_obstacles  = np.vstack([left_obstacles,  -1*np.ones((1,20))])  # (21,2)

         last_col = np.array([[0, 0],
                              [0, 0],
                              [1,-1]])  # (3,2)

         C = np.hstack([right_obstacles, left_obstacles, last_col]) #might need to make this a np.double type

         G = np.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 0.0001]], dtype = np.double)
         
         a = np.zeros((3), dtype = np.double)
         
         b = np.concatenate([np.ones(40), [-0.99, -0.99]])



         w = solve_qp(G,a,C,b,0)[0]
        
         return w #outputs 

     # ...
     

    # This function is called whenever a new set of LiDAR data is received; bulk of your controller implementation should go here 
    def lidar_callback(self, data):      
         self.angle_increment = data.angle_increment
         self.angle_min = data.angle_min
         self.angle_max = data.angle_max
         self.range_min = data.range_min
         self.range_max = data.range_max

    # Pre-process LiDAR data as necessary
    # ...

         preprocessed = self.preprocess_lidar(data.ranges)
    
    # Find the widest gape in front of vehicle
    # ...

         start, end = self.find_max_gap(preprocessed)
	 print("START: ", start, " END: " , end)
    
    # Find the Best Direction of Travel
    # ...

         best_dir = self.find_best_point(start, end, preprocessed) 
	 best_dir = int(best_dir)
	 print("Best DIR: ", math.degrees(best_dir))



    # Set up the QP for finding the two parallel barrier lines
    # ...

         #need to find the points and assign values for left array and right array

         angle_br = best_dir - math.radians(105)
         angle_al = best_dir + math.radians(95)
         al_start_index = int(self.wrap_to_pi(angle_al)/self.angle_increment)

         ### These values aren't being used v
         angle_ar = best_dir - math.radians(95) 
         angle_bl = best_dir + math.radians(105)

         ar_start_index = int(self.wrap_to_pi(angle_ar)/self.angle_increment)
         br_start_index = int(self.wrap_to_pi(angle_bl)/self.angle_increment)
         ### These values aren't being used ^

         for i in range(20): #righ side
              beam_length = data.ranges[i + br_start_index] 
              theta = angle_bl + self.angle_increment*i
              x = beam_length*math.cos(theta)
              y = beam_length*math.sin(theta)
              self.right_obstacles[i] = [x,y]


     #   for i in range(20): #right
              
          #    self.right_obstacles[i,0] = data.ranges[int(angle_br/(self.angle_increment)) + #i]*math.cos(angle_br + i*self.angle_increment) ### i = 0 [0,0] = > x and []
          #    self.right_obstacles[i,1] = data.ranges[int(angle_br/(self.angle_increment)) + #i]*math.sin(angle_br + i*self.angle_increment)

         for i in range(20): #left 
              beam_length = data.ranges[i + al_start_index] 
              theta = angle_al + self.angle_increment*i
              x = beam_length*math.cos(theta)
              y = beam_length*math.sin(theta)
              self.left_obstacles[i] = [x,y]
    

    # Solve the QP problem to find the barrier lines parameters w,b



         x = self.getWalls(self.left_obstacles.T, self.right_obstacles.T, 1, 1, 1)

         w = x[0:2]
         s = x[2]

    # Compute the values of the variables needed for the implementation of feedback linearizing+PD controller
    # ...
         v_s = self.vel 
         if self.vel == 0:
             self.vel = 1e-3

         w_r = w/(s-1)
         w_l = w/(s+1)
         
         d_l = 1/np.linalg.norm(w_l)
         d_r = 1/np.linalg.norm(w_r)

         w_hat_l = d_l*w_l
         w_hat_r = d_r*w_r

         d_dot_l = np.dot([v_s, 0], w_hat_l)
         d_dot_r = np.dot([v_s, 0], w_hat_r)

         d_dot_lr = d_dot_l - d_dot_r 
         d_lr = d_l - d_r    #check these conditions if not working

	 print("DDOTLR: ", d_dot_lr, "DLR: " , d_lr)

         cos_alpha_l = np.dot([0, -1], w_hat_l)
         cos_alpha_r = np.dot([0, 1], w_hat_r)

    # Compute the steering angle command
         term1 = -self.L/(v_s**2*(cos_alpha_l+cos_alpha_r))
         term2 = (-self.k_p*d_lr - self.k_d*d_dot_lr)

         delta = math.atan(term1 * term2)
         delta_c = max(-self.delta_max, min(self.delta_max, delta))

        # Centering error
        
    # Find the closest obstacle point in a narrow field of view in fronnt of the vehicle and compute the velocity command accordingly    
    # ...

         d_ob = float('inf')

         for index, val in enumerate(data.ranges):
            if val < data.range_min or math.isnan(val) or math.isinf(val):
                continue

            angle = data.angle_min + index * data.angle_increment

            if abs(angle - math.pi) <= self.delta_theta / 2.0:
                d_ob = min(d_ob, val)

         if math.isinf(d_ob):
            d_ob = data.range_max


         v_s_c = self.v_d * (1.0 - math.exp(-max(d_ob - self.d_stop, 0.0) / self.d_tau))


         v_s_c = min(v_s_c, self.max_speed)

        
    # Publish the steering and speed commands to the drive topic
    # ...

         drive_msg = AckermannDriveStamped()
         drive_msg.header.stamp = rospy.Time.now()
         drive_msg.header.frame_id = "base_link"

         drive_msg.drive.steering_angle = delta_c
         drive_msg.drive.speed = v_s_c

         self.drive_pub.publish(drive_msg)
    
    # Odometry callback 
    def odom_callback(self, odom_msg):
        # update current speed
         self.vel = odom_msg.twist.twist.linear.x
         
         self.position_x = odom_msg.pose.pose.position.x 
         self.position_y = odom_msg.pose.pose.position.y 
         self.position_z = odom_msg.pose.pose.position.z 

         siny = 2 * (odom_msg.pose.pose.orientation.w*odom_msg.pose.pose.orientation.z + odom_msg.pose.pose.orientation.x*odom_msg.pose.pose.orientation.y)
         cosy = 1 - 2 * (odom_msg.pose.pose.orientation.x**2 + odom_msg.pose.pose.orientation.z**2)
         self.yaw = math.atan2(siny, cosy)

def main(args):
    rospy.init_node("GapWallFollow_node", anonymous=True)
    wf = GapBarrier()
    rospy.sleep(0.1)
    rospy.spin()

if __name__=='__main__':
	main(sys.argv)
