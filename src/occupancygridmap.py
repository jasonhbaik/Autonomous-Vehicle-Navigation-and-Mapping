#!/usr/bin/python

import numpy as np
import sys
import cv2
import time
import rospy
import tf2_ros
import math
import tf.transformations


from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, OccupancyGrid
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive


class OccupancyGridMap:
    def __init__(self):
        #Topics & Subs, Pubs
        # Read paramters form params.yaml
        lidarscan_topic =rospy.get_param('~scan_topic')
        map_topic =rospy.get_param('~map_topic')
        odom_topic="/odom"

        self.t_prev=rospy.get_time()
        self.max_lidar_range=rospy.get_param('~scan_range')
        self.scan_beams=rospy.get_param('~scan_beams')
        
        # Read the map parameters from *.yaml file
        # ....
        map_res = rospy.get_param('~map_res')
        map_width = rospy.get_param('~map_width')
        map_height = rospy.get_param('~map_height')
        self.p_occ = rospy.get_param('~p_occ') 
        self.p_free = rospy.get_param('~p_free')
        self.l_occ = self.prob_to_log(self.p_occ)
        self.l_free = self.prob_to_log(self.p_free)
	self.scan_to_base_link = rospy.get_param('~scan_distance_to_base_link')

        self.map_occ_grid_msg = OccupancyGrid()

        # Initialize the map meta info in the Occupancy Grid Message, e.g., frame_id, stamp, resolution, width, height, etc.
        # ...

        self.map_occ_grid_msg.header.frame_id  = rospy.get_param('~odom_frame')
        self.map_occ_grid_msg.header.stamp = rospy.Time.now()

        self.map_occ_grid_msg.info.resolution = rospy.get_param('~map_res')
        self.map_occ_grid_msg.info.width = map_width
        self.map_occ_grid_msg.info.height = map_height

        self.map_occ_grid_msg.info.origin.position.x =  -map_width*map_res/2
        self.map_occ_grid_msg.info.origin.position.y =  -map_height*map_res/2

        self.map_occ_grid_msg.info.origin.orientation.x = 0
        self.map_occ_grid_msg.info.origin.orientation.y = 0
        self.map_occ_grid_msg.info.origin.orientation.z = 0
        self.map_occ_grid_msg.info.origin.orientation.w = 1
          
        self.position_x = 0
        self.position_y = 0 
        self.position_z = 0


        self.origin_x = self.map_occ_grid_msg.info.origin.position.x
        self.origin_y = self.map_occ_grid_msg.info.origin.position.y
        self.yaw = 0

	self.larray = np.zeros((map_height, map_width), dtype = np.float32)



        # Initialize the cell occuopancy probabilites to 0.5 (unknown) with all cell data in Occupancy Grid Message set to unknown 
 	#STORING AS LOG ODDS
        self.data = [0.0]*(map_width*map_height)

        self.map_occ_grid_msg.data = [-1] * (map_width*map_height)

        # Subscribe to Lidar scan and odomery topics with corresponding lidar_callback() and odometry_callback() functions 
        # ...
        rospy.Subscriber(lidarscan_topic, LaserScan, self.lidar_callback)
        rospy.Subscriber(odom_topic, Odometry, self.odom_callback)

        # Create a publisher for the Occupancy Grid Map

        self.map_pub = rospy.Publisher(map_topic, OccupancyGrid, queue_size = 10)
        # ...            

    def prob_to_log(self, prob): #converting probability to log odds
	if prob >= 1.0:
	    prob = 0.99999
        elif prob <= 0:
            prob = 0.00001 
        return (math.log(prob/(1-prob)))
    
    def log_to_prob(self, log): #converting log odds to probability 
	 prob = (1-(1/(1+math.exp(log))))
	 if prob >= 1:
		return 1
	 elif prob <= 0:
		return 0
         return

    # lidar_callback () uses the current LiDAR scan and Wheel Odometry data to uddate and publish the Grid Occupancy map 
    
    def lidar_callback(self, data):      

        #M X N columsn = A[i][j] = (ixN + j)

        #for each mij
        # 1 Find the closest LiDAR ray to the line connecting the center of the cell


        res = self.map_occ_grid_msg.info.resolution
        baselinkX = self.position_x
        baselinkY = self.position_y
        lidarX = baselinkX + self.scan_to_base_link*math.cos(self.yaw)
        lidarY = baselinkY + self.scan_to_base_link*math.sin(self.yaw)
        lidarYaw = self.yaw + math.pi

        
        for row in range(self.map_occ_grid_msg.info.height):
            for col in range(self.map_occ_grid_msg.info.width):
                 
                 cellX = col*res + res/2 + self.origin_x
                 cellY = row*res + res/2 + self.origin_y

                 vectorX = cellX - lidarX
                 vectorY = cellY - lidarY

                 relative_lidar_to_cell_yaw = math.atan2(vectorY,vectorX) - lidarYaw

	         angle = math.atan2(math.sin(relative_lidar_to_cell_yaw), math.cos(relative_lidar_to_cell_yaw))
                 

                 between_distance = math.sqrt(vectorX**2 + vectorY**2)


                 closest_ray = int((angle-data.angle_min)/data.angle_increment)

                 print("Angle min: ", data.angle_min , "Angle Increment: ", data.angle_increment)

                 ray_length = data.ranges[closest_ray]

                 if ray_length == float("inf") or ray_length < data.range_min or ray_length > data.range_max:
	            continue
                 
                 difference = abs(between_distance-ray_length)
		 #print("Ray Length: ", ray_length, " Difference: ", difference)

		#RAY LENGTH, RANGE LENGTH, DIFFERENCE OF THE TWO

                 if difference <res*math.sqrt(2)/2: #occupied
                    inverse_sensor = self.l_occ
                 elif between_distance < ray_length:#free
                    inverse_sensor = self.l_free
                 else:
                     continue



		 self.larray[row,col] += inverse_sensor


		 lprev = self.larray[row,col]

                 map_pub = -1 		
 
                 if lprev > self.l_occ: 
                     map_pub = 100
                 elif lprev < self.l_free:	
                      map_pub = 0
		 else:
	       	      continue
                
                 self.map_occ_grid_msg.data[row*self.map_occ_grid_msg.info.width + col] = map_pub
		#print(log_cell)

    # ...
       
        # Publish to map topic
        self.map_occ_grid_msg.header.stamp = rospy.Time.now()
        self.map_pub.publish(self.map_occ_grid_msg)

    # odom_callback() retrives the wheel odometry data from the publsihed odom_msg
     
    def odom_callback(self, odom_msg):
         self.position_x = odom_msg.pose.pose.position.x 
         self.position_y = odom_msg.pose.pose.position.y 
         self.position_z = odom_msg.pose.pose.position.z 

         siny = 2 * (odom_msg.pose.pose.orientation.w*odom_msg.pose.pose.orientation.z + odom_msg.pose.pose.orientation.x*odom_msg.pose.pose.orientation.y)
         cosy = 1 - 2 * (odom_msg.pose.pose.orientation.x**2 + odom_msg.pose.pose.orientation.z**2)
         self.yaw = math.atan2(siny, cosy)


def main(args):
    rospy.init_node("occupancygridmap", anonymous=True)
    OccupancyGridMap()
    rospy.sleep(0.1)
    rospy.spin()

if __name__=='__main__':
	main(sys.argv)
