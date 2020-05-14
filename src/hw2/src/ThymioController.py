#!/usr/bin/env python
import rospy
import roslib
import sys
import std_srvs.srv
import time
import numpy as np
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose, Twist, Vector3
from std_srvs.srv import Empty as EmptyServiceCall

class MyTController:
	def __init__(self):
		# Initialize the node
		rospy.init_node('Mighty_Thymio_Controller') # name of the node

		# Publish to the topic '/thymio10/cmd_vel'
		self.cmd_vel_topic='/thymio10/cmd_vel'
		self.velocity_publisher = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=10)

		# initialize linear and angular velocities to 0
		self.velocity = Twist()
		
		# Odometry topic
		self.odometry_topic = '/thymio10/odom'
		# create pose subscriber
		self.pose_subscriber = rospy.Subscriber(
			self.odometry_topic,  # name of the topic
			Odometry,  # message type
			self.log_odometry  # function that hanldes incoming messages
		)
		
		# tell ros to call stop when the program is terminated
		rospy.on_shutdown(self.stop) 

		odom = Odometry()

		self.odom_position = odom.pose.pose.position
		# the rotation, quaternion created from yaw
		self.odom_quat = odom.pose.pose.orientation

		self.time = rospy.Duration(1)			

		# Publish at this rate
		self.rate = rospy.Rate(10)


	def log_odometry(self, data):
		"""Updates robot pose and velocities, and logs pose to console."""

		# Odometry knows how far the robot has gone and how much the angle of rotation is 
		# calculates the position and orientation for us
		self.odom_position = data.pose.pose.position

		# odom_trans.transform.rotation = odom_quat
		# which is the rotation, quaternion created from yaw
		self.odom_quat = data.pose.pose.orientation

	def get_control(self):
		self.velocity.linear.x = 1
		self.velocity.linear.y = 0
		self.velocity.linear.z = 0
		self.velocity.angular.x = 0
		self.velocity.angular.y = 0

		self.velocity.angular.z = -1
		time.sleep(4)
		self.velocity.angular.z = 1

		self.velocity_publisher.publish(self.velocity)
		rospy.sleep(self.time)

	def run(self):
		"""Controls the Thymio."""
		while not rospy.is_shutdown():

			# decide control action
			velocity = self.get_control()
			
			# sleep until next step
			self.rate.sleep()

	def stop(self):
		"""Stops the robot."""

		self.velocity_publisher.publish(
			Twist()  # set velocities to 0
		)

		self.rate.sleep()


if __name__ == '__main__':
	controller = MyTController()

	try:	
		controller.run()	
	except rospy.ROSInterruptException as e:
		pass