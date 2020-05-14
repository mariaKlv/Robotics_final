#!/usr/bin/env python

import rospy
from std_msgs.msg import String


class HelloworldPub:

    def __init__(self):
        """Initialization."""

        
        
        # initialize the node
        rospy.init_node(
            'helloworld_pub'
            )
        
        self.name = 'maria'

        # create string publisher
        self.publisher = rospy.Publisher('/helloworld',String,
            queue_size=10)
        # set node update frequency in Hz
        self.rate = rospy.Rate(10)

    def publish(self):
        """Publishes a message."""
        while not rospy.is_shutdown():
            self.publisher.publish(self.name)
            self.rate.sleep()

if __name__ == '__main__':
    publisher = HelloworldPub()

    publisher.publish()

    try:
        publisher.publish()
    except rospy.ROSInterruptException as e:
        pass