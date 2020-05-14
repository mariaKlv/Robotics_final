#!/usr/bin/env python

import rospy
from std_msgs.msg import String


class HelloworldSub:

    def __init__(self):
        """Initialization."""

        # initialize the node
        rospy.init_node(
            'helloworld_sub'  # name of the node
        )

        # create string subscriber
        self.subscriber = rospy.Subscriber(
            '/helloworld',  # name of the topic
            String,  # message type
            self.greet  # function that handles incoming messages
        )

        # set node update frequency in Hz
        self.rate = rospy.Rate(10)

    def greet(self, data):
        """Greets the person."""
        rospy.loginfo_once('Hello %s!' % data.data)

    def wait_for_termination(self):
        """Waits until the node is terminated."""

        while not rospy.is_shutdown():

            # sleep until next step
            self.rate.sleep()


if __name__ == '__main__':
    subscriber = HelloworldSub()

    try:
        subscriber.wait_for_termination()
    except rospy.ROSInterruptException as e:
        pass
