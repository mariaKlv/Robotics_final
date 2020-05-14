This file contains instructions on how to run the file usi_angry_turtle.py.
In order to teleoperate through keyboard the turtle2 which is the spawned turtle that has been created as a Subscriber to the topic /turlte1/cmd_vel. The desired output topic is 
/turtle2/cmd_vel, so for that reason we have to remap the argument giving the control to the second turtle by running:
	rosrun turtlesim turtle_teleop_key /turtle1/cmd_vel:=/turtle2/cmd_vel
