git clone https://github.com/Mirko-Nava/thymio_course_skeleton.git
roslaunch thymio_course_skeleton thymio_gazebo_bringup.launch name:=thymio10 world:=empty
roslaunch thymio_course_skeleton controller.launch robot_name:=thymio10
catkin create pkg HW2 --catkin-deps roscpp rospy std_msgs geometry_msgs
​rostopic pub /thymio10/cmd_vel geometry_msgs/Twist "linear:
  x: 11.0
  y: 0.0
  z: 0.0
angular:
  x: 0.0
  y: 0.0
  z: 11.0" -r 1