
## Run ros
in a new terminal type: `roscore`

## Set up the world 

Open the "world" folder

Copy and paste the `maze.world` file into  ->  `/catkin_ws/src/thymio_course_skeleton/launch/worlds`

Copy and paste the `model.config` and `model.sdf` files into -> `/home/usi/catkin_ws/src/thymio_course_skeleton/launch/models`

Open the world with typing on a new terminal: 

```roslaunch thymio_course_skeleton thymio_gazebo_bringup.launch name:=thymio10 world:=maze```


## Run the code
in a new terminal access `/catkin_ws/src/FinalProj/src`

build the package with:
```catkin build```

### For model 1 
execute the python file with:
```python model1.py```

### For model 2 (using gazebo)
execute the python file with:
```python final.py```

*NOTE*: Both models wait for a key when initially run. To start running it click the image and hit any key. \
:bulb: Both models require the image `maze.png` to be in the same directory. 


