cmake_minimum_required(VERSION 3.0.0)

cmake_policy(SET CMP0048 NEW)

project(posthoc_learn VERSION 0.0.1)

find_package(catkin REQUIRED COMPONENTS
  message_generation
  rospy
  std_msgs
)

####### Optional: Message Generation ##############

#add_service_files(
#  FILES
#)

#generate_messages(
#  DEPENDENCIES
#  std_msgs
#)

############# End Message Generation ##############

catkin_python_setup()

catkin_package()