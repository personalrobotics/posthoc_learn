cmake_minimum_required(VERSION 3.0.0)

cmake_policy(SET CMP0048 NEW)

project(posthoc_learn VERSION 0.0.1)

find_package(catkin REQUIRED COMPONENTS
  message_generation
  rospy
  std_msgs
  sensor_msgs
)

####### Message Generation ##############

add_service_files(
  FILES
  GetAction.srv
  PublishLoss.srv
)

catkin_python_setup()

generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
)

############# End Message Generation ##############

catkin_package()