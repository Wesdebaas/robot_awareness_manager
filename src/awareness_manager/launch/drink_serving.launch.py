"""
drink_serving.launch.py — Awareness-managed drink-serving Gazebo demo.

Starts:
  1. Gazebo (gzserver + gzclient) with the KRR house world
  2. MIRTE Master robot (spawn + controllers) — delayed 20 s
  3. Nav2 (AMCL + navigation stack) + RViz
  4. DrinkServingNode — AM-integrated kitchen ↔ living_room patrol

Usage:
    ros2 launch awareness_manager drink_serving.launch.py
    ros2 launch awareness_manager drink_serving.launch.py strategy:=reactive
    ros2 launch awareness_manager drink_serving.launch.py budget:=2 observation_interval:=15.0
"""

import os
import sys
from os import environ, pathsep

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_xml.launch_description_sources import XMLLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:

    args = [
        DeclareLaunchArgument('budget',               default_value='2'),
        DeclareLaunchArgument('observation_interval', default_value='15.0'),
        DeclareLaunchArgument('dwell_time',           default_value='30.0'),
        DeclareLaunchArgument('start_delay',          default_value='38.0'),
        DeclareLaunchArgument('alpha',                default_value='0.5'),
        DeclareLaunchArgument('strategy',             default_value='awareness_manager'),
        DeclareLaunchArgument('use_rviz',             default_value='true'),
    ]

    pkg_robocup    = get_package_share_directory('robocup_home_simulation')
    pkg_mirte      = get_package_share_directory('mirte_gazebo')
    pkg_nav        = get_package_share_directory('mirte_navigation')
    pkg_am         = get_package_share_directory('awareness_manager')
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')

    world_file = os.path.join(pkg_robocup, 'worlds', 'book_finding.world')

    # ── Gazebo path env ───────────────────────────────────────────────────────
    sys.path.insert(0, os.path.join(pkg_gazebo_ros, 'launch'))
    from scripts import GazeboRosPaths  # noqa: E402
    gz_model, gz_plugin, gz_media = GazeboRosPaths.get_paths()

    if 'GAZEBO_MODEL_PATH' in environ:
        gz_model += pathsep + environ['GAZEBO_MODEL_PATH']
    if 'GAZEBO_PLUGIN_PATH' in environ:
        gz_plugin += pathsep + environ['GAZEBO_PLUGIN_PATH']
    if 'GAZEBO_RESOURCE_PATH' in environ:
        gz_media += pathsep + environ['GAZEBO_RESOURCE_PATH']

    pkg_mirte_src  = get_package_share_directory('mirte_gazebo')
    gz_media += pathsep + os.path.join(pkg_mirte_src, 'media')

    gz_media  += pathsep + '/usr/share/gazebo-11'
    gz_plugin += pathsep + '/usr/lib/x86_64-linux-gnu/gazebo-11/plugins'
    gz_model  += pathsep + '/usr/share/gazebo-11/models'

    # Add book models so gzserver can resolve model://book (world file references them)
    book_models_path = os.path.join(pkg_am, 'models')
    gz_model += pathsep + book_models_path

    gazebo_env = {
        'GAZEBO_MODEL_PATH':    gz_model,
        'GAZEBO_PLUGIN_PATH':   gz_plugin,
        'GAZEBO_RESOURCE_PATH': gz_media,
        'OGRE_RESOURCE_PATH':   '/usr/lib/x86_64-linux-gnu/OGRE-1.9.0',
    }

    set_env_actions = [
        SetEnvironmentVariable('GAZEBO_MODEL_PATH',    gz_model),
        SetEnvironmentVariable('GAZEBO_PLUGIN_PATH',   gz_plugin),
        SetEnvironmentVariable('GAZEBO_RESOURCE_PATH', gz_media),
        SetEnvironmentVariable('OGRE_RESOURCE_PATH',   '/usr/lib/x86_64-linux-gnu/OGRE-1.9.0'),
    ]

    # ── gzserver ──────────────────────────────────────────────────────────────
    gzserver_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gzserver.launch.py')
        ),
        launch_arguments={'world': world_file, 'params_file': '', 'verbose': 'true'}.items(),
    )

    # ── gzclient ──────────────────────────────────────────────────────────────
    gzclient_process = ExecuteProcess(
        cmd=['gzclient', '--verbose'],
        output='screen',
        additional_env={**gazebo_env, 'OGRE_RTT_MODE': 'Copy'},
    )

    # ── Robot + controllers (delayed 20 s) ────────────────────────────────────
    spawn_launch = IncludeLaunchDescription(
        XMLLaunchDescriptionSource(
            os.path.join(pkg_mirte, 'launch', 'spawn_mirte_master.launch.xml')
        ),
        launch_arguments={
            'x':                   '1.0527',
            'y':                   '0.509611',
            'z':                   '0.021444',
            'yaw':                 '3.14159',
            'depth_camera_enable': 'False',
        }.items(),
    )

    controllers_1 = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster', 'mirte_master_arm_controller',
                   'mirte_master_gripper_controller'],
        parameters=[{'use_sim_time': True}],
    )

    controllers_2 = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['pid_wheels_controller', 'mirte_base_controller'],
        parameters=[{'use_sim_time': True}],
    )

    delayed_robot = TimerAction(
        period=20.0,
        actions=[spawn_launch, controllers_1, controllers_2],
    )

    # ── Nav2 ──────────────────────────────────────────────────────────────────
    nav_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_nav, 'launch', 'robot_navigation.launch.py')
        ),
        launch_arguments={'use_rviz': LaunchConfiguration('use_rviz')}.items(),
    )

    # ── Drink serving node (AM-integrated patrol) ────────────────────────────
    patrol_node = Node(
        package='awareness_manager',
        executable='drink_serving_node',
        name='drink_serving',
        output='screen',
        parameters=[{
            'use_sim_time':         True,
            'budget':               LaunchConfiguration('budget'),
            'observation_interval': LaunchConfiguration('observation_interval'),
            'dwell_time':           LaunchConfiguration('dwell_time'),
            'start_delay':          LaunchConfiguration('start_delay'),
            'alpha':                LaunchConfiguration('alpha'),
            'strategy':             LaunchConfiguration('strategy'),
        }],
    )

    return LaunchDescription(
        args + set_env_actions + [
            nav_launch,
            gzserver_launch,
            gzclient_process,
            delayed_robot,
            patrol_node,
        ]
    )
