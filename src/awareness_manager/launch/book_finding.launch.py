"""
Find My Book — full ROS2 + Gazebo launch.

Starts:
  1. Gazebo gzserver with the KRR house world + book props
  2. gzclient launched directly WITHOUT libgazebo_ros_eol_gui.so (that plugin
     accesses the UserCamera before it is initialised, causing the
     boost::shared_ptr<Camera> px != 0 assertion crash on world load)
  3. MIRTE Master robot (spawn + controllers) — delayed 5 s
  4. Nav2 (AMCL + navigation stack)
  5. BookFindingNode (awareness manager)

Usage:
    ros2 launch awareness_manager book_finding.launch.py
    ros2 launch awareness_manager book_finding.launch.py budget:=4 f6:=false
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
        DeclareLaunchArgument('budget',               default_value='3'),
        DeclareLaunchArgument('tick_rate',            default_value='10.0'),
        DeclareLaunchArgument('observation_interval', default_value='5.0'),
        DeclareLaunchArgument('alpha',                default_value='0.5'),
        DeclareLaunchArgument('nav_eta',              default_value='15.0'),
        DeclareLaunchArgument('f6',                   default_value='true'),
        DeclareLaunchArgument('dwell_time',           default_value='8.0'),
        DeclareLaunchArgument('loop',                 default_value='true'),
        DeclareLaunchArgument('start_delay',          default_value='38.0'),
        DeclareLaunchArgument('use_rviz',             default_value='true'),
    ]

    pkg_robocup   = get_package_share_directory('robocup_home_simulation')
    pkg_mirte     = get_package_share_directory('mirte_gazebo')
    pkg_nav       = get_package_share_directory('mirte_navigation')
    pkg_plasys    = get_package_share_directory('plasys_house_world')
    pkg_aws       = get_package_share_directory('aws_robomaker_small_house_world')
    pkg_am        = get_package_share_directory('awareness_manager')
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')

    book_models_path = os.path.join(pkg_am, 'models')
    world_file = os.path.join(pkg_robocup, 'worlds', 'book_finding.world')

    # ── Build Gazebo path env from package.xml exports ────────────────────────
    # GazeboRosPaths.get_paths() scans all installed packages for
    # <gazebo_ros gazebo_model_path/gazebo_plugin_path> exports.
    sys.path.insert(0, os.path.join(pkg_gazebo_ros, 'launch'))
    from scripts import GazeboRosPaths  # noqa: E402
    gz_model, gz_plugin, gz_media = GazeboRosPaths.get_paths()

    if 'GAZEBO_MODEL_PATH' in environ:
        gz_model += pathsep + environ['GAZEBO_MODEL_PATH']
    if 'GAZEBO_PLUGIN_PATH' in environ:
        gz_plugin += pathsep + environ['GAZEBO_PLUGIN_PATH']
    if 'GAZEBO_RESOURCE_PATH' in environ:
        gz_media += pathsep + environ['GAZEBO_RESOURCE_PATH']

    # Add mirte media path (normally set by gazebo_mirte_world_generated.launch.xml)
    gz_media += pathsep + os.path.join(pkg_mirte, 'media')
    # Explicitly add book models so gzserver can resolve model://book
    gz_model += pathsep + book_models_path

    # ── Add system Gazebo 11 base paths ──────────────────────────────────────
    # GazeboRosPaths.get_paths() only scans ROS packages, not the system Gazebo
    # installation. Without /usr/share/gazebo-11 in GAZEBO_RESOURCE_PATH,
    # gzclient's RTShaderSystem can't find its shader libraries and the OGRE
    # scene fails to initialize, leaving the UserCamera null → px != 0 crash.
    gz_media   += pathsep + '/usr/share/gazebo-11'
    gz_plugin  += pathsep + '/usr/lib/x86_64-linux-gnu/gazebo-11/plugins'
    gz_model   += pathsep + '/usr/share/gazebo-11/models'

    gazebo_env = {
        'GAZEBO_MODEL_PATH':    gz_model,
        'GAZEBO_PLUGIN_PATH':   gz_plugin,
        'GAZEBO_RESOURCE_PATH': gz_media,
        'OGRE_RESOURCE_PATH':   '/usr/lib/x86_64-linux-gnu/OGRE-1.9.0',
    }

    # SetEnvironmentVariable propagates to ALL processes launched after it,
    # including gzserver (which computes its own paths from GazeboRosPaths but
    # also appends GAZEBO_MODEL_PATH from the environment).
    set_env_actions = [
        SetEnvironmentVariable('GAZEBO_MODEL_PATH',    gz_model),
        SetEnvironmentVariable('GAZEBO_PLUGIN_PATH',   gz_plugin),
        SetEnvironmentVariable('GAZEBO_RESOURCE_PATH', gz_media),
        SetEnvironmentVariable('OGRE_RESOURCE_PATH',   '/usr/lib/x86_64-linux-gnu/OGRE-1.9.0'),
    ]

    # ── gzserver ──────────────────────────────────────────────────────────────
    # params_file='' prevents nav_launch's params_file declaration from
    # propagating into gzserver.launch.py via the shared launch context.
    # verbose=true shows world-loading progress on the terminal so you can
    # see when gzserver is ready before spawn_entity.py times out.
    gzserver_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gzserver.launch.py')
        ),
        launch_arguments={'world': world_file, 'params_file': '', 'verbose': 'true'}.items(),
    )

    # ── gzclient — launched directly so we can omit libgazebo_ros_eol_gui.so ─
    # The EOL plugin (--gui-client-plugin=libgazebo_ros_eol_gui.so) accesses
    # get_active_camera() during its Load() callback before the UserCamera is
    # initialised by gzclient's scene setup, producing the assertion crash.
    # OGRE_RTT_MODE=Copy avoids any secondary RTT-related init failures.
    gzclient_process = ExecuteProcess(
        cmd=['gzclient', '--verbose'],
        output='screen',
        additional_env={**gazebo_env, 'OGRE_RTT_MODE': 'Copy'},
    )

    # ── Robot + controllers (delayed 20 s) ───────────────────────────────────
    # gzserver typically loads the world within 5 s on a warm model cache
    # and within ~22 s on a cold cache.  spawn_entity.py has a 30 s service
    # timeout so it will wait for /spawn_entity even if we start early.
    spawn_launch = IncludeLaunchDescription(
        XMLLaunchDescriptionSource(
            os.path.join(pkg_mirte, 'launch', 'spawn_mirte_master.launch.xml')
        ),
        launch_arguments={
            'x':                   '1.0527',
            'y':                   '0.509611',
            'z':                   '0.021444',
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

    zero_cmd_vel = ExecuteProcess(
        cmd=['ros2', 'topic', 'pub', '/zero_cmd_vel',
             'geometry_msgs/msg/Twist', '{}', '-r', '100', '--print=4294967295'],
    )

    twist_mux = Node(
        package='twist_mux',
        executable='twist_mux',
        remappings=[('/cmd_vel_out', '/cmd_vel')],
        parameters=[
            os.path.join(pkg_mirte, 'config', 'twist_mux.yaml'),
            {'use_sim_time': True},
        ],
    )

    delayed_robot = TimerAction(
        period=20.0,
        actions=[spawn_launch, controllers_1, controllers_2, zero_cmd_vel, twist_mux],
    )

    # ── Nav2 ──────────────────────────────────────────────────────────────────
    nav_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_nav, 'launch', 'robot_navigation.launch.py')
        ),
        launch_arguments={'use_rviz': LaunchConfiguration('use_rviz')}.items(),
    )

    # ── BookFinding AM node ───────────────────────────────────────────────────
    am_node = Node(
        package='awareness_manager',
        executable='book_finding_node',
        name='book_finding_awareness',
        output='screen',
        parameters=[{
            'budget':               LaunchConfiguration('budget'),
            'tick_rate':            LaunchConfiguration('tick_rate'),
            'observation_interval': LaunchConfiguration('observation_interval'),
            'alpha':                LaunchConfiguration('alpha'),
            'nav_eta':              LaunchConfiguration('nav_eta'),
            'f6':                   LaunchConfiguration('f6'),
        }],
    )

    # ── Scripted room-tour navigator ──────────────────────────────────────────
    # Drives the robot through all rooms in sequence so the AM can demonstrate
    # F2 anticipatory pre-tuning without a manual decision-maker.
    navigator_node = Node(
        package='awareness_manager',
        executable='scripted_navigator_node',
        name='scripted_navigator',
        output='screen',
        parameters=[{
            'dwell_time':  LaunchConfiguration('dwell_time'),
            'loop':        LaunchConfiguration('loop'),
            'start_delay': LaunchConfiguration('start_delay'),
        }],
    )

    return LaunchDescription(
        args + set_env_actions + [
            nav_launch,
            gzserver_launch,
            gzclient_process,
            delayed_robot,
            am_node,
            navigator_node,
        ]
    )
