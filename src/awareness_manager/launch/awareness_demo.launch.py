"""
Launch the AwarenessManager node and the birdhouse visualizer.

Usage (defaults):
    ros2 launch awareness_manager awareness_demo.launch.py

Override any parameter:
    ros2 launch awareness_manager awareness_demo.launch.py goal_id:=workbench budget:=2
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:

    args = [
        DeclareLaunchArgument('scenario',             default_value='birdhouse',       description='KB scenario to load'),
        DeclareLaunchArgument('goal_id',              default_value='build_birdhouse', description='Initial mission goal'),
        DeclareLaunchArgument('alpha',                default_value='0.5',             description='Spreading activation decay factor'),
        DeclareLaunchArgument('max_distance',         default_value='4.0',             description='Max weighted graph distance'),
        DeclareLaunchArgument('budget',               default_value='1',               description='Max concepts scheduled per tick'),
        DeclareLaunchArgument('tick_rate',            default_value='10.0',            description='Simulation tick rate (Hz)'),
        DeclareLaunchArgument('observation_interval', default_value='2.0',             description='Seconds between auto-observations'),
    ]

    awareness_node = Node(
        package='awareness_manager',
        executable='awareness_node',
        name='awareness_manager',
        output='screen',
        parameters=[{
            'scenario':             LaunchConfiguration('scenario'),
            'goal_id':              LaunchConfiguration('goal_id'),
            'alpha':                LaunchConfiguration('alpha'),
            'max_distance':         LaunchConfiguration('max_distance'),
            'budget':               LaunchConfiguration('budget'),
            'tick_rate':            LaunchConfiguration('tick_rate'),
            'observation_interval': LaunchConfiguration('observation_interval'),
        }],
    )

    visualizer = Node(
        package='awareness_manager',
        executable='run_birdhouse_viz',
        name='birdhouse_visualizer',
        output='screen',
    )

    return LaunchDescription(args + [awareness_node, visualizer])
