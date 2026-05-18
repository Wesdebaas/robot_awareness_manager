"""
Launch the AwarenessManager node for the social serving scenario.

Usage (defaults):
    ros2 launch awareness_manager awareness_demo.launch.py

Override any parameter:
    ros2 launch awareness_manager awareness_demo.launch.py budget:=4 f6:=false
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:

    args = [
        DeclareLaunchArgument('goal_id',              default_value='serve_people_drinks', description='Initial mission goal'),
        DeclareLaunchArgument('alpha',                default_value='0.5',                 description='Spreading activation decay factor'),
        DeclareLaunchArgument('max_distance',         default_value='4.0',                 description='Max weighted graph distance'),
        DeclareLaunchArgument('budget',               default_value='3',                   description='Max concepts scheduled per tick'),
        DeclareLaunchArgument('tick_rate',            default_value='10.0',                description='Simulation tick rate (Hz)'),
        DeclareLaunchArgument('observation_interval', default_value='2.0',                 description='Seconds between auto-observations'),
        DeclareLaunchArgument('f6',                   default_value='true',                description='Enable F6 spatial opportunity cost'),
    ]

    awareness_node = Node(
        package='awareness_manager',
        executable='awareness_node',
        name='awareness_manager',
        output='screen',
        parameters=[{
            'scenario':             'social_serving',
            'goal_id':              LaunchConfiguration('goal_id'),
            'alpha':                LaunchConfiguration('alpha'),
            'max_distance':         LaunchConfiguration('max_distance'),
            'budget':               LaunchConfiguration('budget'),
            'tick_rate':            LaunchConfiguration('tick_rate'),
            'observation_interval': LaunchConfiguration('observation_interval'),
            'f6':                   LaunchConfiguration('f6'),
        }],
    )

    dashboard = ExecuteProcess(
        cmd=['ros2', 'run', 'awareness_manager', 'run_dashboard',
             '--scenario', 'social_serving', '--ros'],
        output='screen',
    )

    return LaunchDescription(args + [awareness_node, dashboard])
