"""
Launch the AwarenessManager node and the PV inspection visualizer.

Usage (defaults):
    ros2 launch awareness_manager pv_inspection_demo.launch.py

Override any parameter:
    ros2 launch awareness_manager pv_inspection_demo.launch.py goal_id:=emergency_landing
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:

    args = [
        DeclareLaunchArgument('goal_id',              default_value='inspect_pv_field', description='Initial mission goal'),
        DeclareLaunchArgument('alpha',                default_value='0.5',              description='Spreading activation decay factor'),
        DeclareLaunchArgument('max_distance',         default_value='4.0',              description='Max weighted graph distance'),
        DeclareLaunchArgument('budget',               default_value='2',                description='Max concepts scheduled per tick'),
        DeclareLaunchArgument('tick_rate',            default_value='10.0',             description='Simulation tick rate (Hz)'),
        DeclareLaunchArgument('observation_interval', default_value='2.0',              description='Seconds between auto-observations'),
        DeclareLaunchArgument('memory_budget',        default_value='-1',               description='F4 budget B; -1 = disabled'),
        DeclareLaunchArgument('f1',                   default_value='true',             description='Enable F1 spreading activation'),
        DeclareLaunchArgument('f2',                   default_value='true',             description='Enable F2 anticipatory horizon'),
        DeclareLaunchArgument('f3',                   default_value='true',             description='Enable F3 utility saturation'),
        DeclareLaunchArgument('f4',                   default_value='true',             description='Enable F4 quadratic cost constraint'),
        DeclareLaunchArgument('f5',                   default_value='true',             description='Enable F5 epistemic drift'),
        DeclareLaunchArgument('f6',                   default_value='false',            description='Enable F6 spatial opportunity cost'),
    ]

    awareness_node = Node(
        package='awareness_manager',
        executable='awareness_node',
        name='awareness_manager',
        output='screen',
        parameters=[{
            'scenario':             'pv_inspection',
            'goal_id':              LaunchConfiguration('goal_id'),
            'alpha':                LaunchConfiguration('alpha'),
            'max_distance':         LaunchConfiguration('max_distance'),
            'budget':               LaunchConfiguration('budget'),
            'tick_rate':            LaunchConfiguration('tick_rate'),
            'observation_interval': LaunchConfiguration('observation_interval'),
            'memory_budget':        LaunchConfiguration('memory_budget'),
            'f1':                   LaunchConfiguration('f1'),
            'f2':                   LaunchConfiguration('f2'),
            'f3':                   LaunchConfiguration('f3'),
            'f4':                   LaunchConfiguration('f4'),
            'f5':                   LaunchConfiguration('f5'),
            'f6':                   LaunchConfiguration('f6'),
        }],
    )

    return LaunchDescription(args + [awareness_node])
