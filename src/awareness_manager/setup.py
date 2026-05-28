from glob import glob
from setuptools import find_packages, setup

package_name = 'awareness_manager'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', [
            'launch/awareness_demo.launch.py',
            'launch/pv_inspection_demo.launch.py',
            'launch/book_finding.launch.py',
        ]),
        # Gazebo model for book_finding scenario
        ('share/' + package_name + '/models/book',
            glob('awareness_manager/scenarios/models/book/*')),
    ],
    package_data={
        'awareness_manager.scenarios': ['*.ttl'],
    },
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Wessel',
    maintainer_email='wesselremmelzwaan@gmail.com',
    description='Top-down robot awareness management for cognitive robotics.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'awareness_node          = awareness_manager.ros_node:main',
            'book_finding_node       = awareness_manager.book_finding_node:main',
            'scripted_navigator_node = awareness_manager.scripted_navigator_node:main',
            'run_dashboard           = demos.run_dashboard:main',
        ],
    },
)
