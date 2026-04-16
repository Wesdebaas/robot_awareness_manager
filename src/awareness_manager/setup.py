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
        ]),
    ],
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
            'run_birdhouse_viz       = demos.run_birdhouse_viz:main',
            'run_pv_inspection_viz   = demos.run_pv_inspection_viz:main',
        ],
    },
)
