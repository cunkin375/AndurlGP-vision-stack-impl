from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'gate_vision'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'models'), glob('models/*')),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
    ],
    install_requires=['setuptools', 'torch', 'torchvision', 'ultralytics', 'opencv-python'],
    zip_safe=True,
    maintainer='garym',
    maintainer_email='garym@example.com',
    description='ROS2 Gate Localization Pipeline',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'gate_localization_node = gate_vision.gate_localization_node:main'
        ],
    },
)
