from setuptools import find_packages, setup
import os
from glob import glob
package_name = 'checkers_game'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    package_data={package_name: ['assets/*']},
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*')))
    ],
    install_requires=['setuptools', 'mediapipe'],
    zip_safe=True,
    maintainer='jozef',
    maintainer_email='jkosecky54@gmail.com',
    description='TODO: Package description',
    license='Apache-2.00',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'checkers_node = checkers_game.checkers_node:main'
        ],
    },
)
