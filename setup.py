
from setuptools import find_packages, setup
import os

package_name = 'pose_state_publisher'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # msg 파일도 설치되도록 추가
        (os.path.join('share', package_name, 'msg'), ['msg/PoseInfo.msg']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='seorin',
    maintainer_email='isssweetner@gmail.com',
    description='Pose state publisher with MediaPipe (multi-person)',
    license='Apache-2.0',
    # tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        	'pose_state_publish = pose_state_publisher.pose_state_publish:main',
        ],
    },
)
