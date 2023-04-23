from setuptools import setup

package_name = 'team23_final'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Tofunmi Sodimu',
    maintainer_email='tofsodimu@gmail.com',
    description='Lab 6 Final',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'train_classifier = team23_final.train_classifier:main',
            'classifier_node = team23_final.classifier_node:main',
            'image_node = team23_final.image_node:main',
            
        ],
    },
)
