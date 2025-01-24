from setuptools import setup
import os

# 读取 README.md 作为长描述
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='EasYoloD',
    version='0.1',
    description='easyolo python library',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='zaixia108',
    author_email='xvbowen2012@gmail.com',  # 添加您的邮箱
    license='MIT',
    packages=['EasYoloD'],
    install_requires=[
        'opencv-python<=4.8.0.74',
        'numpy<=1.26'
    ],
    python_requires='>=3.8, <3.14',
    url='https://github.com/zaixia108/easyolo',
    project_urls={
        "Bug Tracker": "https://github.com/zaixia108/easyolo/issues",
        "Documentation": "https://github.com/zaixia108/easyolo/wiki",
        "Source Code": "https://github.com/zaixia108/easyolo",
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Operating System :: OS Independent',
    ],
    keywords='yolo, object detection, computer vision',
)