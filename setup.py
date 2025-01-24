from setuptools import setup

setup(
    name='easyolo',
    version='0.1',
    description='easyolo python library',
    author='zaixia108',
    license='MIT',
    packages=['easyolo'],  # 直接指定包名
    install_requires=[
        'opencv-python<=4.8.0.74',
        'numpy<=1.26',
    ],
    python_requires='>=3.8, <3.14',  # 指定 Python 版本范围 3.8-3.13
    author_email='',
    url='',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Development Status :: 3 - Alpha',  # 添加开发状态
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    # 添加项目创建时间和作者信息
    project_urls={
        'Source': '',  # 可以添加您的代码仓库链接
    },
    keywords=['easyolo', 'computer vision'],  # 添加关键词便于搜索
)