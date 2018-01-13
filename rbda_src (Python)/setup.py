from setuptools import setup

setup(name='rbda',
      version='0.7.0',
      description='Numeric and symbolic functions for rigid body simulation and analysis',
      url='http://github.com/...',
      author='Fabian Reyes',
      author_email='burgundianvolker@gmail.com',
      license='MIT',
      packages=['rbda'],
      install_requires=[
        'numpy',
        'sympy',
      ],
      zip_safe=False)
