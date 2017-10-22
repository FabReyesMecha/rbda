from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='sym_rbda',
      version='0.7.0',
      description='Numeric and symbolic functions for rigid body simulation and analysis',
      url='http://github.com/...',
      author='Fabian Reyes',
      author_email='burgundianvolker@gmail.com',
      license='MIT',
      packages=['sym_rbda'],
      install_requires=[
        'numpy',
        'math',
        'copy',
        'sympy',

      ],
      zip_safe=False)
