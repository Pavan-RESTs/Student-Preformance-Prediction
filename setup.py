from setuptools import find_packages,setup
from typing import List

HYPHEN_E_DOT = '-e .'

def get_requirements(file_path:str)->List[str]:
    ''''
    This function will return the list of requirements
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [libs.replace('\n','') for libs in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements
        

setup(
    name='Student-Performance-Prediction',
    version='0.0.1',
    author='Pavan',
    author_email='pavanbindhu54@gmail.com',
    packages=find_packages(),
    requires=get_requirements('requirements.txt')
)