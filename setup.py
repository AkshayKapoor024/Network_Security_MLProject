from setuptools import find_packages,setup
from typing import List
# Responsible for creating whole python project as a package with all dependencies and code written as a whole

# We need to remove -e . which was added in requirements.txt to trigger execution for setup.py here while reading requirements.txt it should be removed 
HYPHEN_E_DOT = '-e .'

# Helper function to get all libraries that needs to be required and installed while building the package using setup
def get_requirements(file_path:str)->List[str]:
    try:
        '''
        This function will return the list of requirements
        '''
        requirements = []
        with open(file_path) as file_obj:
            requirements = file_obj.readlines()
            # Removing \n new line characters from list found by reading requirements.txt 
            requirements=[req.replace('\n','') for req in requirements]
        
            if HYPHEN_E_DOT in requirements:
                requirements.remove(HYPHEN_E_DOT)
    except FileNotFoundError:
        print('Requirements.txt file not found')

setup(
    name='Network_Security_Project',
    version='0.0.1',
    author='Akshay kapoor',
    author_email='work.akshaykapoor24@gmail.com',
    # finds folders containing __init__.py file and consider that folder as a package and will try to build this package 
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)