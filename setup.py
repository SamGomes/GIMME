from setuptools import dist, setup, Extension
import glob

sfc_module = Extension(
  'GIMMESolver', 
  include_dirs = ['solverModules/GIMME_solver_modules'],
  sources = glob.glob('solverModules/GIMME_solver_modules/*.cpp'),
  language='c++',
  )

with open('README.md', 'r') as file:
    readmeFile = file.read()

setup(
    name="GIMMECore",
    version="1.5.0",
    license = "CC BY 4.0",
    author="Samuel Gomes",
    author_email = "samuel.gomes@tecnico.ulisboa.pt",
    description="GIMME (Group Interactions Management for Multiplayer sErious games) is a research tool which focuses on the management of interactions in groups so that the collective ability improves.",
    long_description=readmeFile,
    long_description_content_type="text/markdown",
    url='https://github.com/SamGomes/GIMME',
    packages=['GIMMECore', 'GIMMECore.ModelBridge', 'GIMMECore.AlgDefStructs'],
    classifiers = [
    'Development Status :: 4 - Beta',
    'Environment :: Console',
    'Intended Audience :: Science/Research',
    'Programming Language :: Python :: 3',
    'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
    install_requires=[
      'python-decouple',
      'deap',
      'pandas',
      'scikit-learn',
      'matplotlib',
      'pymongo'
    ],
    ext_modules=[sfc_module]
)

