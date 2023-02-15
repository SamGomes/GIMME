from setuptools import setup, Extension
import glob
#import pybind11

sfc_module = Extension(
  'GIMMESolver', 
  sources = glob.glob('solverModules/GIMME_solver_modules/*.cpp'),
  #include_dirs=[pybind11.get_include()],
  language='c++',
  )
print(sfc_module)

setup(
    name="GIMMECore",
    version="1.3.0",
    license = "CC BY 4.0",
    author="Samuel Gomes",
    author_email = "samuel.gomes@tecnico.ulisboa.pt",
    packages=['GIMMECore', 'GIMMECore.ModelBridge', 'GIMMECore.AlgDefStructs'],
    classifiers = [
    'Development Status :: Development',
    'Intended Audience :: Science/Research',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Topic :: Scientific/Engineering :: Adaptation'
    ],
    install_requires=[
      'numpy',
      'scikit-learn',
      'deap', 
      'pymongo'
    ]

)

setup(
    name='GIMMESolver',
    version='1.0',
    ext_modules=[sfc_module]
   )

