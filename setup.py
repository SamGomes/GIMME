from setuptools import dist, setup, Extension
import glob

sfc_module = Extension(
  'GIMMESolver', 
  sources = glob.glob('solverModules/GIMME_solver_modules/*.cpp'),
  #include_dirs=[pybind11.get_include()],
  language='c++',
  )

setup(
    name="GIMMECore",
    version="1.4.0",
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
      'python-decouple',
      'deap',
      'pandas',
      'scikit-learn',
      'matplotlib',
      'pymongo'
    ],
    ext_modules=[sfc_module]
)

