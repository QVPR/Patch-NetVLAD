import os, sys
from setuptools import setup, find_packages


here = os.path.abspath(os.path.dirname(__file__))

# Get the long description from the README file
with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


install_require_list = [
    'numpy', 'torch', 'torchvision',
    'tqdm', 'scipy', 'Pillow', 'scikit-learn',
    'faiss', 'natsort', 'matplotlib']

# workaround as opencv-python does not show up in "pip list" within a conda environment
# we do not care as conda recipe has py-opencv requirement anyhow
is_conda = os.path.exists(os.path.join(sys.prefix, 'conda-meta'))
if not is_conda:
    install_require_list.append('opencv-python')

setup(name='patchnetvlad',
      version='0.1.4',
      description='Patch-NetVLAD: An open-source Python implementation of the CVPR2021 paper',
      long_description = long_description,
      long_description_content_type='text/markdown',
      author='Stephen Hausler, Sourav Garg, Ming Xu, Michael Milford and Tobias Fischer',
      author_email='stephen.hausler@hdr.qut.edu.au',
      url='https://github.com/QVPR/Patch-NetVLAD',
      license='MIT',
      classifiers=[
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
      ],
      python_requires='>=3.6',
      install_requires=install_require_list,
      packages=find_packages(),
      keywords=[
          'python', 'place recognition', 'image retrieval', 'computer vision', 'robotics'
      ],
      scripts=['feature_extract.py', 'feature_match.py', 'match_two.py', 'download_models.py'],
      entry_points={
        'console_scripts': ['patchnetvlad-match-two=match_two:main',
                            'patchnetvlad-feature-match=feature_match:main',
                            'patchnetvlad-feature-extract=feature_extract:main',
                            'patchnetvlad-download-models=download_models:download_all_models'],
      },
      package_data={'': ['configs/*.ini', 'dataset_gt_files/*.npz', 'example_images/*',
                         'output_features/.hidden', 'pretrained_models/.hidden', 'results/.hidden',
                         'dataset_imagenames/*.txt']}
)
