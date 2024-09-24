from setuptools import setup, find_packages
setup(
    name = 'w1ot',
    version = '0.0.1',
    description = 'Wasserstein-1 neural optimal transport',
    author = 'Yanshuo Chen',
    author_email = 'poseidonchan@icloud.com',
    url = 'https://github.com/poseidonchan/w1ot',
    license = 'GPL-3.0 License',
    packages = find_packages(),
    python_requires='>=3.10',
    platforms = 'any',
    install_requires = [
        'torch',
        'numpy',
        'matplotlib',
        'tqdm',
        'scikit-learn',
    ],
)