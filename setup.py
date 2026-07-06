from setuptools import find_packages, setup


setup(
    name="waterint",
    version="0.1.0",
    description="Config-driven analysis tools for water-containing molecular simulations.",
    packages=find_packages(include=["waterint", "waterint.*"]),
    include_package_data=True,
    package_data={"waterint.ui": ["static/*"]},
    install_requires=["numpy", "scipy", "matplotlib", "PyYAML"],
    entry_points={"console_scripts": ["waterint=waterint.cli:main"]},
    python_requires=">=3.9",
)
