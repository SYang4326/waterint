from setuptools import find_packages, setup


setup(
    name="waterint",
    version="0.7.0.dev0",
    description="Config-driven analysis tools for water-containing molecular simulations.",
    package_dir={"": "src"},
    packages=find_packages(where="src", include=["waterint", "waterint.*"]),
    include_package_data=True,
    install_requires=["numpy", "scipy", "matplotlib", "PyYAML"],
    entry_points={"console_scripts": ["waterint=waterint.cli:main"]},
    python_requires=">=3.9",
)
