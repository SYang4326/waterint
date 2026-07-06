from setuptools import find_packages, setup


setup(
    name="waterint",
    version="0.1.0",
    description="Config-driven analysis tools for water-containing molecular simulations.",
    package_dir={"": "v0"},
    packages=find_packages(where="v0", include=["waterint", "waterint.*"]),
    include_package_data=True,
    install_requires=["numpy", "scipy", "matplotlib", "PyYAML"],
    entry_points={"console_scripts": ["waterint=waterint.cli:main"]},
    python_requires=">=3.9",
)
