from setuptools import setup, find_packages
from pathlib import Path

HERE = Path(__file__).parent

# 🧾 Manifiesto reproducible
README = (HERE / "README.md").read_text(encoding="utf-8") if (HERE / "README.md").exists() else ""

# 🔍 Lectura modular de dependencias
def _read_requirements():
    for p in ("configs/requirements.txt", "requirements.txt", "requirements.lock"):
        f = HERE / p
        if f.exists():
            return [
                line.strip()
                for line in f.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.strip().startswith("#")
            ]
    return []

install_requires = _read_requirements()

# 🛠️ Configuración del paquete
setup(
    name="ia-core-models",
    version="1.0.0",
    description="Modelos éticos y reproducibles para auditoría técnica",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Develop Aguascalientes",
    author_email="compucell.ags@gmail.com",
    license="Apache-2.0",
    packages=find_packages(exclude=("tests", "docs")),
    include_package_data=True,
    install_requires=install_requires,
    python_requires=">=3.10",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords=[
        "auditoría",
        "ética",
        "reproducibilidad",
        "modelos técnicos",
        "infraestructura modular",
    ],
    project_urls={
        "Documentation": "https://github.com/DevelopAguascalientes/ia-core-models",
        "Source": "https://github.com/DevelopAguascalientes/ia-core-models",
        "Support": "https://buymeacoffee.com/ia-core-models",
    },
)
