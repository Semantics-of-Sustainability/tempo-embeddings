## Badges

(Customize these badges with your own links, and check https://shields.io/ or https://badgen.net/ to see which other badges are available.)

| fair-software.eu recommendations | |
| :-- | :--  |
| (1/5) code repository              | [![github repo badge](https://img.shields.io/badge/github-repo-000.svg?logo=github&labelColor=gray&color=blue)](https://github.com/Semantics-of-Sustainability/tempo-embeddings) |
| (2/5) license                      | [![github license badge](https://img.shields.io/github/license/Semantics-of-Sustainability/tempo-embeddings)](https://github.com/Semantics-of-Sustainability/tempo-embeddings) |
 <!-- | (3/5) community registry           | [![RSD](https://img.shields.io/badge/rsd-tempo_embeddings-00a3e3.svg)](https://www.research-software.nl/software/tempo_embeddings) [![workflow pypi badge](https://img.shields.io/pypi/v/tempo_embeddings.svg?colorB=blue)](https://pypi.python.org/project/tempo_embeddings/) | -->
| (4/5) citation                     | [![DOI](https://zenodo.org/badge/DOI/<replace-with-created-DOI>.svg)](https://doi.org/<replace-with-created-DOI>) |
| (5/5) checklist                    | [![workflow cii badge](https://bestpractices.coreinfrastructure.org/projects/<replace-with-created-project-identifier>/badge)](https://bestpractices.coreinfrastructure.org/projects/<replace-with-created-project-identifier>) |
| howfairis                          | [![fair-software badge](https://img.shields.io/badge/fair--software.eu-%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8B-yellow)](https://fair-software.eu) |
| **Other best practices**           | &nbsp; |
| Static analysis                    | [![workflow scq badge](https://sonarcloud.io/api/project_badges/measure?project=Semantics-of-Sustainability_tempo-embeddings&metric=alert_status)](https://sonarcloud.io/dashboard?id=Semantics-of-Sustainability_tempo-embeddings) |
| Coverage                           | [![workflow scc badge](https://sonarcloud.io/api/project_badges/measure?project=Semantics-of-Sustainability_tempo-embeddings&metric=coverage)](https://sonarcloud.io/dashboard?id=Semantics-of-Sustainability_tempo-embeddings) |
| Documentation                      | [![Documentation Status](https://readthedocs.org/projects/tempo-embeddings/badge/?version=latest)](https://tempo-embeddings.readthedocs.io/en/latest/?badge=latest) |
| **GitHub Actions**                 | &nbsp; |
| Build                              | [![build](https://github.com/Semantics-of-Sustainability/tempo-embeddings/actions/workflows/build.yml/badge.svg)](https://github.com/Semantics-of-Sustainability/tempo-embeddings/actions/workflows/build.yml) |
| Citation data consistency               | [![cffconvert](https://github.com/Semantics-of-Sustainability/tempo-embeddings/actions/workflows/cffconvert.yml/badge.svg)](https://github.com/Semantics-of-Sustainability/tempo-embeddings/actions/workflows/cffconvert.yml) |
| SonarCloud                         | [![sonarcloud](https://github.com/Semantics-of-Sustainability/tempo-embeddings/actions/workflows/sonarcloud.yml/badge.svg)](https://github.com/Semantics-of-Sustainability/tempo-embeddings/actions/workflows/sonarcloud.yml) |
| MarkDown link checker              | [![markdown-link-check](https://github.com/Semantics-of-Sustainability/tempo-embeddings/actions/workflows/markdown-link-check.yml/badge.svg)](https://github.com/Semantics-of-Sustainability/tempo-embeddings/actions/workflows/markdown-link-check.yml) |

## How to use tempo_embeddings

A library for analysing (temporal) word embeddings.

The project setup is documented in [project_setup.md](project_setup.md). Feel free to remove this document (and/or the link to this document) if you don't need it.

See the [design document](design.md) for more information on the design of this project.

## Installation

To install tempo_embeddings from GitHub repository, do:

```console
git clone git@github.com:Semantics-of-Sustainability/tempo-embeddings.git
cd tempo-embeddings
python3 -m pip install .
```
## How to use tempo_embeddings on SURF Research Cloud?
SURF Research Cloud offers a ready-to-use environment for running tools without the need to install Python and other required libraries.

[Here](https://www.surf.nl/en/services/surf-research-cloud) you can apply for access.

If your Research Cloud account is set up for the Semantics of Sustainability project, you can use the tempo-embeddings tool by following these steps:
- Log in to [Research cloud](https://portal.live.surfresearchcloud.nl)  environment.
- Create a new workspace using the "semantics-of-sustainability" catalog item.
- Log in to the workspace you created in the previous step. 
- Open a new terminal and run:
```console
/etc/miniconda/bin/conda init
```
- Close the terminal, open a new one, and run:
```console
cd /scratch/tempo-embeddings/
conda activate tempo-embeddings
```
- Start to play with the notebooks using:
```console
jupyter lab notebooks/[1_compute_embeddings_nl.ipynb]
```

## Documentation

Include a link to your project's full documentation here.

## Contributing

If you want to contribute to the development of tempo_embeddings,
have a look at the [contribution guidelines](CONTRIBUTING.md).

## Credits

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [NLeSC/python-template](https://github.com/NLeSC/python-template).
