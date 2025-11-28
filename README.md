<br/>
<div align="center">
<a href="h[ttps://github.com/ShaanCoding/makeread.me](https://github.com/Ray0716/teenhacks-2025)">
<img src="logo.gif" alt="Logo" width=600px height=auto > 
</a>
<h1 align="center">Kinematic Synthesis of Planar Walking Mechanisms through
Large-Scale Dataset Generation, Geometric Filtering, and
Optimization</h3>
<p align="center">
Github repository for research project conducted at Stony Brook University through the Simons Summer Program Fellowship. 
Submitted to Regeneron STS

<br/>
<br/>
<a href="https://google"><strong>Explore demo (coming soon) »</strong></a>
<br/>
<br/>

</p>
</div>



<div align="center">
  
![GitHub repo size](https://img.shields.io/github/repo-size/Ray0716/walking-mechanism-pipeline?style=for-the-badge&logo=files&logoColor=white)
![GitHub commit activity (branch)](https://img.shields.io/github/commit-activity/t/Ray0716/walking-mechanism-pipeline?style=for-the-badge&logo=git&logoColor=white&color=orange)
![GitHub last commit (branch)](https://img.shields.io/github/last-commit/Ray0716/walking-mechanism-pipeline?style=for-the-badge&logo=commit&logoColor=white)
![GitHub top language](https://img.shields.io/github/languages/top/Ray0716/walking-mechanism-pipeline?style=for-the-badge&logo=javascript&logoColor=white)
![GitHub Repo stars](https://img.shields.io/github/stars/Ray0716/walking-mechanism-pipeline?style=for-the-badge)
![Contributors](https://img.shields.io/github/contributors/Ray0716/walking-mechanism-pipeline?style=for-the-badge&color=dark-green) ![Issues](https://img.shields.io/github/issues/Ray0716/walking-mechanism-pipeline?style=for-the-badge) ![License](https://img.shields.io/github/license/Ray0716/walking-mechanism-pipeline?style=for-the-badge)

</div>

## Table of Contents

- [Table of Contents](#table-of-contents)
- [About The Project](#about-the-project)
  - [Built With](#built-with)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Features](#features)
  - [Notes](#notes)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)
- [Acknowledgments](#acknowledgments)

## The Project

<img src="demo.jpeg" alt="Logo" width=700px height=auto > 

This repository contains the dataset of 23K mechanisms, the code for generating the dataset, the optimization code, and the JS Kinematic Solver (from Deshpande et. al.)


## Getting Started

Below are instructions on how to run this project locally.


### Installation

Please follow the following steps for successful installation:

1. **Clone the Repository:** Get started by cloning the repository to your local machine.

   ```
   https://github.com/Ray0716/walking-mechanism-pipeline
   ```
2. Install dependencies

3. write this later

## Running the Project

Dataset generation/filtration is very comutationally expensive; this code was written and ran on the Seawulf GPU clusters in Stony Brook University. On computing clusters like Seawulf, expect filtration of 4 million mechanisms of one mechanism type to take around 7 hours. Running this on an M1 Macbook gave runtimes 4 times slower. 
We strongly reccomend running this project on a cloud computing service or computing cluster if you plan on generating a large-scale dataset. 

### Generating Random Initial Values

This is done with the ```Generate Random Init State.ipynb``` file. It generates a numpy matrix of random values used for generating random initial mechanisms for the dataset filtration. 

The included ```Randpos.npy``` numpy matrix contains 10 million groups of 11 (x, y) coordinates, suitable for generating 10 million mechanisms. However, if you wish to generate more mechanisms, you will need to run the ```Generate Random Init State.ipynb``` notebook to generate a larger initialization matrix. The size of your matrix can be determined by multiplying (number of mechanism types * mechanisms per mechanism type). 

For the original research project, a matrix of dimensions 11 (x,y pairs) by 100 million was used. This was too large to upload to github. If you plan on generating your own larger random initilization matrix, please be certain you have enough storage. 
```Randpos.npy``` included in the repo has a size of 1.6 GB. Roughly estimating, this means an initilization matrix of 1 million random mechanism parameters would be 1.6 / 10 = 0.16 GB. The 100 million matrix used for the original research project was 16 GB. 



## Contact

If you have any questions or suggestions, feel free to reach out to me:

