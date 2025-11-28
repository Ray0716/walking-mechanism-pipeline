Related papers: 

Simulation: 
(Zhijie Lyu, Wei Liao, Anurag Purwar) A Unified Real-Time Motion Generation Algorithm for Approximate Position Analysis of Planar N-Bar Mechanisms 

Path Synthesis Pipeline: 
(Zhijie Lyu, Anurag Purwar) Deep Learning Conceptual Design of Sit-to-Stand Parallel Motion Six-Bar Mechanisms

Dataset: 
(Anar Nurizada, Rohit Dhaipule, Zhijie Lyu, Anurag Purwar) A Dataset of 3M Single-DOF Planar 4-, 6-, and 8-bar Linkage Mechanisms with Open and Closed Coupler Curves for MachineLearning-Driven Path Synthesis 

This is the project pack for mechanism sampling. 

First, you need to do simulation to create images. To do: 
    1. Start a local motiongen headless server. (Check the local port. It should be 4000/4001/4002/4003)
    2. Run 1.0 ***.ipynb file to generate mechanisms. These are four-bar examples. 
        - You can generate other types of mechanisms. Treat 1.0 notebooks as examples and the starting points. 
        - The types of mechanisms are saved in the .json dictionary (BSIDict)
    3. Once you generate the mechanism samples, run 2.0 script to read image and get their latent representations. 
        - These files are stored in .npy format. The latent representation files are labelled with z.npy 

Next, when you finish generating the samples, you can run main.py to start a local server for mechanism query. 

Contributors to the project, and what they contributed: 

    Anar Nurizada: 
        - The training of VAE. You can see the model in model.py, and the saved training parameters in the .ckpt file. 
        - Project and paper lead. 
    
    Rohit Dhaipule: 
        - Generation of six-bar and eight-bar linkages. 
        - Decomposition algorithm for eight-bar linkages. 
        - Topology dictionary creation. He enumerated the six- and eight-bar topologies.  
        - These samples are uploaded by Anar and can be accessed online. Please check our paper (A dataset of 3M mechanisms) for the link. 

    Zhijie Lyu: 
        - Creation of this project. He designed the code and data structure used by the above two. 
        - Topology dictionary creation. KV and VK structure is his design for smaller data packet. 
        - Modification of the headless simulator to do approximated simulation of P-joint included mechanisms. 
        - Improved normalization method. His paper about Sit-To-Stand mechanism design elaborates the data process. 
