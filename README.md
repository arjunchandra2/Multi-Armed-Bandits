# Multi-Armed-Bandits

This repo contains code to run simulations for three multi-armed bandit algorithms:

1. Explore then Commit
2. Epsilon-Greedy
3. Successive Elimination 

The algorithms are implemented for two Gaussian arms. The optimal arm is $\mathcal{N}(0,1)$ and the sub-optimal arm is $\mathcal{N}(-\Delta,1)$. In order to run the simulations, first install the required dependencies:

`pip install -r requirements.txt`

You can then run `main.py` to run the simulations with default parameters. The script should output a plot with theoretical bounds and empirical results. The script will also create a `logs` directory in the working directory and save log files as the simulations are run so that intermediate results are saved in case of a crash. You can also experiment with differents parameters for the simulation by changing them in `main.py`. These parameters are described below. 

Global parameters: 
- `horizon`: The horizon for each MAB algorithm. Default is 1000. 
- `num_simulations`: The number of simulations to run for each MAB algorithm with each gap parameter. Default is 100. 
- `gap_parameters`: A list of suboptimality gap parameters for the suboptimal arm. Default is `[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]`. 
- `seed`: Seed for reproducibility. Default is 592. 

Algorithm-specific parameters: 
- `c`: Parameter for Epsilon-Greedy which uses $\epsilon_t = \text{min} \left( 1, \frac{c}{t} \right)$. Default is 50. 
