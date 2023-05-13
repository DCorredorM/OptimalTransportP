from robust_pricing.deePricing import HedgingStrategy
from copy import deepcopy



def run_for_gamma(
    gamma, 
    generator,
    option_portfolio,
    target_function,
    number_of_observations=100,
    number_of_episodes=200,
    test_data=None,
    no_trading_strategy=True,
    zeros=True
):
    f = target_function
    option_portfolio.reset_weights(zeros=zeros)
    
    # define sup/sub headging strategies
    h_sup = HedgingStrategy(
        option_portfolio=deepcopy(option_portfolio),
        target_function=f,
        super_hedge=True,
        no_trading_strategy=no_trading_strategy
    )
    h_sub = HedgingStrategy(
        option_portfolio=deepcopy(option_portfolio),
        target_function=f,
        super_hedge=False,
        no_trading_strategy=no_trading_strategy
    )
    
    # Set the given gamma
    h_sup.penalization_function.gamma = gamma
    h_sub.penalization_function.gamma = gamma
        
    # optimize for the given generator model     
    h_sup.optimize(
        generator=generator, 
        number_of_observations=number_of_observations, 
        number_of_episodes=number_of_episodes,
    )

    h_sub.optimize(
        generator=generator, 
        number_of_observations=number_of_observations, 
        number_of_episodes=number_of_episodes,
    )
    
    # Record replication errors in the real model
    if test_data is not None:
        replication_errors = float(h_sup.replication_error(test_data).detach()), float(h_sub.replication_error(test_data).detach())
    else:
        replication_errors = float(h_sup.training_status['replication_error'][-1].detach()), float(h_sub.training_status['replication_error'][-1].detach())
    
    
    bounds = float(h_sup.value), float(h_sub.value)
    
    return bounds[::-1], replication_errors[::-1]


def plot_beta(p_factor=2, max_gamma=11):
    p = PenaltyFunction(initial_gamma=1, penalty_type_kwargs=dict(p_factor=p_factor))
    import plotly.graph_objects as go
    # Create traces
    fig = go.Figure()
    for gamma in range(1, max_gamma):
        p.gamma = gamma
        fig.add_trace(go.Scatter(
            x=x,
            y=p(torch.Tensor(x)),
            mode='lines',
            name=r'$\gamma='+str(gamma)+r'$')
        )
    fig.show()
    
    
#     for gamma in range(1, 11):
#         p.gamma = gamma
#         plt.plot(x, p(torch.Tensor(x)), label=r'$\gamma$='+str(gamma))
#     plt.legend()
#     plt.xlabel(r'$x$')
#     plt.ylabel(r'$\beta_{\gamma}(x)$')
#     plt.show()
    
    fig = go.Figure()
    for gamma in range(1, max_gamma):
        p.gamma = gamma
        fig.add_trace(go.Scatter(
            x=x,
            y=p.derivative(torch.Tensor(x)),
            mode='lines',
            name=r'$\gamma='+str(gamma)+r'$')
        )
    fig.show()
    
#     for gamma in range(1, 11):
#         p.gamma = gamma
#         plt.plot(x, p.derivative(torch.Tensor(x)), label=r'$\gamma='+str(gamma)+r'$')
#     plt.legend()
#     plt.xlabel(r'$x$')
#     plt.ylabel(r'$\beta_{\gamma}(x)$')


def hist_2d(data, time_1=0, time_2=1):
    import plotly.express as px
    import pandas as pd
    
    path_length = data.shape[-1]
    
    df = pd.DataFrame(data, columns = [f'S_{i+1}'for i in range(path_length)])
    fig = px.density_heatmap(df, x=f'S_{time_1 + 1}', y=f'S_{time_2 + 1}', marginal_x="histogram", marginal_y="histogram", nbinsx=80, nbinsy=80)
    fig.show()
    
    
def run_for_gammas_in(
    generator, 
    test_data,
    option_portfolio,
    target_function, 
    low=1, 
    high=41, 
    step=1, 
    number_of_observations=1000, 
    number_of_episodes=1000,
    no_trading_strategy=True,
    zeros=True
): 
    # define the number_of_observations to be used in every iteration

    gammas = list(range(low, high, step))


    lower_bound = []
    upper_bound = []

    sup_replication_error = []
    sub_replication_error = []


    for gamma in gammas:
        bounds, replication_errors = run_for_gamma(
            gamma, 
            option_portfolio=option_portfolio,
            target_function=target_function, 
            number_of_observations=number_of_observations,
            number_of_episodes=number_of_episodes,
            generator=generator,
            test_data=test_data,
            no_trading_strategy=no_trading_strategy,
            zeros=zeros
        )
        print(bounds)
        print(replication_errors)   

        lower_bound.append(bounds[0])
        upper_bound.append(bounds[-1])

        sup_replication_error.append(replication_errors[0])
        sub_replication_error.append(replication_errors[-1])
    
    return lower_bound, upper_bound, sup_replication_error, sub_replication_error
