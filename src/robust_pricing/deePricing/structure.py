import logging

from time import time

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from typing import List, Optional, Callable

import torch
from torch import nn
from abc import abstractmethod

from robust_pricing.path_generators import GaussianMartingale, Uniform
from robust_pricing.path_generators._base import _PathGenerator

device = "cuda" if torch.cuda.is_available() else "cpu"


class BaseOption:
    def __init__(self, strike, price, bid=None, ask=None):
        self.strike = strike
        self.price = price
        self._bid = bid
        self._ask = ask
    
    @property
    def bid(self):
        if self._bid is None:
            return self.price
        else:
            return self._bid

    @property
    def ask(self):
        if self._ask is None:
            return self.price
        else:
            return self._ask
        
    def __repr__(self):
        return f'{self.__class__.__name__}(strike={self.strike}, price={self.price})'
    
    def __call__(self, x):
        return self.pay_off(x)
    
    @abstractmethod
    def pay_off(self, x):
        ...


class Put(BaseOption):
    def pay_off(self, x):
        return torch.relu(self.strike - x)
    
    def to_quantlib_object(self, maturity=None, underlying_ticker=None):
        from quantlib.options.options import PutOption
        return PutOption(strike_price=self.strike, maturity=maturity, underlying_ticker=underlying_ticker)


class Call(BaseOption):
    def pay_off(self, x):
        return torch.relu(x - self.strike)
    
    def to_quantlib_object(self, maturity=None, underlying_ticker=None):
        from quantlib.options.options import CallOption
        return CallOption(strike_price=self.strike, maturity=maturity, underlying_ticker=underlying_ticker)


class BasePortfolio(nn.Module):
    DEFAULT_GENERATION_KWARGS = dict(
        type_='gaussian_martingale',
        mean=100,
        variance=3
    )
    
    DEFAULT_GENERATION_KWARGS_Visual = dict(
        type_='uniform',
        min_value=0,
        max_value=300
    )
    
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.training_status = {}
    
    def forward(self, x):
        pass
    
    @abstractmethod
    def loss_function(self, *args, **kwargs):
        ...
    
    @property
    def value(self):
        ...
    
    def optimize(
            self,
            generator: Optional[_PathGenerator] = None,
            number_of_observations=1000,
            number_of_episodes=100,
            optimization_kwargs={}
    ):
        
        if generator is None:
            generator = GaussianMartingale(
                path_length=self.times,
                mean=BasePortfolio.DEFAULT_GENERATION_KWARGS['mean'],
                variance=BasePortfolio.DEFAULT_GENERATION_KWARGS['variance'],
            )
        
        optimizer = torch.optim.Adam(self.parameters(), **optimization_kwargs)
        
        for i in range(number_of_episodes):
            X = generator(number_of_observations)
            loss_ = self.loss_function(self(X))
            optimizer.zero_grad()
            loss_.backward()
            optimizer.step()
    
    def visualize_payoff(self, *args, **kwargs):
        ...
        
    def record_training_status(self, **kwargs):
        for k, v in kwargs.items():
            if k in self.training_status:
                if k == 'epoch':
                    v = max(v, max(self.training_status[k]) + 1)
                self.training_status[k].append(v)
            else:
                self.training_status[k] = [v]
        
    def visualize_training_status(self):
        import plotly.graph_objects as go

        # Create traces
        fig = go.Figure()
        x = self.training_status.get('epoch')
        for k, v in self.training_status.items():
            if not isinstance(v, list):
                continue
            if k == 'epoch':
                continue
            if x is None:
                x = list(range(len(v)))
            fig.add_trace(go.Scatter(
                x=x,
                y=v,
                mode='lines',
                name=k)
            )

        fig.show()
    
    @staticmethod
    def weight_reset(m, zeros=False):
        if isinstance(m, nn.Linear):
            if zeros:
                m.weight = torch.nn.Parameter(m.weight * 10e-7, requires_grad=True)
            else:
                m.reset_parameters()
    
    def reset_weights(self, zeros=False):
        self.apply(lambda x: BasePortfolio.weight_reset(x, zeros=zeros))
    

class OneMaturityOptionPortfolio(BasePortfolio):
    DEFAULT_SIMPLEX_PENALIZATION_FACTOR = 100
    
    def __init__(self, calls: List[BaseOption], puts: List[BaseOption], simplex_penalization_factor=None):
        super().__init__()
        
        self.time_horizon = 1
        
        self.calls = calls
        self.calls_strike_tensor = OneMaturityOptionPortfolio._tensorize_vanilla(calls, 'strike')
        self.calls_price_tensor = OneMaturityOptionPortfolio._tensorize_vanilla(calls, 'price')
        self.calls_bid_tensor = OneMaturityOptionPortfolio._tensorize_vanilla(calls, 'bid')
        self.calls_ask_tensor = OneMaturityOptionPortfolio._tensorize_vanilla(calls, 'ask')
        
        self.puts = puts
        self.puts_strike_tensor = OneMaturityOptionPortfolio._tensorize_vanilla(puts, 'strike')
        self.puts_price_tensor = OneMaturityOptionPortfolio._tensorize_vanilla(puts, 'price')
        self.puts_bid_tensor = OneMaturityOptionPortfolio._tensorize_vanilla(puts, 'bid')
        self.puts_ask_tensor = OneMaturityOptionPortfolio._tensorize_vanilla(puts, 'ask')
        
        self.base_options_count = len(calls) + len(puts)
        
        self._long_price_weights = torch.cat((self.calls_ask_tensor, self.puts_ask_tensor), 0)
        self._short_price_weights = torch.cat((self.calls_bid_tensor, self.puts_bid_tensor), 0)
        
        self.decision_variables = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.base_options_count, 1, bias=False)
        )
        
        self.simplex_penalization_factor = simplex_penalization_factor if simplex_penalization_factor is not None else self.DEFAULT_SIMPLEX_PENALIZATION_FACTOR
    
    @property
    def portfolio(self):
        w = self.weights.detach().numpy()
        options = self.calls + self.puts
        return dict(zip(options, w))
    
    @property
    def weights(self):
        return list(self.parameters())[0][0]
    
    @property
    def value(self):
        # When we buy we buy at the ask price
        long = torch.dot(torch.relu(self.weights), self._long_price_weights)
        
        # When we sell we sell at the bid price
        short = torch.dot(torch.relu(-self.weights), self._short_price_weights)
        
        # w
        return long - short
    
    @staticmethod
    def _tensorize_vanilla(vanillas: List[BaseOption], attribute):
        return torch.Tensor([getattr(c, attribute) for c in vanillas])
    
    def forward(self, x):
        # compute the payoff of each option
        
        calls = x.repeat((1, len(self.calls))) - self.calls_strike_tensor
        
        puts = self.puts_strike_tensor - x.repeat((1, len(self.puts)))
        
        logits = self.decision_variables(torch.cat((calls, puts), 1))
        return logits
    
    def visualize_payoff(self, xmin=0, xmax=500, num_points=100):
        import matplotlib.pyplot as plt
        import numpy as np
        
        x = np.linspace(start=xmin, stop=xmax, num=num_points)
        y = self(torch.Tensor(list(map(lambda x: [x], x)))).detach().numpy() - self.value.detach().numpy()
        
        plt.plot(x, y)
        plt.show()
    
    def loss_function(self, output):
        return self.value - torch.mean(output) + self.simplex_penalization()

    def simplex_penalization(self):
        sum_to_1 = None
        lp1_penalization = None
        
        for parameters in self.parameters():
            if sum_to_1 is None:
                sum_to_1 = torch.relu(torch.sum(parameters) - 1)
            else:
                sum_to_1 += torch.relu(torch.sum(parameters) - 1)
            if lp1_penalization is None:
                lp1_penalization = torch.relu(parameters.abs() - 1).sum()
            else:
                lp1_penalization += torch.relu(parameters.abs() - 1).sum()
    
        return self.simplex_penalization_factor * (sum_to_1 + lp1_penalization)
    

class DiagonalOptionPortfolio(BasePortfolio):
    def __init__(self, option_portfolios: List[OneMaturityOptionPortfolio]):
        super().__init__()
        self.time_horizon = len(option_portfolios)
        
        self.option_portfolios = option_portfolios
        for t in range(self.time_horizon):
            exec(f'self.option_portfolio_{t + 1} = self.option_portfolios[t]')
    
    @property
    def value(self):
        return sum([o.value for o in self.option_portfolios])
    
    def forward(self, x):
        # compute the payoff of each option and sum them
        m = len(x)
        return sum(
            [self.option_portfolios[i](x.select(1, i).reshape(m, 1)) for i in range(self.time_horizon)]
        )
    
    def loss_function(self, output):
        return sum([o.loss_function(output) for o in self.option_portfolios])
    
    def visualize_payoff(self, generator=None):
        import plotly.graph_objs as go
        from plotly.offline import iplot
        import numpy as np
        from scipy.interpolate import griddata
        
        if generator is None:
            generator = Uniform(
                path_length=self.time_horizon,
                mean=BasePortfolio.DEFAULT_GENERATION_KWARGS_Visual['max_value'] / 2,
                variance=BasePortfolio.DEFAULT_GENERATION_KWARGS_Visual['max_value'] ** 2
            )
        
        X_tensor = generator(number_of_samples=1000)
        
        x, y = X_tensor.select(1, 0).detach().numpy().T, X_tensor.select(1, 1).detach().numpy().T
        z = self(X_tensor).detach().numpy().T[0]
        
        xi = np.linspace(x.min(), x.max(), 100)
        yi = np.linspace(y.min(), y.max(), 100)
        X, Y = np.meshgrid(xi, yi)
        
        Z = griddata((x, y), z, (X, Y), method='linear')
        
        # Create a Plotly figure object
        fig = go.Figure()
        
        # Add a 3D surface trace to the figure
        fig.add_trace(go.Surface(x=X, y=Y, z=Z))
        
        # Customize the plot
        fig.update_layout(
            title='3D Surface Plot',
            scene=dict(
                xaxis_title=r'$S_1$',
                yaxis_title=r'$S_2$',
                zaxis_title=r'$f(S_1, S_2)$',
            )
        )
        
        return fig


class TradingStrategySingleTime(BasePortfolio):
    
    def __init__(self, times, num_layers=4, width_layers=100):
        super().__init__()
        self.times = times
        
        self.decision_variables = nn.Sequential(
            *[
                nn.Linear(self.times, width_layers, bias=True),
                nn.ReLU(),
                *[
                    nn.Linear(width_layers, width_layers, bias=True),
                    nn.ReLU(),
                ] * num_layers,
                nn.Linear(width_layers, 1, bias=True),
            ]
        )
    
    @property
    def value(self):
        return 0
    
    def forward(self, x):
        x = x[:, :self.times]
        return self.decision_variables(x)
    
    def loss_function(self, full_paths):
        # number of samples
        m = len(full_paths)
        
        # The differences between the last price seen by the strategy and the next time
        delta = (full_paths.select(1, self.times) - full_paths.select(1, self.times - 1)).reshape(m, 1)
        
        # The average profit/loss
        return -torch.mean(self(full_paths) * delta)


class TradingStrategy(BasePortfolio):
    def __init__(self, time_horizon, trading_kwargs):
        super().__init__()
        
        self.time_horizon = time_horizon
        
        self.strategy_of_time = []
        for t in range(time_horizon - 1):
            strategy_t = TradingStrategySingleTime(
                times=t + 1,
                **trading_kwargs,
            )
            self.strategy_of_time.append(strategy_t)
            exec(f'self.strategy_of_time_{t + 1} = strategy_t')
            
    def loss_function(self, full_paths):
        return sum(
            [s.loss_function(full_paths) for s in self.strategy_of_time]
        )
    
    def forward(self, x):
        return sum(
            [-s.loss_function(x) for s in self.strategy_of_time]
        )

    def optimize(
            self,
            generator: Optional[_PathGenerator] = None,
            number_of_observations=1000,
            number_of_episodes=100,
            optimization_kwargs={}
    ):
    
        if generator is None:
            generator = GaussianMartingale(
                path_length=self.times,
                mean=BasePortfolio.DEFAULT_GENERATION_KWARGS['mean'],
                variance=BasePortfolio.DEFAULT_GENERATION_KWARGS['variance'],
            )
    
        optimizer = torch.optim.Adam(self.parameters(), **optimization_kwargs)
    
        for i in range(number_of_episodes):
            X = generator(number_of_observations)
            loss_ = self.loss_function(X)
            optimizer.zero_grad()
            loss_.backward()
            optimizer.step()


class HedgingStrategy(BasePortfolio):
    DEFAULT_STATS_SLIDING_WINDOW = 1000
    DEFAULT_COVERGENCE_TOLERANCE = 1e-5
    
    def __init__(
            self,
            option_portfolio: DiagonalOptionPortfolio,
            target_function: Callable,
            trading_strategy: TradingStrategy = None,
            penalization_function: 'PenaltyFunction' = None,
            super_hedge=True,
            trading_kwargs=None,
            no_trading_strategy=False,
            stats_sliding_window=None,
            convergence_tolerance=None
    ):
        super().__init__()
        
        self.time_horizon = option_portfolio.time_horizon
        
        if trading_kwargs is None:
            trading_kwargs = {}
        
        if trading_strategy is None:
            if no_trading_strategy:
                trading_strategy = lambda x: 0
            else:
                trading_strategy = TradingStrategy(
                    time_horizon=option_portfolio.time_horizon,
                    trading_kwargs=trading_kwargs,
                )
        else:
            assert trading_strategy.time_horizon == option_portfolio.time_horizon
        
        self.trading_strategy = trading_strategy
        self.option_portfolio = option_portfolio
        
        for t in range(1, self.time_horizon + 1):
            exec(f'self.option_portfolio_{t} = self.option_portfolio.option_portfolio_{t}')
            if t < self.time_horizon:
                if not no_trading_strategy:
                    exec(f'self.strategy_of_time_{t} = self.trading_strategy.strategy_of_time_{t}')
        
        if penalization_function is None:
            from robust_pricing.deePricing import PenaltyFunction
            penalization_function = PenaltyFunction()
        
        self.penalization_function = penalization_function
        self._target_function = target_function
        
        self.super_hedge = super_hedge
        self._value_multiplier = 1 if super_hedge else -1
        
        self.stats_sliding_window = self.DEFAULT_STATS_SLIDING_WINDOW if stats_sliding_window is None else stats_sliding_window

        self.epsilon = self.DEFAULT_COVERGENCE_TOLERANCE if convergence_tolerance is None else convergence_tolerance
    
    @property
    def value(self):
        return self.option_portfolio.value

    def forward(self, x):
        return self.trading_strategy(x) + self.option_portfolio(x)
        # return self.option_portfolio(x)
    
    def replication_error(self, full_paths):
        if self.super_hedge:
            return torch.mean(self.penalization_function(self.target_function(full_paths) - self(full_paths)))
        else:
            return torch.mean(self.penalization_function(self(full_paths) - self.target_function(full_paths)))
    
    def loss_function(self, full_paths):
        return self._value_multiplier * self.value + self.replication_error(full_paths) + sum(
            o.simplex_penalization() for o in self.option_portfolio.option_portfolios
        )
    
    def optimize(
            self,
            generator: Optional[_PathGenerator] = None,
            number_of_observations=1000,
            number_of_episodes=None,
            optimization_kwargs={}
    ):
        if generator is None:
            generator = GaussianMartingale(
                path_length=self.times,
                mean=BasePortfolio.DEFAULT_GENERATION_KWARGS['mean'],
                variance=BasePortfolio.DEFAULT_GENERATION_KWARGS['variance'],
            )
    
        optimizer = torch.optim.Adam(self.parameters(), **optimization_kwargs)
        if number_of_episodes is None:
            number_of_episodes = 1e5
        
        i = 0
        start_time = time()
        while i <= number_of_episodes:
            i += 1
            
            X = generator(number_of_observations)
            loss_ = self.loss_function(X)
            
            optimizer.zero_grad()
            loss_.backward()
            optimizer.step()
            
            portfolio_value = float(self.value.detach().numpy())
            portfolio_values = np.array(self.training_status.get('portfolio_value', []) + [portfolio_value])
            
            if len(portfolio_values) > self.stats_sliding_window:
                sliding_window = portfolio_values[-self.stats_sliding_window:]

                if self._value_multiplier == 1:
                    # We are minimizing
                    cumulative_portfolio_value = np.min(sliding_window)
                else:
                    # We are maximizing
                    cumulative_portfolio_value = np.max(sliding_window)
            else:
                cumulative_portfolio_value = portfolio_value
                
            self.record_training_status(
                epoch=i,
                training_loss_function=float(loss_.detach().numpy()) * self._value_multiplier,
                portfolio_value=portfolio_value,
                replication_error=float(loss_.detach().numpy()) - float(self.value.detach().numpy()) * self._value_multiplier,
                cumulative_portfolio_value=cumulative_portfolio_value,
                execution_time=time() - start_time
            )
            cumulative_portfolio_values = self.training_status.get('cumulative_portfolio_value', [])
            comparison_index = self.stats_sliding_window // 2
            
            logging.debug(self.training_status)
            
            if len(cumulative_portfolio_values) > comparison_index:
                if abs(
                        cumulative_portfolio_values[-comparison_index] - cumulative_portfolio_values[-1]
                ) < self.epsilon:
                    # If the portfolio value has not changed much, we consider the problem as converged and stop optimization.
                    break
                elif abs(
                        cumulative_portfolio_values[-comparison_index] - cumulative_portfolio_values[-1]
                ) > 100:
                    self.training_status['diverged'] = True
                    break
        
        self.training_status["total_execution_time"] = time() - start_time
        self.training_status["total_num_episodes"] = i
        self.training_status["best_bound"] = self.training_status["cumulative_portfolio_value"][-1]
        
    def target_function(self, x):
        fx = self._target_function(x)
        return fx.reshape(len(x), 1)
    