```python
import plotly.express as px
import panel as pn
pn.extension('plotly')
import locale
locale.setlocale(locale.LC_ALL, 'en_CA.UTF-8')
import warnings
warnings.filterwarnings('ignore')
import plotly.express as px
import hvplot.pandas
import matplotlib.pyplot as plt
from panel.interact import interact
from panel import widgets
import numpy as np
import pandas as pd
import datetime as dt
from IPython.display import display
import pandas_datareader as pdr
import time
from scipy.optimize import minimize
import plotly.graph_objects as go
```


```python
def get_age():
    age_dict = {
        10 : "Above 60 years", 
        20 : "Between 50 and 60 years" ,
        30 : "Between 40 and 50 years" ,
        40 : "Between 30 and 40 years" ,
        50 : "Between 20 and 30 years" 
    }
    age_prob = [0.2] * 5
    age_choice = np.random.choice(list(age_dict.values()), size = 1, p = age_prob)
    age_selection = [age for age, value in age_dict.items() if value == age_choice][0]
    return age_selection
```


```python
def get_inv_horizon():
    inv_horizon_dict = {
        10 : "<= 1 year", 
        20 : "> 1 year and <= 3 years" ,
        30 : "> 3 years and <= 5 years",
        40 : "> 5 years and <= 7 years",
        50 : "> 7 years" 
    }
    inv_horizon_prob = [0.2] * 5
    inv_horizon_choice = np.random.choice(list(inv_horizon_dict.values()), size = 1, p = inv_horizon_prob)
    inv_hzn_selection = [inv_hzn for inv_hzn, value in inv_horizon_dict.items() if value == inv_horizon_choice][0]
    return inv_hzn_selection
```


```python
def get_goal():
    goal_dict = {
        10 : "Minimize Risk, preservation of capital even if not meeting inflation", 
        20 : "Modest appreciation with atleast meeting inflation" ,
        30 : "Periodic Income" ,
        40 : "Some Growth + Periodic Income" ,
        50 : "High Returns and Investment Growth" 
    }
    goal_prob = [0.2] * 5
    goal_choice = np.random.choice(list(goal_dict.values()), size = 1, p = goal_prob)
    goal_selection = [goal for goal, value in goal_dict.items() if value == goal_choice][0]
    return goal_selection
```


```python
def get_25percent_decline_response():
    response_dict = {
        10 : "Take out money immediately", 
        20 : "Move money to other investments" ,
        30 : "Concerned but wait a  bit more to see if things improved" ,
        40 : "Expected Risk and market downturn, leave money in place and expect things to improve" ,
        50 : "Can tolerate higher degrees of fluctuation in the value, to invest more expecting future growth" 
    }
    response_prob = [0.2] * 5
    response_choice = np.random.choice(list(response_dict.values()), size = 1, p = response_prob)
    response_selection = [response for response, value in response_dict.items() if value == response_choice][0]
    return response_selection
```


```python
def get_inv_knowledge():
    knowledge_dict = {
        10 : "First time investor. No familiarity or interest", 
        20 : "Little familiarity" ,
        30 : "Understand the importance of diversification" ,
        40 : "Understand market fluctuation, different market sectors and growth characteristics requirements",
        50 : "Experienced with all investment classes" 
    }
    knowledge_prob = [0.2] * 5
    knowledge_choice = np.random.choice(list(knowledge_dict.values()), size = 1, p = knowledge_prob)
    knowledge_selection = [knowledge for knowledge, value in knowledge_dict.items() if value == knowledge_choice][0]
    return knowledge_selection
```


```python
def get_exp_return():
    exp_return_dict = {
        10 : "Inflation + <=1%"      , 
        20 : "Inflation + >1% & <=3%",
        30 : "Inflation + >3% & <=6%",
        40 : "Inflation + >6% & <=9%",
        50 : "Inflation + >9%" 
    }
    exp_return_prob = [0.2] * 5
    exp_return_choice = np.random.choice(list(exp_return_dict.values()), size = 1, p = exp_return_prob)
    exp_return_selection = [ret for ret, value in exp_return_dict.items() if value == exp_return_choice][0]
    return exp_return_selection
```


```python
def get_portfolio_type(total):
    if  total >= 260: return "Very Aggressive"
    elif  260 > total >= 210: return "Aggressive"
    elif  210 > total >= 150: return "Moderate & Balanced"
    elif  150 > total >= 80: return "Conservative"
    elif  80 > total: return "Very Conservative"
```


```python
age = 30 # get_age()
hzn = 30 # get_inv_horizon()
goal = 30 # get_goal()
decl_resp = 30 # get_25percent_decline_response()
expr = 30 # get_inv_knowledge()
eret = 30 # get_exp_return()

total = age + hzn + goal + decl_resp + expr + eret
risk_score_quantile = round(total / 300,3)

port_type = get_portfolio_type(total)

allo_dict = {
    "Very Aggressive": {"Stocks" : "90%", "Bonds" : "10%", "Cash": "0%"},
    "Aggressive": {"Stocks" : "70%", "Bonds" : "25%", "Cash": "5%"},
    "Moderate & Balanced": {"Stocks" : "50%", "Bonds" : "40%", "Cash": "10%"},
    "Conservative": {"Stocks" : "35%", "Bonds" : "50%", "Cash": "15%"},
    "Very Conservative": {"Stocks" : "20%", "Bonds" : "60%", "Cash": "20%"}
}
```


```python
# ETF tickers for the main assest classes
# local_equities = ['^GSPC','^IXIC','^W5000','^DJI','^GSPTSE']
# foreign_equities = ['^FCHI','^GDAXI','^IBEX','^N225','^HSI']
# local_debt = ['VBMFX','FBNDX','LQD','PHB']
# intl_debt = ['PCY','BWX','EMB','AGG']
# cash_equivalents = ['BIL','GSY']
# alternatives_commodities = ['XOP','VNQ','HVPE.L','RWO','XGD.TO']
# tickers = local_equities + foreign_equities + local_debt + intl_debt + cash_equivalents + alternatives_commodities
tickers = ['QQQ','VOO','VTI','GDX','IWM','BND','XLE','IVV','AGG','VTV','TLT','IBB','GLD','SPY','VGK','VWO','IJH','QID','SHY','SQQQ','VEA','EFA','IWF','IJR']
```


```python
start_date = (dt.date.today() - dt.timedelta(10*365)).isoformat()
end_date = dt.date.today().isoformat()
close_prices_rawdata = pd.DataFrame(columns=tickers)
for ticker in tickers:
    close_prices_rawdata[ticker] = pdr.DataReader(ticker,'yahoo',start_date,end_date)['Adj Close']
close_prices_rawdata.fillna(method='bfill',inplace=True)
```


```python
split_row = int(close_prices_rawdata.shape[0] * 0.8)
test_data = close_prices_rawdata.iloc[split_row:]

close_prices_rawdata = close_prices_rawdata.iloc[0:split_row]
```


```python
# Initial Weights
num_assets = len(tickers)
init_weights = num_assets * [1 / num_assets]

returns = close_prices_rawdata.pct_change().dropna()
log_returns = np.log(close_prices_rawdata/close_prices_rawdata.shift(1)).dropna()

bounds = tuple((0,1) for i in range(len(close_prices_rawdata.columns)))
init_weights = [1/(len(close_prices_rawdata.columns))] * (len(close_prices_rawdata.columns))
```


```python
mkt = pdr.DataReader('^GSPC','yahoo',start_date,end_date)['Adj Close']
mkt_data = mkt.iloc[split_row:]
mkt_returns = mkt_data.fillna(method='bfill').pct_change().dropna()
```


```python
def calculate_return(weights):
    weights = np.array(weights)
    return np.sum(log_returns.mean() * weights) * 252
    # return np.dot(returns.mean(), weights) * 252
    # return np.sum(returns.mean() * weights) * 252
```


```python
def calculate_sharpe(weights):
    weights = np.array(weights)
    return calculate_return(weights) / calculate_std_dev(weights)
```


```python
def negative_sharpe(weights):
    weights = np.array(weights)
    return calculate_sharpe(weights) * -1
    #return -calculate_sharpe(w)
```


```python
def maximize_return(weights):
    weights = np.array(weights)
    return calculate_return(weights) * -1
    # return np.dot(returns.mean(), weights) * 252
    # return np.sum(returns.mean() * weights) * 252
```


```python
def minimize_volatility(weights):
    weights = np.array(weights)
    return np.sqrt(np.dot(weights.T, np.dot(log_returns.cov()*252, weights)))
    # return np.sqrt(np.dot(weights, np.dot(returns.cov(), weights)) * 252)
```


```python
def calculate_std_dev(weights):
    weights = np.array(weights)
    return np.sqrt(np.dot(weights.T, np.dot(returns.cov(), weights)) * 252)
```


```python
def check_sum(weights):
    weights = np.array(weights)
    #return 0 if sum of the weights is 1
    return np.sum(weights)-1
```


```python
def print_weights(weights):
    w = [round(i*100,2)for i in weights]
    w = pd.DataFrame(w, columns = ['Weight (%)'], index = returns.columns)
    print("\n",w[w['Weight (%)']>0])
    new_w = w[w['Weight (%)']>0]
    return new_w
```


```python
def negative_return(w):
    return -calculate_return(w)
```


```python
# Calculate estimated return for each stock
individual_returns = returns.mean() * 252
#display(individual_returns)
# Estimate individual stock risks for comparison
individual_risks = np.std(returns) * np.sqrt(252)
#display(individual_risks)
individual_analysis = pd.concat([individual_returns,individual_risks],axis=1)
individual_analysis.columns = ['Returns','Risks']
individual_analysis['Sharpe_Ratios'] = individual_analysis.Returns / individual_analysis.Risks

sorted_securities = individual_analysis.sort_values(by='Returns', ascending = False)
```


```python
# Get minimum and maximum risk possible given the asset universe
cons_min_risk = {'type':'eq','fun': check_sum}
min_risk = round(minimize(fun=minimize_volatility, x0=init_weights, bounds=bounds,constraints=cons_min_risk)['fun'],3)
# max_risk = mr = round(minimize(fun=min_max_volatility, x0=init_weights, bounds=bounds,constraints=cons_min_risk)['fun'],3) * -1
max_risk = round(individual_risks.max(),3)
all_risks = np.linspace(min_risk,max_risk).round(3)
```


```python
start = time.time()

max_returns = []
weights_min_vol_max_ret = []
for each_risk_num in all_risks: 
    cons_returns = ({'type':'eq', 'fun': lambda w: w.sum() - 1},
                    {'type':'eq', 'fun': lambda w: minimize_volatility(w) - each_risk_num})   
    ret_given_risk_results = minimize(maximize_return, x0=init_weights, bounds = bounds, constraints = cons_returns)
    max_returns.append(calculate_return(ret_given_risk_results['x']))
    weights_min_vol_max_ret.append(ret_given_risk_results['x'])
    
max_return_volatility = minimize_volatility(weights_min_vol_max_ret[np.argmax(max_returns)])
frontier_return = np.linspace(0,max_return_volatility,100)

frontier_volatility = []
weights_array = []
returns_array = []

for possible_return in frontier_return:
    cons_risk = ({'type':'eq', 'fun':check_sum},
                 {'type':'eq', 'fun': lambda w: calculate_return(w) - possible_return})
    
    results = minimize(minimize_volatility,init_weights,method='SLSQP', bounds=bounds, constraints=cons_risk)
    frontier_volatility.append(results['fun'])
    weights_array.append(results['x'])
    returns_array.append(calculate_return(results['x']))

ef_df = pd.DataFrame({'Returns':returns_array, 'Volatility':frontier_volatility})
ef_df['Sharpe'] = ef_df['Returns'] /  ef_df['Volatility']
ef_df_pos_sharpe = ef_df[ef_df['Sharpe']>= 1]
opt_sharpe = ef_df_pos_sharpe[ef_df_pos_sharpe.Sharpe == ef_df_pos_sharpe.Sharpe.max()]
print('Elapsed Time: %.2f seconds' % (time.time() - start))
```

    Elapsed Time: 111.77 seconds
    


```python
goal_risk = np.percentile(ef_df_pos_sharpe.Volatility,q=(risk_score_quantile*100))
goal_cons = ({'type':'eq', 'fun': lambda w: w.sum() - 1},
             {'type':'eq', 'fun': lambda w: calculate_std_dev(w) - goal_risk})

opt_goal = minimize(negative_return, x0=init_weights, bounds = bounds, constraints=goal_cons)
opt_goal_weights = opt_goal['x']

print("Computational Optimized weights for our risk goal:")
goal_allo = print_weights(opt_goal_weights)
print("\n")
print("Indicators of Optimal Portfolio given our risk goal:")
print("Return {:.3f}, Volatility {:.3f}, Sharpe {:.3f}".format(calculate_return(opt_goal_weights), calculate_std_dev(opt_goal_weights),calculate_sharpe(opt_goal_weights)))
```

    Computational Optimized weights for our risk goal:
    
          Weight (%)
    QQQ       22.12
    BND       16.29
    AGG       12.82
    VTV       17.14
    TLT       30.25
    IBB        1.39
    
    
    Indicators of Optimal Portfolio given our risk goal:
    Return 0.078, Volatility 0.062, Sharpe 1.269
    


```python
# Set constrains and initial guess for optimal Sharpe
cons_opt_sharpe = {'type': 'eq', 'fun': lambda w: w.sum() - 1}

opt_results = minimize(negative_sharpe, x0=init_weights, bounds = bounds, constraints=cons_opt_sharpe)
opt_w = opt_results['x']

print("Computational Optimized weights:")
print_weights(opt_w)
print("\n")
print("Indicators of Optimal Portfolio:")
print("Return {:.3f}, Volatility {:.3f}, Sharpe {:.3f}".format(calculate_return(opt_w), calculate_std_dev(opt_w),calculate_sharpe(opt_w)))
```

    Computational Optimized weights:
    
          Weight (%)
    QQQ        3.89
    BND       17.23
    AGG        5.81
    VTV        3.14
    IBB        0.28
    SHY       69.64
    
    
    Indicators of Optimal Portfolio:
    Return 0.020, Volatility 0.014, Sharpe 1.455
    


```python
# Set constrains and initial guess for minimum variance
cons_opt_var = {'type':'eq','fun': lambda w: np.sum(w) - 1}
opt_var_results = minimize(fun=calculate_std_dev,bounds=bounds,x0=init_weights,constraints=cons_opt_var)
opt_var_w = opt_var_results['x']
print("Computational Optimized weights:")
print_weights(opt_var_w)
print("\n")
print("Indicators of Minimum Variance Portfolio:")
print("Return {:.3f}, Volatility {:.3f}, Sharpe {:.3f}".format(calculate_return(opt_var_w), calculate_std_dev(opt_var_w),calculate_sharpe(opt_var_w)))
```

    Computational Optimized weights:
    
           Weight (%)
    QQQ        51.91
    VOO         0.06
    VTV         0.01
    IBB         0.08
    QID        14.55
    SHY        24.31
    SQQQ        7.99
    IWF         1.00
    IJR         0.08
    
    
    Indicators of Minimum Variance Portfolio:
    Return -0.026, Volatility 0.004, Sharpe -6.861
    


```python
start = time.time()

#calculate mean daily return and covariance of daily returns
mean_daily_returns = returns.mean()
# cov_matrix = returns.cov() * 252
cov_matrix = returns.cov()

#set number of runs of random portfolio weights
num_portfolios = num_assets * 20000

#set up array to hold results
results = np.zeros((num_portfolios,3+num_assets))
portfolio_returns_records = [] # Define an empty array for portfolio returns
portfolio_volatility_records = [] # Define an empty array for portfolio volatility
portfolio_weights_records = [] # Define an empty array for asset weights
portfolio_sharpe_ratio_records = []# Define an empty list for sharpe ratios

for i in range(num_portfolios):
    #select random weights for portfolio holdings
    # weights = np.random.dirichlet(np.ones(num_assets),size=1)
    mc_weights = np.array(np.random.random(num_assets))
    #rebalance weights to sum to 1
    mc_weights /= np.sum(mc_weights)
    
    #calculate portfolio return and volatility
    portfolio_return = np.sum(mean_daily_returns * mc_weights) * 252
    #port_ret = np.sum(log_ret.mean() * wts)
    #port_ret = (port_ret + 1) ** 252 - 1
    portfolio_std_dev = np.sqrt(np.dot(np.transpose(mc_weights),np.dot(cov_matrix, mc_weights))) * np.sqrt(252)
    
    #store results in results array
    results[i,0] = portfolio_return
    results[i,1] = portfolio_std_dev
    #store Sharpe Ratio (return / volatility) - risk free rate element excluded for simplicity
    results[i,2] = results[i,0] / results[i,1]
    #iterate through the weight vector and add data to results array
    for j in range(len(mc_weights)):
        results[i,j+3] = mc_weights[j]
    portfolio_returns_records.append(portfolio_return)
    portfolio_volatility_records.append(portfolio_std_dev)
    portfolio_weights_records.append(mc_weights)
    portfolio_sharpe_ratio_records.append(results[i,2])

#convert results array to Pandas DataFrame
portfolio_returns_records = np.array(portfolio_returns_records) 
portfolio_volatility_records = np.array(portfolio_volatility_records)
portfolio_weights_records = np.array(portfolio_weights_records)
portfolio_sharpe_ratio_records = np.array(portfolio_sharpe_ratio_records)
stocks_weights = [f'{stock} weight' for stock in returns.columns.tolist()]
# for counter, symbol in enumerate(closes.columns.tolist()):
#     [symbol+' weight'] = [w[counter] for w in p_weights]
column_order = ['Returns', 'Volatility', 'Sharpe_Ratios'] + [stock+' Weight' for stock in tickers]    
results_frame = pd.DataFrame(results,columns=column_order)
print('Elapsed Time: %.2f seconds' % (time.time() - start))

# #locate position of portfolio with highest Sharpe Ratio
# max_sharpe_port = results_frame.iloc[results_frame['Sharpe_Ratios'].idxmax()]
# #max_sharpe_port = results_frame.iloc[results_frame['Sharpe_Ratios'].argmax()]
# print ('Max Sharpe Ratio Portfolio')
# print('*'*30)
# print(max_sharpe_port)
# print('*'*30)
# print ('Min Volatility Portfolio')
# print('*'*30)
# #locate positon of portfolio with minimum standard deviation
# min_vol_port = results_frame.iloc[results_frame['Volatility'].idxmin()]
# #max_sharpe_port = results_frame.iloc[results_frame['Sharpe_Ratios'].argmax()]
# print(min_vol_port)
```

    Elapsed Time: 125.95 seconds
    


```python
plot_boundary = results_frame.Volatility.max()
plt.figure(figsize=(12,7))
plt.title('Effecient Frontier')
plt.xlabel('Volatility')
plt.ylabel('Return')
plt.scatter(results_frame[results_frame.Volatility <= plot_boundary].Volatility,results_frame[results_frame.Volatility <= plot_boundary].Returns,c=results_frame.Sharpe_Ratios,cmap='RdYlBu')
plt.colorbar()
#plt.plot(frontier_volatility,frontier_return, 'b--', linewidth=3)
plt.plot(ef_df[ef_df.Volatility <= plot_boundary].Volatility,ef_df[ef_df.Volatility <= plot_boundary].Returns, 'b--', linewidth=3)
plt.plot(calculate_std_dev(opt_goal_weights),calculate_return(opt_goal_weights), 'y*', markersize = 15.0, label='Goal Portfolio')
plt.plot(opt_sharpe.Volatility,opt_sharpe.Returns, c='red', marker='D', markersize = 5.0, label='maximum sharpe')
plt.scatter(calculate_std_dev(opt_var_w),calculate_return(opt_var_w), c='green', s=50, label='Minimum Variance Portfolio')
plt.legend(frameon=False)
plt.show()

```


    
![png](README_files/README_30_0.png)
    



```python
mkt_cum_rets = (1 + mkt_returns).cumprod()
goal_portfolio_returns = test_data.pct_change().dropna().dot(opt_goal_weights)
cummulative_returns = (1 + goal_portfolio_returns).cumprod()

# plt.figure(figsize=(12,7))
# (cummulative_returns*100000).plot(c='darkorange', linewidth=3)
# (mkt_cum_rets*100000).plot(c='red', linewidth=3)
# plt.title("Backtesting with market returns for Investor time Horizon")
# plt.show()

plt1 = (cummulative_returns*100000).hvplot(c='darkorange', line_width=3,label='Portfolio') 
plt2 = (mkt_cum_rets*100000).hvplot(c='red', line_width=3,label='Market')
combplt = (plt1 * plt2).opts(legend_position='top_left',title = 'Backtesting Portfolio value of $100k investment',yformatter='%.00f')
combplt #.opts(legend_position='top_left',title = 'Backtesting Portfolio value of $100k investment')

#.opts(legend_position='top_left',title = 'Backtesting Portfolio value of $100k investment')
```






<div id='3473'>





  <div class="bk-root" id="02a4ac36-1987-4a0e-be24-54988f2083aa" data-root-id="3473"></div>
</div>
<script type="application/javascript">(function(root) {
  function embed_document(root) {
    var docs_json = {"2af5d7b3-07b5-4d88-98b6-8f0a1b16d380":{"roots":{"references":[{"attributes":{"source":{"id":"3517"}},"id":"3524","type":"CDSView"},{"attributes":{"days":[1,4,7,10,13,16,19,22,25,28]},"id":"3533","type":"DaysTicker"},{"attributes":{"data_source":{"id":"3517"},"glyph":{"id":"3520"},"hover_glyph":null,"muted_glyph":{"id":"3522"},"nonselection_glyph":{"id":"3521"},"selection_glyph":null,"view":{"id":"3524"}},"id":"3523","type":"GlyphRenderer"},{"attributes":{},"id":"3518","type":"Selection"},{"attributes":{"months":[0,2,4,6,8,10]},"id":"3537","type":"MonthsTicker"},{"attributes":{},"id":"3511","type":"DatetimeTickFormatter"},{"attributes":{"months":[0,1,2,3,4,5,6,7,8,9,10,11]},"id":"3536","type":"MonthsTicker"},{"attributes":{"months":[0,4,8]},"id":"3538","type":"MonthsTicker"},{"attributes":{"format":"%.00f"},"id":"3510","type":"PrintfTickFormatter"},{"attributes":{"days":[1,15]},"id":"3535","type":"DaysTicker"},{"attributes":{},"id":"3540","type":"YearsTicker"},{"attributes":{"margin":[5,5,5,5],"name":"HSpacer05407","sizing_mode":"stretch_width"},"id":"3474","type":"Spacer"},{"attributes":{},"id":"3497","type":"PanTool"},{"attributes":{"children":[{"id":"3474"},{"id":"3479"},{"id":"3765"}],"margin":[0,0,0,0],"name":"Row05403","tags":["embedded"]},"id":"3473","type":"Row"},{"attributes":{"callback":null,"formatters":{"@{Date}":"datetime"},"renderers":[{"id":"3552"}],"tags":["hv_created"],"tooltips":[["Date","@{Date}{%F %T}"],["Adj Close","@{Adj_Close}"]]},"id":"3478","type":"HoverTool"},{"attributes":{},"id":"3484","type":"LinearScale"},{"attributes":{},"id":"3573","type":"UnionRenderers"},{"attributes":{"margin":[5,5,5,5],"name":"HSpacer05408","sizing_mode":"stretch_width"},"id":"3765","type":"Spacer"},{"attributes":{"end":153406.2519010024,"reset_end":153406.2519010024,"reset_start":80052.37481544711,"start":80052.37481544711,"tags":[[["0","0",null]]]},"id":"3476","type":"Range1d"},{"attributes":{"callback":null,"formatters":{"@{Date}":"datetime"},"renderers":[{"id":"3523"}],"tags":["hv_created"],"tooltips":[["Date","@{Date}{%F %T}"],["0","@{A_0}"]]},"id":"3477","type":"HoverTool"},{"attributes":{"axis_label":"Date","bounds":"auto","formatter":{"id":"3511"},"major_label_orientation":"horizontal","ticker":{"id":"3489"}},"id":"3488","type":"DatetimeAxis"},{"attributes":{"text":"Backtesting Portfolio value of $100k investment","text_color":{"value":"black"},"text_font_size":{"value":"12pt"}},"id":"3480","type":"Title"},{"attributes":{"label":{"value":"Market"},"renderers":[{"id":"3552"}]},"id":"3575","type":"LegendItem"},{"attributes":{"below":[{"id":"3488"}],"center":[{"id":"3491"},{"id":"3495"},{"id":"3544"}],"left":[{"id":"3492"}],"margin":[5,5,5,5],"min_border_bottom":10,"min_border_left":10,"min_border_right":10,"min_border_top":10,"plot_height":300,"plot_width":700,"renderers":[{"id":"3523"},{"id":"3552"}],"sizing_mode":"fixed","title":{"id":"3480"},"toolbar":{"id":"3502"},"x_range":{"id":"3475"},"x_scale":{"id":"3484"},"y_range":{"id":"3476"},"y_scale":{"id":"3486"}},"id":"3479","subtype":"Figure","type":"Plot"},{"attributes":{"num_minor_ticks":5,"tickers":[{"id":"3529"},{"id":"3530"},{"id":"3531"},{"id":"3532"},{"id":"3533"},{"id":"3534"},{"id":"3535"},{"id":"3536"},{"id":"3537"},{"id":"3538"},{"id":"3539"},{"id":"3540"}]},"id":"3489","type":"DatetimeTicker"},{"attributes":{"end":1610064000000.0,"reset_end":1610064000000.0,"reset_start":1547164800000.0,"start":1547164800000.0,"tags":[[["Date","Date",null]]]},"id":"3475","type":"Range1d"},{"attributes":{},"id":"3498","type":"WheelZoomTool"},{"attributes":{},"id":"3493","type":"BasicTicker"},{"attributes":{"axis":{"id":"3492"},"dimension":1,"grid_line_color":null,"ticker":null},"id":"3495","type":"Grid"},{"attributes":{"line_color":"darkorange","line_width":3,"x":{"field":"Date"},"y":{"field":"0"}},"id":"3520","type":"Line"},{"attributes":{"axis":{"id":"3488"},"grid_line_color":null,"ticker":null},"id":"3491","type":"Grid"},{"attributes":{},"id":"3486","type":"LinearScale"},{"attributes":{"click_policy":"mute","items":[{"id":"3545"},{"id":"3575"}],"location":"top_left"},"id":"3544","type":"Legend"},{"attributes":{},"id":"3496","type":"SaveTool"},{"attributes":{"line_alpha":0.1,"line_color":"darkorange","line_width":3,"x":{"field":"Date"},"y":{"field":"0"}},"id":"3521","type":"Line"},{"attributes":{"line_alpha":0.2,"line_color":"red","line_width":3,"x":{"field":"Date"},"y":{"field":"Adj Close"}},"id":"3551","type":"Line"},{"attributes":{"axis_label":"","bounds":"auto","formatter":{"id":"3510"},"major_label_orientation":"horizontal","ticker":{"id":"3493"}},"id":"3492","type":"LinearAxis"},{"attributes":{"days":[1,8,15,22]},"id":"3534","type":"DaysTicker"},{"attributes":{"months":[0,6]},"id":"3539","type":"MonthsTicker"},{"attributes":{"base":24,"mantissas":[1,2,4,6,8,12],"max_interval":43200000.0,"min_interval":3600000.0,"num_minor_ticks":0},"id":"3531","type":"AdaptiveTicker"},{"attributes":{"data_source":{"id":"3546"},"glyph":{"id":"3549"},"hover_glyph":null,"muted_glyph":{"id":"3551"},"nonselection_glyph":{"id":"3550"},"selection_glyph":null,"view":{"id":"3553"}},"id":"3552","type":"GlyphRenderer"},{"attributes":{"data":{"0":{"__ndarray__":"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","dtype":"float64","order":"little","shape":[503]},"A_0":{"__ndarray__":"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","dtype":"float64","order":"little","shape":[503]},"Date":{"__ndarray__":"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","dtype":"float64","order":"little","shape":[503]}},"selected":{"id":"3518"},"selection_policy":{"id":"3542"}},"id":"3517","type":"ColumnDataSource"},{"attributes":{"days":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]},"id":"3532","type":"DaysTicker"},{"attributes":{},"id":"3547","type":"Selection"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"3477"},{"id":"3478"},{"id":"3496"},{"id":"3497"},{"id":"3498"},{"id":"3499"},{"id":"3500"}]},"id":"3502","type":"Toolbar"},{"attributes":{"source":{"id":"3546"}},"id":"3553","type":"CDSView"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"3501","type":"BoxAnnotation"},{"attributes":{"base":60,"mantissas":[1,2,5,10,15,20,30],"max_interval":1800000.0,"min_interval":1000.0,"num_minor_ticks":0},"id":"3530","type":"AdaptiveTicker"},{"attributes":{"data":{"Adj Close":{"__ndarray__":"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","dtype":"float64","order":"little","shape":[503]},"Adj_Close":{"__ndarray__":"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","dtype":"float64","order":"little","shape":[503]},"Date":{"__ndarray__":"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","dtype":"float64","order":"little","shape":[503]}},"selected":{"id":"3547"},"selection_policy":{"id":"3573"}},"id":"3546","type":"ColumnDataSource"},{"attributes":{"overlay":{"id":"3501"}},"id":"3499","type":"BoxZoomTool"},{"attributes":{"mantissas":[1,2,5],"max_interval":500.0,"num_minor_ticks":0},"id":"3529","type":"AdaptiveTicker"},{"attributes":{"line_alpha":0.2,"line_color":"darkorange","line_width":3,"x":{"field":"Date"},"y":{"field":"0"}},"id":"3522","type":"Line"},{"attributes":{},"id":"3500","type":"ResetTool"},{"attributes":{"line_alpha":0.1,"line_color":"red","line_width":3,"x":{"field":"Date"},"y":{"field":"Adj Close"}},"id":"3550","type":"Line"},{"attributes":{"label":{"value":"Portfolio"},"renderers":[{"id":"3523"}]},"id":"3545","type":"LegendItem"},{"attributes":{},"id":"3542","type":"UnionRenderers"},{"attributes":{"line_color":"red","line_width":3,"x":{"field":"Date"},"y":{"field":"Adj Close"}},"id":"3549","type":"Line"}],"root_ids":["3473"]},"title":"Bokeh Application","version":"2.2.3"}};
    var render_items = [{"docid":"2af5d7b3-07b5-4d88-98b6-8f0a1b16d380","root_ids":["3473"],"roots":{"3473":"02a4ac36-1987-4a0e-be24-54988f2083aa"}}];
    root.Bokeh.embed.embed_items_notebook(docs_json, render_items);
  }
  if (root.Bokeh !== undefined && root.Bokeh.Panel !== undefined && root['Plotly'] !== undefined ) {
    embed_document(root);
  } else {
    var attempts = 0;
    var timer = setInterval(function(root) {
      if (root.Bokeh !== undefined && root.Bokeh.Panel !== undefined && root['Plotly'] !== undefined) {
        clearInterval(timer);
        embed_document(root);
      } else if (document.readyState == "complete") {
        attempts++;
        if (attempts > 100) {
          clearInterval(timer);
          console.log("Bokeh: ERROR: Unable to run BokehJS code because BokehJS library is missing");
        }
      }
    }, 10, root)
  }
})(window);</script>




```python
print (f'Market Risk: {round(mkt_returns.std() * np.sqrt(252),3)}')
print (f'Goal Portfolio Risk: {round(goal_portfolio_returns.std() * np.sqrt(252),3)}')
```

    Market Risk: 0.258
    Goal Portfolio Risk: 0.101
    


```python
annualized_mkt_return = mkt_returns.mean() * 252
annualized_mkt_risk = np.std(mkt_returns) * np.sqrt(252)

annualized_portfolio_return = goal_portfolio_returns.mean() * 252
annualized_portfolio_risk = np.std(goal_portfolio_returns) * np.sqrt(252)

annualized_mkt_sharpe = annualized_mkt_return / annualized_mkt_risk
annualized_portfolio_sharpe = annualized_portfolio_return / annualized_portfolio_risk

display (f'S&P 500: {annualized_mkt_sharpe}')
display (f'Goal Portfolio Sharpe: {annualized_portfolio_sharpe}')

```


    'S&P 500: 0.8810653234616839'



    'Goal Portfolio Sharpe: 1.7698848540005034'



```python
port_type = get_portfolio_type(total)
print (f'Based on your risk score, your recommended portfolio type was {port_type} and recommended allocations as:\n{allo_dict[port_type]}')
```

    Based on your risk score, your recommended portfolio type was Moderate & Balanced and recommended allocations as:
    {'Stocks': '50%', 'Bonds': '40%', 'Cash': '10%'}
    


```python
goal_allo = print_weights(opt_goal_weights)
```

    
          Weight (%)
    QQQ       20.99
    BND       16.51
    AGG       13.54
    VTV       17.86
    TLT       29.65
    IBB        1.45
    


```python
# Configuring a Monte Carlo simulation to forecast 30 years cumulative returns
simulation_results = pd.DataFrame()
runs = 500
forecasted_days = 3*252
avg = goal_portfolio_returns.mean()
std_dev = goal_portfolio_returns.std()
for iteration in range(runs):
    daily_returns = pd.Series(np.random.normal(avg,std_dev,forecasted_days))
    cum_rets = (1 + daily_returns).cumprod()
    simulation_results[iteration+1] = cum_rets
MC_results = pd.Series(simulation_results.tail(1).values.flatten())

# Use the lower and upper `95%` confidence intervals to calculate the range of the possible outcomes of our $10,000 investments in Coca-Cola and Microsoft stocks
ini_inv = 100000
tbl = np.percentile(MC_results,[0.025,0.975])
ci_lower = round(tbl[0]*ini_inv,2)
ci_upper = round(tbl[1]*ini_inv,2)

# Print results
print(f'There is a 95% chance that an initial investment of {locale.currency(ini_inv,grouping=True)} in the portfolio'
      f' over the next {forecasted_days/252} years will end within in the range of'
      f' {locale.currency(ci_lower,grouping=True)} and {locale.currency(ci_upper,grouping=True)}')
```

    There is a 95% chance that an initial investment of $100,000.00 in the portfolio over the next 3.0 years will end within in the range of $92,558.24 and $112,434.83
    
