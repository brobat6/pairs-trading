from asyncore import close_all
import pandas as pd
import numpy as np
import datetime
from sklearn.model_selection import train_test_split
import backtrader as bt
from statsmodels.tsa.stattools import coint
import quantstats as qs

# function to find cointegrated pairs
def find_cointegrated_pairs(data):
    n = data.shape[1]
    pvalue_matrix = np.ones((n, n))
    keys = data.keys()
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            result = coint(data[keys[i]], data[keys[j]])
            pvalue_matrix[i, j] = result[1]
            if result[1] < 0.05:
                pairs.append((keys[i], keys[j], result[1]))
    return pairs


# calculate z-score
def zscore(series):
    return (series - series.mean()) / np.std(series)


# To get pairs with no common stocks
def get_distinct_pairs(pairs):
    pairs_used = dict()
    ans = []

    for pair in pairs:
        if (pairs_used.get(pair[0], 0) + pairs_used.get(pair[1], 0) == 0):
            ans.append(pair)
            pairs_used[pair[0]] = 1
            pairs_used[pair[1]] = 1
    
    return ans


class PairsTrading(bt.Strategy):
    
    def __init__(self):
        self.current_position = dict()

        # Number of stock pairs
        self.num = len(self.datas) // 3

        self.zmean = [0] * self.num
        self.zstddev = [0] * self.num
        self.status = [0] * self.num

        # Calculating % of portfolio to invest in each stock
        self.pstock = 0.5 / self.num
        
        for i in range(self.num):

            # Pair  0   1   2   x
            # Ratio 2   5   8   3x+2
            # Buflen gives length of the data
            self.len = self.datas[3 * i + 2].close.buflen()

            # Calculating mean
            cnt = 0
            sum = 0
            for j in range(self.len):
                sum += self.datas[3 * i + 2].close[j]
                cnt += 1
            self.zmean[i] = sum / cnt

            # Calculating stddev
            sum = 0
            for j in range(self.len):
                sum += (self.datas[3 * i + 2].close[j] - self.zmean[i]) * (self.datas[3 * i + 2].close[j] - self.zmean[i])
            self.zstddev[i] = np.sqrt(sum / cnt)
        
         
        for data in self.datas:
            self.current_position[data._name] = False
        
    
    def next(self):

        # Data is in the form Pair0StockA,Pair0StockB,Pair0Zscore, Pair1StockA,Pair1StockB....
        # Pairs     0       1       2       i
        # Data used 0,1,2   3,4,5   6,7,8   3i, 3i+1,3i+2
        
        for i in range(self.num):
            position1 = self.current_position[self.datas[3 * i]._name]
            position2 = self.current_position[self.datas[3 * i + 1]._name]

            zlow = self.zmean[i] - self.zstddev[i]
            zup = self.zmean[i] + self.zstddev[i]
        
            if (self.datas[3 * i + 2].close < zlow) and (self.status[i] != 1):
                
                self.status[i] = 1

                if position2:
                    # Selling asset2
                    self.close(data = self.datas[3 * i + 1])
                    self.current_position[self.datas[3 * i + 1]._name] = False

                if not position1:
                    # Buying asset1
                    self.order_target_percent(data=self.datas[3 * i], target = self.pstock)
                    self.current_position[self.datas[3 * i]._name] = True

            
            elif (self.datas[3 * i + 2].close > zup) and (self.status != -1):
                
                self.status[i] = -1
                
                if position1:
                    # Selling asset1
                    self.close(data = self.datas[3 * i])
                    self.current_position[self.datas[3 * i]._name] = False

                if not position2:
                    # Buying asset2
                    self.order_target_percent(data=self.datas[3 * i + 1], target = self.pstock)
                    self.current_position[self.datas[3 * i + 1]._name] = True
            
            elif (self.datas[3 * i + 2].close < zup) and (self.datas[3 * i + 2].close > zlow):

                if self.status[i] == -1:
                
                    if not position1:
                        self.order_target_percent(data=self.datas[3 * i], target = self.pstock)
                        self.current_position[self.datas[3 * i]._name] = True

                    if position2:
                        self.close(data = self.datas[3 * i + 1])
                        self.current_position[self.datas[3 * i + 1]._name] = False

                if self.status[i] == 1:

                    if not position2:
                        self.order_target_percent(data=self.datas[3 * i + 1], target = self.pstock)
                        self.current_position[self.datas[3 * i + 1]._name] = True

                    if position1:
                        self.close(data = self.datas[3 * i])
                        self.current_position[self.datas[3 * i]._name] = False
                
                self.status[i] = 0

    def stop(self):

        for i in range(self.num):
            self.close(data = self.datas[3 * i])
            self.close(data = self.datas[3 * i + 1])




def main():
    #read metadata csv
    nifty_meta = pd.read_csv(r'pairs_nifty_meta.csv')

    # get the ticker list with industry is equal to FINANCIAL SERVICES
    tickers = list(nifty_meta[nifty_meta.Industry == 'FINANCIAL SERVICES'].Symbol)

    # start and end dates for backtesting
    fromdate = datetime.datetime(2010, 1, 1)
    todate = datetime.datetime(2020, 6, 15)

    # read back the pricing data
    prices = pd.read_csv(r'pairs_prices.csv', index_col=['ticker','date'], parse_dates=True)


    idx = pd.IndexSlice
    # remove tickers where we have less than 10 years of data.
    min_obs = 2520
    nobs = prices.groupby(level='ticker').size()
    keep = nobs[nobs>min_obs].index
    prices = prices.loc[idx[keep,:], :]

    # final tickers list
    TICKERS = list(prices.index.get_level_values('ticker').unique())


    # unstack and take close price
    close = prices.unstack('ticker')['close'].sort_index()
    close = close.dropna()

    # print(close.first('1Y'))

    STARTING_PORTFOLIO_VALUE = 200_000
    portfolio_val = STARTING_PORTFOLIO_VALUE
    print(f'Starting Portfolio value is {portfolio_val}')

    # Starting year
    year1 = 2010

    # To store results of each year
    returns_main = pd.Series()

    for i in range(10, 2, -1):


        train_size = close.first('2Y').shape[0]  # Train size = Rows of first two years

        # Start period and end period for train and test close
        print(f'Train period: {year1} to {year1 + 1}')
        print(f'Test year: {year1 + 2}')
        
        # Train test split
        train_close, test_close = train_test_split(close.first('3Y'), train_size= train_size, shuffle=False)


        # calculate p-values
        pairs = find_cointegrated_pairs(train_close)
        pairs = sorted(pairs, key = lambda x: x[2])
        
        # No two pairs have a common stock
        pairs = get_distinct_pairs(pairs)
        
        # Number of pairs we get for the present year
        print(f'Number of pairs: {len(pairs)}')
        print()

        cerebro = bt.Cerebro()
        
        for pair in pairs:

            df1 = pd.DataFrame()
            df1['close'] = test_close[pair[0]]
            df2 = pd.DataFrame()
            df2['close'] = test_close[pair[1]]

            # Every 3rd data is the zscore of the ratio of closing prices of the previous two datas
            df3 = pd.DataFrame()
            df3['close'] = zscore(test_close[pair[0]] / test_close[pair[1]])

            data1 = bt.feeds.PandasData(
                dataname = df1,
                datetime = None,
                open = 0,
                close = 0,
                high = 0,
                low = 0,
                volume = None,
                openinterest = None
            )

            data2 = bt.feeds.PandasData(
                dataname = df2,
                datetime = None,
                open = 0,
                close = 0,
                high = 0,
                low = 0,
                volume = None,
                openinterest = None 
            )

            data3 = bt.feeds.PandasData(
                dataname = df3,
                datetime = None,
                open = 0,
                close = 0,
                high = 0,
                low = 0,
                volume = None,
                openinterest = None 
            )

            cerebro.adddata(data1, name = pair[0])
            cerebro.adddata(data2, name = pair[1])
            cerebro.adddata(data3, name = 'Pair')


        
        cerebro.broker.set_cash(portfolio_val)
        cerebro.addstrategy(PairsTrading)
        cerebro.addanalyzer(bt.analyzers.Transactions, _name = "trans")
        cerebro.addanalyzer(bt.analyzers.PyFolio, _name='PyFolio')

        result = cerebro.run()

        # Pyfolio Functions
        portfolio_stats = result[0].analyzers.getbyname('PyFolio')
        returns, positions, transactions, gross_lev = portfolio_stats.get_pf_items()
        returns.index = returns.index.tz_convert(None)
        returns_main = returns.append(returns_main)

        #Terminal Outputs
        result[0].analyzers.getbyname('trans').print()


        # Preparation for next year
        portfolio_val = round(cerebro.broker.getvalue(), 2)
        close = close.last(f'{i}Y')     # Removing the first one year of data
        year1 += 1

    print(f'Ending Portfolio value is {round(cerebro.broker.getvalue(), 2)}')
    
    # Perfomance Stats are saved in a html output
    returns_main.sort_index(inplace = True)
    qs.reports.html(returns_main, output='Pairs_Stats.html', title='Pairs Trading')



if __name__ == '__main__':
    main()