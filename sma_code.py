# Imports
import backtrader as bt
import pandas as pd
import quantstats as qs


# The Strategy
class MovingAverageStrategy(bt.Strategy):
    
    params = (
        ('pfast', 50),
        ('pslow', 200)
    )

    def __init__(self):

        self.crossover = dict()
        self.current_position = dict()
        self.available = dict() # 0 --> Money is free. 1 --> Money is in position. 2 --> Money is not in possition or free.
        
        for data in self.datas:
            #Indicators and Signal Generation
            ma_fast = bt.ind.SMA(data, period = self.params.pfast)
            ma_slow = bt.ind.SMA(data, period = self.params.pslow)
            self.crossover[data._name] = bt.ind.CrossOver(ma_fast, ma_slow)
            self.current_position[data._name] = False
            self.available[data._name] = 0

    def execute(self, data):
        name = data._name
        crossover = self.crossover[name]
        position = self.current_position[name]
        if not position:
            if crossover > 0:
                # print("BUY!", name, "on", data.datetime.date(0), "at price", data.close[0])
                # To handle constant weigths for each position
                self.order_target_percent(data=data, target=0.25)
                self.current_position[name] = True
        else:
            if crossover < 0:
                # print("SELL!", name, "on", data.datetime.date(0), "at price", data.close[0])
                self.close(data = data)
                self.current_position[name] = False

    def prenext(self):
        # To handle stocks starting from different dates
        count = [0, 0, 0]
        for data in self.datas:
            if not self.crossover[data._name]:
                continue
            self.execute(data)

    def next(self):
        for data in self.datas:
            self.execute(data)


# Processing Data
def ProcessingData():
    df = pd.read_csv('sma_data.csv')
    df['tradedate'] = pd.to_datetime(df['tradedate'].astype(str), format='%Y%m%d')
    # df = df[df['tradedate'] > '2017-11-01']
    symbs = df['symbol'].unique()

    for symb in symbs:
        d_ = df.loc[df['symbol'] == symb]

        data = bt.feeds.PandasData(
        dataname=d_, datetime='tradedate', close='close', open='close',
        high='close', low='close', volume=None, openinterest=None
        )

        cerebro.adddata(data, name = symb)  # Add the data feed

# Initialize the broker
STARTING_PORTFOLIO_VALUE = 1000000
cerebro = bt.Cerebro()
cerebro.broker.setcash(STARTING_PORTFOLIO_VALUE)

# Run and analyze
ProcessingData()
cerebro.addstrategy(MovingAverageStrategy)
cerebro.addanalyzer(bt.analyzers.Transactions, _name = "trans")
cerebro.addanalyzer(bt.analyzers.PyFolio, _name='PyFolio')
result = cerebro.run() 

# Pyfolio Functions
portfolio_stats = result[0].analyzers.getbyname('PyFolio')
returns, positions, transactions, gross_lev = portfolio_stats.get_pf_items()
returns.index = returns.index.tz_convert(None)

#Terminal Outputs
result[0].analyzers.getbyname('trans').print()
print(f'Starting Portfolio value is {STARTING_PORTFOLIO_VALUE}')
print(f'Ending Portfolio value is {round(cerebro.broker.getvalue(), 2)}')

#Perfomance Stats are saved in a html output
qs.reports.html(returns, output='SMA_Stats.html', title='MA Crossover')

# Plot of Signals
cerebro.plot(volume=False)  
