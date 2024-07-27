import pandas as pd
import numpy as np


class FeaturesExtractionLOB:
    
    def __init__(self):
        pass
    
    @staticmethod
    def extract(df):
        # Mid price
        self.df['MP'] = (self.df['P_Ask_1'] + self.df['P_Bid_1']) / 2
        # Wheigted mid price
        self.df['WMP'] = (self.df['P_Ask_1'] * self.df['V_Ask_1'] + self.df['P_Bid_1'] * self.df['V_Bid_1']) / (self.df['V_Ask_1'] + self.df['V_Bid_1'])
        # Bid-Aks spread
        self.df['Spread'] = self.df['P_Ask_1'] - self.df['P_Bid_1']
        # Order Book Imbalance
        self.df['Imbalance'] = (self.df['V_Ask_1'] - self.df['V_Bid_1']) / (self.df['V_Ask_1'] + self.df['V_Bid_1'])
        # Bid, Ask and Total Volume
        self.df['Depth_Bid'] = self.df.filter(regex='V_Bid').sum(axis=1)
        self.df['Depth_Ask'] = self.df.filter(regex='V_Ask').sum(axis=1)
        self.df['Depth_Total'] = self.df['Depth_Bid'] + self.df['Depth_Ask']
        
        return self.df
