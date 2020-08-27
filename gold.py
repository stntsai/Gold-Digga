# 匯入機器學習套件 sklearn & 預計使用的 decision tree 套件
from sklearn.datasets import load_diabetes
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor


# 用 pandas 將訓練組 & 測試組檔案分別讀進 python
import pandas as pd
df = pd.read_excel(r'data_0609.xlsx')
df_train = pd.read_excel(r'data_train.xlsx')
df_test = pd.read_excel(r'data_test.xlsx')


# 將預計要使用的欄位列出來以便後續變數增減
col=['Lastprice','personal income', 'pce durable', 'pce nondurable', 'pce service',
      'hourwage', 'participate rate', 'House(catershiller', 'corecpi',
       'cpifood', 'cpienergy', 'gdp', 'VTI-Close', 'VTI-adjClose', 'VTI-Vol',
       'SPX-Close', 'SPX-adjClose', 'SPX-Vol', 'DJI-Close', 'DJI-adjClose',
       'DJI-Vol', 'NASDAQ', 'VIX', 'DGS10', 'DGS5', 'DGS1', 'DGS3MO', 'DGS1MO',
       'T10YIE', 'DAAA', 'T10_DAAA', 'DBAA', 'T10Y3M', 'T10Y2Y', 'TEDRATE',
       'DFF', 'HighYieldSPREAD', 'VIX.1', 'value(u.s cent per pound)', 
       'GS: wealth management: (全球)', 'GS: typhoon: (全球)', 'GS: stock: (全球)',
       'GS: rolls royce: (全球)', 'GS: rolex: (全球)', 'GS: oil: (全球)',
       'GS: layoff: (全球)', 'GS: job: (全球)', 'GS: immigrate: (全球)',
       'GS: Gold: (全球)', 'GS: election: (全球)', 'GS: earthquake: (全球)',
       'Prod_Merc_Positions_NET', 'Swap__Positions_NET',
       'M_Money_Positions_NET', 'ICSA', 'CCSA']


# 若將黃金價格用抽樣跑預測顯然與現實世界不符，故我們手動將資料分為訓練組前7年，
# 測試組後3年，將資料分別匯入並命名。
x_train = df_train[col]
y_train = df_train['XAUUSD close']
x_test = df_test[col]
y_test = df_test['XAUUSD close']


#開始用DecisionTreeRegressor跑模型，訓練電腦針對訓練組做最佳擬合
model = DecisionTreeRegressor(random_state=1, max_depth = 8, min_samples_leaf = 5)
model.fit(x_train, y_train)


# 讓電腦用訓練組得出的擬合結果，對測試組的資料跑黃金價格的預測
y_pred = model.predict(x_test)


# 預測結果出來之後，將現實黃金價格走勢 & 預測價格走勢放在一起，方便檢視預測準確度
predict_doc = pd.DataFrame()
predict_doc['Date'] = df_test['Date']
predict_doc['y'] = df_test['XAUUSD close']
predict_doc['y_pred'] = y_pred


#匯出檔案繪製預測圖形以檢視測試準確度
predict_doc.to_csv('predict_doc.csv') 


#import 繪圖套件 graphviz
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus  


#製作決策樹，並匯出圖檔
dot_data = StringIO()
export_graphviz(model, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = col)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('Decision Tree pre-adjust.png')
Image(graph.create_png())




