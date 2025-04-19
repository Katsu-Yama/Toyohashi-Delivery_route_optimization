import folium
import networkx as nx
import geopandas as gpd
import pandas as pd
import osmnx as ox
import numpy as np
from sklearn import *
import pandas as pd
import geopandas as gpd
import os
import json
from geopy.distance import geodesic


dir_name = "/Q-quest2024/teamC/"
root_dir = "/content/drive/MyDrive/" + dir_name
node_data= "Node/kyoten_geocode_Revised.json"


df=pd.read_json(root_dir+node_data)


place = {'city' : 'Odawara',
         'state' : 'Kanagawa',
         'country' : 'Japan'}
G = ox.graph_from_place(place, network_type='drive')

# print(G)

# ox.plot_graph(G)


# ノードリストの作成。'M'または'K'を含むノードを抽出
node_list=pd.concat([df[df['Node'].str.contains('M') ],df[df['Node'].str.contains('K')]], ignore_index=True)

# 距離行列の初期化。ノード数×ノード数のゼロ行列を作成
distance_matrix=np.zeros((len(node_list),len(node_list)))
# パスリストの初期化。空のリストを作成
path_list=[]

# 各始点ノードについて
for i in range(len(node_list)):
  # 始点ノード名、緯度、経度を取得
  start_node_name=node_list.iloc[i]['Node']
  start_lat = node_list.iloc[i]['緯度']
  start_lon = node_list.iloc[i]['経度']
  #print(start_lat)
  #print(start_lon)
  # 始点ノードのOSM IDを取得
  start_id = ox.distance.nearest_nodes(G,start_lon,start_lat)

  # 各終点ノードについて
  for j in range(len(node_list)):
    # 終点ノード名、緯度、経度を取得
    goal_node_name=node_list.iloc[j]['Node']
    goal_lat = node_list.iloc[j]['緯度']
    goal_lon = node_list.iloc[j]['経度']
    # 終点ノードのOSM IDを取得
    goal_id = ox.distance.nearest_nodes(G,goal_lon,goal_lat)

    # 最短経路を計算
    route = ox.shortest_path(G, start_id, goal_id)
    # 最短経路の距離を計算
    distance = nx.shortest_path_length(G, start_id, goal_id, weight='length')

    # 距離行列に距離を格納
    distance_matrix[i][j]=distance
    # パス情報を辞書に格納
    path={}
    path['m']=i
    path['n']=j
    path['start_node']=start_node_name
    path['goal_node']=goal_node_name
    path['route']=route
    path['distance']=distance
    # パスリストにパス情報を追加
    path_list.append(path)

    #print(f'start_no.{i}({start_node_name})-goal_no.{j}({goal_node_name})->{distance}')


file_path=root_dir+"distance_matrix.csv"
np.savetxt(file_path,distance_matrix,delimiter=",")


file_path=root_dir+"path_list.json"
with open(file_path, 'w') as f:
    json.dump(path_list, f)


