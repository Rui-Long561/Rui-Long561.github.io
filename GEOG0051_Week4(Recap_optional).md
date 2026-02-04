Mining Social and Geographic Datasets
-----------------------------------
GEOG0051 Computer Lab 4 (optional)
-------------------------------

Note: Notebook might contain scripts and instructions adapted from GEOG0115, GEOG0051. 
Contributors: Stephen Law, Mateo Neira, Nikki Tanu, Thomas Keel, Gong Jie, Jason Tang and Demin Hu.


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import geopandas as gpd
import osmnx as ox
import contextily as ctx

```


### Week 1: Python Fundamentals


Week 1 covered the basics of Python programming, including data types, control structures, and working with data using Pandas.

### 1.1 Basic Data Types

Python has several fundamental data types:

| Type | Description | Example |
|------|-------------|--------|
| `str` | Text/strings | `"hello world"` |
| `int` | Whole numbers | `42` |
| `float` | Decimal numbers | `3.14` |
| `bool` | True/False values | `True` |


```python
my_string = "Hello, GEOG0051!"
my_int = 2024
my_float = 51.5074 
my_bool = True

print(f"String: {my_string}, Type: {type(my_string)}")
print(f"Integer: {my_int}, Type: {type(my_int)}")
print(f"Float: {my_float}, Type: {type(my_float)}")
print(f"Boolean: {my_bool}, Type: {type(my_bool)}")
```

    String: Hello, GEOG0051!, Type: <class 'str'>
    Integer: 2024, Type: <class 'int'>
    Float: 51.5074, Type: <class 'float'>
    Boolean: True, Type: <class 'bool'>
    

### 1.2 Data Structures

Python provides several ways to organize data:

| Structure | Description | Syntax | Mutable? |
|-----------|-------------|--------|----------|
| List | Ordered collection | `[1, 2, 3]` | Yes |
| Tuple | Immutable ordered collection | `(1, 2, 3)` | No |
| Dictionary | Key-value pairs | `{'a': 1, 'b': 2}` | Yes |
| Set | Unique elements only | `{1, 2, 3}` | Yes |


```python
cities = ["London", "Paris", "Tokyo", "New York"]
print(f"First city: {cities[0]}")
print(f"Last city: {cities[-1]}")
print(f"Slice [1:3]: {cities[1:3]}")

# Adding to a list
cities.append("Sydney") # add an element to the end of the list
print(f"After append: {cities}")
```

    First city: London
    Last city: New York
    Slice [1:3]: ['Paris', 'Tokyo']
    After append: ['London', 'Paris', 'Tokyo', 'New York', 'Sydney']
    


```python
city_info = {
    'name': 'London',
    'population': 8982000,
    'country': 'UK',
    'coordinates': (51.5074, -0.1278)
}

print(f"City: {city_info['name']}")
print(f"Population: {city_info['population']:,}")
```

    City: London
    Population: 8,982,000
    

### 1.3 Control Structures

Control structures allow you to control the flow of your program.


```python
temperature = 25

if temperature > 30:
    print("It's hot!")
elif temperature > 20:
    print("It's warm.")
else:
    print("It's cool.")
```

    It's warm.
    


```python
cities = ["London", "Paris", "Tokyo"]

for city in cities:
    print(f"Processing: {city}")


# use range()
for i in range(5):
    print(i, end=' ')
```

    Processing: London
    Processing: Paris
    Processing: Tokyo
    0 1 2 3 4 


```python
count = 0
while count < 5:
    print(f"Count: {count}")
    count += 1
```

    Count: 0
    Count: 1
    Count: 2
    Count: 3
    Count: 4
    

### 1.4 Functions

Functions allow you to encapsulate reusable code.


```python
# Defining a function
def calculate_median(numbers):
    """Calculate the median of a list of numbers."""
    sorted_nums = sorted(numbers)
    n = len(sorted_nums)
    
    if n % 2 == 1:
        return sorted_nums[n // 2]
    else:
        mid = n // 2
        return (sorted_nums[mid - 1] + sorted_nums[mid]) / 2

# Using the function
data = [3, 1, 4, 1, 5, 9, 2, 6, 5]
print(f"Data: {data}")
print(f"Median: {calculate_median(data)}")
```

    Data: [3, 1, 4, 1, 5, 9, 2, 6, 5]
    Median: 4
    

### 1.5 Pandas DataFrames

Pandas is the primary library for data manipulation in Python. A DataFrame is a 2D table structure.


```python
data = {
    'City': ['London', 'Paris', 'Tokyo', 'New York', 'Sydney'],
    'Population': [8982000, 2161000, 13960000, 8336000, 5312000],
    'Area_km2': [1572, 105, 2194, 783, 12368],
    'Country': ['UK', 'France', 'Japan', 'USA', 'Australia']
}

df = pd.DataFrame(data)
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>City</th>
      <th>Population</th>
      <th>Area_km2</th>
      <th>Country</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>London</td>
      <td>8982000</td>
      <td>1572</td>
      <td>UK</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Paris</td>
      <td>2161000</td>
      <td>105</td>
      <td>France</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Tokyo</td>
      <td>13960000</td>
      <td>2194</td>
      <td>Japan</td>
    </tr>
    <tr>
      <th>3</th>
      <td>New York</td>
      <td>8336000</td>
      <td>783</td>
      <td>USA</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Sydney</td>
      <td>5312000</td>
      <td>12368</td>
      <td>Australia</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Population</th>
      <th>Area_km2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>5.000000e+00</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>7.750200e+06</td>
      <td>3404.400000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>4.404716e+06</td>
      <td>5072.638889</td>
    </tr>
    <tr>
      <th>min</th>
      <td>2.161000e+06</td>
      <td>105.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>5.312000e+06</td>
      <td>783.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>8.336000e+06</td>
      <td>1572.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>8.982000e+06</td>
      <td>2194.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.396000e+07</td>
      <td>12368.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Adding new columns
df['Density'] = df['Population'] / df['Area_km2']
df['Density'] = df['Density'].round(0) # round to the nearest integer

# Sorting
df_sorted = df.sort_values('Density', ascending=False)
print("Cities sorted by population density:")
df_sorted
```

    Cities sorted by population density:
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>City</th>
      <th>Population</th>
      <th>Area_km2</th>
      <th>Country</th>
      <th>Density</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Paris</td>
      <td>2161000</td>
      <td>105</td>
      <td>France</td>
      <td>20581.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>New York</td>
      <td>8336000</td>
      <td>783</td>
      <td>USA</td>
      <td>10646.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Tokyo</td>
      <td>13960000</td>
      <td>2194</td>
      <td>Japan</td>
      <td>6363.0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>London</td>
      <td>8982000</td>
      <td>1572</td>
      <td>UK</td>
      <td>5714.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Sydney</td>
      <td>5312000</td>
      <td>12368</td>
      <td>Australia</td>
      <td>429.0</td>
    </tr>
  </tbody>
</table>
</div>



### Week 2: Network Science

Week 2 introduced network analysis using NetworkX. Networks (graphs) consist of:
- **Nodes** (vertices): entities in the network
- **Edges** (links): connections between entities

### 2.1 Creating and Visualizing Graphs


```python
G = nx.Graph()

# Add edges
G.add_edges_from([
    ('A', 'B'), ('A', 'C'), ('B', 'C'),
    ('C', 'D'), ('D', 'E'), ('E', 'F'), ('F', 'D')
])


fig, ax = plt.subplots(figsize=(8, 6))
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, 
        node_color='deepskyblue',
        node_size=800,
        with_labels=True,
        font_weight='bold',
        font_color='white',
        edge_color='gray',
        width=2)
plt.title("Simple Network Graph")
plt.show()

print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")
```


    
![png](output_20_0.png)
    


    Number of nodes: 6
    Number of edges: 7
    

### 2.2 Shortest Paths


```python
# Find shortest path
path = nx.shortest_path(G, source='A', target='F')
print(f"Shortest path from A to F: {' → '.join(path)}")



fig, ax = plt.subplots(figsize=(8, 6))

nx.draw(G, pos, 
        node_color='lightgray',
        node_size=800,
        with_labels=True,
        font_weight='bold',
        edge_color='gray',
        width=1)

# Highlight the path
path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
nx.draw_networkx_edges(G, pos, edgelist=path_edges, 
                       edge_color='red', width=4)
nx.draw_networkx_nodes(G, pos, nodelist=path,
                       node_color='red', node_size=800)

plt.title("Shortest Path from A to F (highlighted in red)")
plt.show()
```

    Shortest path from A to F: A → C → D → F
    


    
![png](output_22_1.png)
    


### 2.3 Centrality Measures

Centrality measures identify the most important nodes in a network:

| Measure | Description | Formula Concept |
|---------|-------------|----------------|
| **Degree** | Number of connections | Count of edges |
| **Closeness** | How close to all other nodes | Average distance to all nodes |
| **Betweenness** | How often on shortest paths | Fraction of shortest paths passing through |


```python
# Calculate all three centrality measures
degree_cent = nx.degree_centrality(G)
closeness_cent = nx.closeness_centrality(G)
betweenness_cent = nx.betweenness_centrality(G)


centrality_df = pd.DataFrame({
    'Degree': degree_cent,
    'Closeness': closeness_cent,
    'Betweenness': betweenness_cent
}).round(3)

print("Centrality Measures for Each Node:")
centrality_df.sort_values('Betweenness', ascending=False)
```

    Centrality Measures for Each Node:
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Degree</th>
      <th>Closeness</th>
      <th>Betweenness</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>C</th>
      <td>0.6</td>
      <td>0.714</td>
      <td>0.6</td>
    </tr>
    <tr>
      <th>D</th>
      <td>0.6</td>
      <td>0.714</td>
      <td>0.6</td>
    </tr>
    <tr>
      <th>A</th>
      <td>0.4</td>
      <td>0.500</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>B</th>
      <td>0.4</td>
      <td>0.500</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>E</th>
      <td>0.4</td>
      <td>0.500</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>F</th>
      <td>0.4</td>
      <td>0.500</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Visualize betweenness centrality
fig, ax = plt.subplots(figsize=(8, 6))


node_colors = [betweenness_cent[node] for node in G.nodes()]

nodes = nx.draw_networkx_nodes(G, pos, 
                               node_color=node_colors,
                               node_size=800,
                               cmap='plasma')
nx.draw_networkx_edges(G, pos, edge_color='gray', width=1)
nx.draw_networkx_labels(G, pos, font_weight='bold', font_color='white')

plt.colorbar(nodes, label='Betweenness Centrality')
plt.title("Network colored by Betweenness Centrality")
plt.axis('off')
plt.show()
```


    
![png](output_25_0.png)
    


### 2.4 Community Detection

Communities are groups of nodes that are more densely connected to each other than to the rest of the network.


```python
G_comm = nx.karate_club_graph()  # Famous social network dataset included in NetworkX

# Detect communities
communities = list(nx.algorithms.community.greedy_modularity_communities(G_comm))
print(f"Found {len(communities)} communities")

# Assign a unique number to each community nodes as the color
node_community = {}
for i, comm in enumerate(communities):
    for node in comm:
        node_community[node] = i

# Visualize
fig, ax = plt.subplots(figsize=(10, 8))
pos_comm = nx.spring_layout(G_comm, seed=42) # spring layout is a popular layout for visualizing networks

colors = [node_community[node] for node in G_comm.nodes()]
nx.draw(G_comm, pos_comm,
        node_color=colors,
        cmap='tab10',
        node_size=300,
        with_labels=True,
        font_size=8,
        edge_color='gray',
        width=0.5)

plt.title("Karate Club Network - Community Detection")
plt.show()
```

    Found 3 communities
    


    
![png](output_27_1.png)
    


### Week 3: Spatial Data Analysis

Week 3 introduced spatial data analysis using **GeoPandas** and street network analysis using **OSMnx**.


### 3.1 GeoPandas Basics

GeoPandas extends Pandas to handle geospatial data. A GeoDataFrame is like a regular DataFrame but with a `geometry` column containing spatial information (points, lines, or polygons).


```python
# Create a GeoDataFrame from coordinates
cities_data = {
    'City': ['London', 'Paris', 'Berlin'],
    'Latitude': [51.5074, 48.8566, 52.5200],
    'Longitude': [-0.1278, 2.3522, 13.4050]
}

df_cities = pd.DataFrame(cities_data)

# Convert to GeoDataFrame using gpd.points_from_xy()
gdf_cities = gpd.GeoDataFrame(
    df_cities,
    geometry=gpd.points_from_xy(df_cities.Longitude, df_cities.Latitude),
    crs='EPSG:4326' 
)


gdf_cities
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>City</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>geometry</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>London</td>
      <td>51.5074</td>
      <td>-0.1278</td>
      <td>POINT (-0.1278 51.5074)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Paris</td>
      <td>48.8566</td>
      <td>2.3522</td>
      <td>POINT (2.3522 48.8566)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Berlin</td>
      <td>52.5200</td>
      <td>13.4050</td>
      <td>POINT (13.405 52.52)</td>
    </tr>
  </tbody>
</table>
</div>



### 3.2 OSMnx for Street Networks

OSMnx downloads street networks from OpenStreetMap and returns them as NetworkX graphs. This allows us to apply all the network analysis techniques from Week 2 to real street networks.


```python

G_street = ox.graph_from_address('Trafalgar Square, London', 
                                  dist=500, 
                                  network_type='drive')  

print(f"Street Network:")
print(f"  Nodes (intersections): {G_street.number_of_nodes()}")
print(f"  Edges (street segments): {G_street.number_of_edges()}")
```

    Street Network:
      Nodes (intersections): 170
      Edges (street segments): 316
    


```python

fig, ax = ox.plot_graph(G_street, 
                        figsize=(8, 8),
                        node_size=15,
                        node_color='red',
                        edge_color='gray',
                        bgcolor='white',
                        show=False,
                        close=False)

plt.axis('off')
plt.show()
```


    
![png](output_33_0.png)
    


### 3.4 Shortest Paths in Street Networks

Since OSMnx graphs are NetworkX graphs, we can use `nx.shortest_path()` to find routes between locations.


```python

G_route = ox.graph_from_address('Trafalgar Square, London', dist=1200, network_type='drive')


origin = (51.508056, -0.128056)  # Trafalgar Square
destination = (51.515312, -0.142025)  # Oxford Street


# Note: ox.nearest_nodes takes (G, X=longitude, Y=latitude)
origin_node = ox.nearest_nodes(G_route, origin[1], origin[0])
dest_node = ox.nearest_nodes(G_route, destination[1], destination[0])

# Find shortest path using edge lengths as weights
route = nx.shortest_path(G_route, origin_node, dest_node, weight='length')
route_length = nx.shortest_path_length(G_route, origin_node, dest_node, weight='length')

print(f"Route found: {len(route)} nodes, {route_length:.0f} meters")
```

    Route found: 35 nodes, 1693 meters
    


```python
# Visualize the route
fig, ax = ox.plot_graph_route(G_route, route,
                              route_color='red',
                              route_linewidth=3,
                              node_size=0,
                              bgcolor='white',
                              edge_color='gray',
                              figsize=(10, 10),
                              show=False,  
                              close=False)
ax.set_title(f"Route: Trafalgar Square to Oxford Street ({route_length:.0f}m)")
plt.show()
```


    
![png](output_36_0.png)
    

