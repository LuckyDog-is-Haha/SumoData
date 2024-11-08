# SumoData
The repository provides all the data and required files for Sumo simulation experiments

Road network file (net.xml), traffic flow file (.trips.xml), simulation file (.sumocfg).

The experimental data contains three folders, which are the corresponding data information of the three groups of experiments.

![image](https://github.com/user-attachments/assets/1f2866e4-3c22-463c-b5fb-87771c8d64bf)

newtry corresponds to the Sumo simulation experiment of EG-MVRP algorithm mentioned in the article.

randomAssortment is a Sumo simulation experiment based on random selection and distribution strategy of game strategy set.

The sumoAlgorith folder corresponds to Sumo simulations of different traditional path planning algorithms.

The file named map.net.xml in all folders is the road network file, and the map.osm file is the road network source file downloaded from the OpenStreetMap website.

newtry folders named 10_xx(10-100) in the newtry folder each contain two files, namely trips.trips_xx.xml and trytrip_xx.xml, which contain the travel information of vehicles (From,To), which are selected from the random traffic flow generated by Sumo.

![image](https://github.com/user-attachments/assets/b03cb5f2-c5ae-440e-8d60-c9b7e8166833)

trips_xx.xml contains the information of all vehicles in ten groups of travel vehicles, each group of xx vehicles (From,To).

![image](https://github.com/user-attachments/assets/99475290-e640-45b4-810c-d5985f8be6ca)

trytrip_xx.xml extracted the travel demand information of xx group vehicles(From，To).

![image](https://github.com/user-attachments/assets/e8396e1c-8aa9-4e72-9c41-5bc8b891acfb)

The tools folder under this folder contains the toolkits needed for Sumo emulation

![image](https://github.com/user-attachments/assets/fb8f9160-78f6-4f1e-9338-f82033ebf4d8)

The try_10_xx folder contains three files: net.xml, EG-MVRP generation policy path information (try.rou_xx.xml file), and simulation file (.sumocfg file).

randomAssortment folder contains a road network file and the simulation folder

The simulation folder contains Sumo simulation files for different vehicle sizes (10-100 vehicles per group, increments per 10 vehicles).

![image](https://github.com/user-attachments/assets/68129dd8-97a9-4f4c-b889-229e5b30e19a)

Each group of ten cars contains all the Sumo simulation profiles required for the five sets of experiments with randomly selected allocation strategies in the policy set.

![image](https://github.com/user-attachments/assets/4fb245da-d403-47ff-bd3a-66e4b7058c76)

SumoAlgorithm folder contains a road network file as well as a trip folder for travel requirements and a simulation folder.

simulation folder contains Sumo simulation files for different vehicle sizes.

![image](https://github.com/user-attachments/assets/c7473e67-6fda-46b1-a87e-04f03e703b6e)

Each group of 50 vehicles contains all the Sumo simulation profiles needed for the experiment using traditional path planning algorithms

![image](https://github.com/user-attachments/assets/9c96c09d-e420-47ba-bbd6-24590012ad81)
