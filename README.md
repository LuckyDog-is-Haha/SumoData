# SumoData
该仓库提供了Sumo仿真实验中的所有数据以及需要文件

路网文件（net.xml）、交通流文件(.trips.xml文件)、仿真文件(.sumocfg文件)。

实验数据下包含三个文件夹，分别为三组实验对应的数据信息。

![image](https://github.com/user-attachments/assets/1f2866e4-3c22-463c-b5fb-87771c8d64bf)

newtry对应文章中所提到的EG——MVRP算法进行的Sumo仿真实验。

randomAssortment对应博弈策略集中随机挑选分配策略进行的Sumo仿真实验。

sumoAlgorithm对应不同传统路径规划算法进行的Sumo仿真。

所有文件夹下命名为map.net.xml的文件均为路网文件，map.osm文件为从OpenStreetMap网站中下载的路网源文件。

newtry文件夹下命名为10_xx(10-100)的文件夹下均含有两个文件，分别为trips.trips_xx.xml和trytrip_xx.xml，包含了车辆的出行信息(From,To)，该出行信息均从Sumo生成的随机车流中选取得出。

![image](https://github.com/user-attachments/assets/b03cb5f2-c5ae-440e-8d60-c9b7e8166833)


trips.trips_xx.xml中包含了十组出行车辆每组xx辆车所有车辆的信息(From,To)

![image](https://github.com/user-attachments/assets/99475290-e640-45b4-810c-d5985f8be6ca)

trytrip_xx.xml则提取了xx组车辆的出行需求信息（From，To）.

![image](https://github.com/user-attachments/assets/e8396e1c-8aa9-4e72-9c41-5bc8b891acfb)

文件夹下的tools文件夹包含了Sumo仿真中所需要的工具包
![image](https://github.com/user-attachments/assets/fb8f9160-78f6-4f1e-9338-f82033ebf4d8)

try_10_xx文件夹下均包含三个文件，即路网文件（net.xml）、EG-MVRP生成策略的路径信息(try.rou_xx.xml文件)、仿真文件(.sumocfg文件)。

randomAssortment文件夹下有一个路网文件以及simulation文件夹

simulation文件夹包含了不同车辆规模下的Sumo仿真文件（每组10-100辆，每10辆车递增）

![image](https://github.com/user-attachments/assets/68129dd8-97a9-4f4c-b889-229e5b30e19a)

以每组十辆车为例其中包含了随机选取策略集中的分配策略进行的五组实验需要的所有Sumo仿真配置文件

![image](https://github.com/user-attachments/assets/4fb245da-d403-47ff-bd3a-66e4b7058c76)

SumoAlgorithm文件夹下有一个路网文件以及出行需求trip文件夹和仿真simulation文件夹

simulation文件夹包含了不同车辆规模下的Sumo仿真文件

![image](https://github.com/user-attachments/assets/c7473e67-6fda-46b1-a87e-04f03e703b6e)

以每组五十辆车为例其中包含了使用传统路径规划算法实验需要的所有Sumo仿真配置文件

![image](https://github.com/user-attachments/assets/9c96c09d-e420-47ba-bbd6-24590012ad81)
