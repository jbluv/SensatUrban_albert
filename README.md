# SensatUrban_albert

**UCL CGVI Absrtact**

The goal of this project is to systematically study the 3D semantic segmentation and improve the performance of the network on the urban-scaled point cloud dataset "SensatUrban”. Ideally, train a robust network that can fuse and understand noisy urban data from multiple sources and create a semantically labeled 3D model of urban scale. 

In this project we studied and evaluated three state-of-art network structures that can be used in semantic segmentation: randlanet, multi-head attention layer and point transformer. 

Various data enhancement techniques based on KPConv and PointNext are also applied to the network. Eventually, compare to the baseline, we achieved an 8.0% of mIoU increasement on the test set.

![image](https://user-images.githubusercontent.com/43678364/202579053-afdfb958-6c4f-40af-a9c2-8abaff6e6592.png)

Thesis:
https://drive.google.com/file/d/1BsRbLxOYe0Xi1sOM2a7SibiqO4zHw2oH/view?usp=sharing

Randlanet 100 epochs model and log file(baseline): 
https://drive.google.com/file/d/1wJlDjykVdnZBe4RXw6J6l01bmBAM2XNr/view?usp=sharing

Pointtrans 50 epochs model and log file(final model): 
https://drive.google.com/file/d/1b9M7IMOTrEX5qt80tZK8f4R23upv6f_i/view?usp=share_link

**Overall Structure Of The Network:**

![image](https://user-images.githubusercontent.com/43678364/202579348-560bf739-397c-4c5c-8233-b0bb6b635c6d.png)

**Result:**

![pointtrans](https://user-images.githubusercontent.com/43678364/202577950-08787ebb-f09d-4ba8-8cbc-137644d5424d.gif)

**Details:**

![image](https://user-images.githubusercontent.com/43678364/202578491-8505aa80-7a03-477e-98ff-c1dce0464554.png)

For robust, unique architecture like the big stadium, the randlanet misclassifies many points into the other classes, whereas the final model made fewer mistakes.

The final model correctly classifies more points than the baseline for high morphable classes like water. In baseline, many points in Water are classified into FootPath(pink).

**Model Performance Comparison**

Test set scores:

![image](https://user-images.githubusercontent.com/43678364/202578853-c6b3a7ff-2c3a-4b31-bf0c-1c5f99ac7140.png)

Baseline and final model comparison:

![image](https://user-images.githubusercontent.com/43678364/202578883-24e1bbd3-5f41-4de7-84b4-94b4db5eed86.png)
