# SensatUrban_albert
UCL CGVI

Randlanet 100 epochs model and log file(baseline): 
https://drive.google.com/file/d/1wJlDjykVdnZBe4RXw6J6l01bmBAM2XNr/view?usp=sharing

Pointtrans 50 epochs model and log file(final model): 

**Model Performance Comparison**

Test set scores:

![image](https://user-images.githubusercontent.com/43678364/202578853-c6b3a7ff-2c3a-4b31-bf0c-1c5f99ac7140.png)

Baseline and final model comparison:

![image](https://user-images.githubusercontent.com/43678364/202578883-24e1bbd3-5f41-4de7-84b4-94b4db5eed86.png)


**Result:**

![pointtrans](https://user-images.githubusercontent.com/43678364/202577950-08787ebb-f09d-4ba8-8cbc-137644d5424d.gif)

**Details:**

![image](https://user-images.githubusercontent.com/43678364/202578491-8505aa80-7a03-477e-98ff-c1dce0464554.png)

For robust, unique architecture like the big stadium, the randlanet misclassifies many points into the other classes, whereas the final model made fewer mistakes.

The final model correctly classifies more points than the baseline for high morphable classes like water. In baseline, many points in Water are classified into FootPath(pink).
