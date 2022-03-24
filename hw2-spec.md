# HW2: Mandelbulb
Due: <span style="color:red;">Tue, 2022/3/29 23:59</span>

[TOC]
## Problem Description

In this assignment, you are asked to implement a simplest Ray Marching algorithm to render the “Classic” Mandelbulb and parallelize your program with MPI and OpenMP libraries. We hope this assignment helps you to learn the following concepts:

* The differences between **Static Scheduling** and **Dynamic Scheduling**.
* The importance of **Load Balancing**.
* The importance of parallel algorithm.

For more information about Mandelbrot Set and Mandelbulb, please refer to Appendix A.

## Provided Materials

* A sequential version of mendelbulb rendering is provided at `/home/ipc22/share/hw2/sample/hw2.cc`.
* The default makefile is provided at `/home/ipc22/share/hw2/sample/Makefile`.
* The public testcases is provided at `/home/ipc22/share/hw2/testcases`.


## Program Execution and I/O Specifications

The program is executed with the following command:
`./executable $num_threads $x1 $y1 $z1 $x2 $y2 $z2 $width $height $filename`

The arguements' type and their meaning are listed in the table below:

|Argument | type | explanation |
|------- | ------------| ------- |
|$num_threads | int | Number of thread per process |
| $x1 | double | camera position x | 
| $y1 | double | camera position y |
| $z1 | double | camera position z |
| $x2 | double | camera target position x |
| $y2 | double | camera target position y | 
| $z2 | double | camera target position z |
| $width | unsigned int | width of the image |
| $height | unsigned int | height of the image |
| $filename | string | file name of the output PNG image |

**Please note that the output image should be a 32bit PNG image with RGBA channels.**

## Parameter For Mandelbulb Rendering

The table below lists the parameter, along with its default value and explaination, in the sequential version. Please note that if you have a better set of parameter that can produce the same result as the sequential program given any input arguement, please provide the parameters used and argue why yours are better.

|Parameter | Default Value | explaination |
|------- | ------------| ------- |
| power | 8.0 | Power of the equation | 
| md_iter | 24 | The mandelbulb's maximum iteration count  |
| ray_step | 10000 | The maximum step count of ray marching |
| shadow_step | 1500 | The maximum step count of shadow casting |
| step_limiter | 0.2 | The limit length of each step when casting shadow | 
| ray_multiplier | 0.1 | A multiplier for ray marching to prevent over-shooting |
| bailout | 2.0 | The escape radius |
| eps | 0.0005 | The precision of the rendering calculation |
| FOV | 1.5 | Field of view |
| far_plane | 100 | The maximum depth of the scene |

## Library Used

The following libraries listed are the libraries used for the sequential version. All these libraries are already installed on apollo. Please refers to TA’s sample code and `Makefile`.

1. [lodepng](https://github.com/lvandeve/lodepng) - loading/saving PNG images. [C/C++]
2. [GLM](https://github.com/g-truc/glm) - vector/matrix arithmetic. [C++]

## Output Verification

A simple script is provided for you to test the correctness of the rendered image. Once invoked, it will compare the two input images and output the percentage of matching pixels. This script can be invoked by the following command:

`/home/ipc22/share/hw2/hw2-diff a.png b.png`

You can also use `hw2-judge` to check the correctness of your program on all testcases and submit your result to the [scoreboard](https://apollo.cs.nthu.edu.tw/ipc22/scoreboard/hw2/).
<font color=red>**Please note that your homework is graded base on the code submitted to EEClass, the statistics on the scoreboard will not be considered**</font>
 
## Report 

Report must contain the following contents, and you can add more as you like. You can write in either **English or Traditional Chinese**.

1. Name, Student ID:
    - Name: your name.
    - StudentID: your student ID.
2. Explain your implementation, especially in the following aspects:
    - How do you implement your program, what scheduling algorithm did you use: static, dynamic, guided, etc.?
    - How do you partition the task?
    - What techniques do you use to reduce execution time?
    - Other efforts you make in your program.
3. Analysis:
    - Design your own plots to show the load balance of your algorithm between threads/processes.
    - If you have modified the default parameter settings, please also compare the results of the default settings and your settings.
    - Other things worth mentioning.
4. Conclusion:
    - What have you learned from this assignment?
    - What difficulty did you encounter in this assignment?
    - Any feedback or suggestions to this assignment or spec.


## Grading
1. (40%) Correctness:
    - Addition hidden testcases will be used to test your program.
    - If the ouptut png file of your program fails to match 95% of the ground truth, you will get 0 points for that test case.
    - If you have more than 95% pixels correct, the score for each testcase is formulated as follows. Please note that in order to get all points of that testcase, you will at least have to compute 99.6% of pixels correctly:
    $[min(1, {\text{number of correct pixels} \over \text{number of total pixels}} \times {1000 \over 996})]^6 \times \text{score allocated to that testcase}$
    
3. (30%) Performance. Based on the total time you solve all the test cases. Note that you are required to produce the right answer to get the performance points.

5. (30%) Report.

## Submission 

Please submit the following files to EEClass:

1. `hw2.cc`
2. `Makefile` (optional)
3. `report.pdf`

## Reminder

1. Please submit your homework before 3/29 23:59
2. If you spot any problem in this specification or are unsure about the specification, please feel free to **raise your question on EEClass**.

