# Mini_Project_2
Skip to content
 
Search or jump to…

Pull requests
Issues
Marketplace
Explore
 @amzjctuan Sign out
0
0 0 amzjctuan/EC601-Mini_Project
 Code  Issues 0  Pull requests 0  Projects 0  Wiki  Insights  Settings
EC601-Mini_Project/ 
README.md
  or cancel
    
 
1
# EC601-Mini_Project: API Tutorial
2
2018 Fall - Boston University - EC601 #project_1
3
API
4
​
5
1. Twitter API: 
6
   Use twitter API to grab the photos form the twitter account.
7
   
8
2. ffmpeg:
9
   Transfer the photos downloaded from twitter into video.
10
   
11
3. Google Visial API:
12
   Describe each of the photos from the video made in the previous step.
13
​
14
# System Environment 
15
​
16
- Python 3.6.5
17
- ffmpeg 3.4.4
18
- Tweepy 3.6.0
19
- google-cloud-videointelligence 1.3.0
20
- Ubuntu 18.04.1 LTS
21
​
22
   
23
# Program Description
24
​
25
There are three functions made in the mini_project_api.py.
26
- tweet_api
27
- ffmpeg
28
- google_analyze
29
​
30
You need to type the twitter account that you wanna grab the photos from.
31
Then, it will automatically help you to download all the photos from the account and translate each of the photos.
32
​
33
​
34
​
35
​
36
# Note
37
Google application credentials:
38
```
39
export GOOGLE_APPLICATION_CREDENTIALS='google_application_credentials.json'
40
```
41
Make sure file 'google_application_credentials.json' has correct path.
42
​
43
​
44
​
45
​
@amzjctuan
Commit changes

Update README.md

Add an optional extended description…
  Commit directly to the master branch.
  Create a new branch for this commit and start a pull request. Learn more about pull requests.
 
© 2018 GitHub, Inc.
Terms
Privacy
Security
Status
Help
Contact GitHub
Pricing
API
Training
Blog
About
Press h to open a hovercard with more details.
