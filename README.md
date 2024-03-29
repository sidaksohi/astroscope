# [Astroscope is LIVE!](http://astrosco.pe)
#### Astroscope uses your Reddit username to predict your Myers-Briggs personality type. Using your Reddit comments, we use machine learning to individually predict each trait (_Introverted or Extroverted, Sensing or Intuition, etc..._). The model is around 75% accurate for each individual trait. 

## To Do:
- Add caching via DB queries (checks last comment and comment count)
- Add multithreading
- Speed up prediction by making one prediction rather than four (will sacrifice a small amount of accuracy)
- Add forum for each personality type
- Create a celebrity view page where users can input any personality type / astrology sign
- Fix bug with celebrity list having duplicates
- Scrape celebrity pictures

# Celebrity Scraper
The Google Colab notebook used for scraping the celebrity data can be found [here](https://colab.research.google.com/drive/1FUCQgghwkz8-myymJyQv1XwmBxQw9MHR?usp=sharing).

# Landing Page
### Desktop
![Desktop](https://i.imgur.com/Nguf48a.png)
### Mobile
![Mobile](https://i.imgur.com/btqDevU.png)

# Results Page
### Using ElonMusk's Reddit Account
![ElonMuskOfficial](https://i.imgur.com/iT3PYie.png)
![ElonMuskOfficial2](https://i.imgur.com/RptbsJh.png)

### Finding Similar Celebrities
#### Will give you a list of celebrities with the same personality type and astrology sign as you!
![ElonMuskOfficial3](https://i.imgur.com/dxA33QD.png)
![ElonMuskOfficial4](https://i.imgur.com/At5LCCf.png)

##### Clicking on 'Capricorn'
Will bring you to the given Celebrity's _FamousBirthdays_ profile. 
![ElonMuskOfficial5](https://i.imgur.com/v1lcHC9.png)
##### Clicking on 'INTP'
Will bring you to the given Celebrity's _PersonalityDatabase_ profile. 
![ElonMuskOfficial6](https://i.imgur.com/w0m9b93.png)

Made in Flask, using Python 3.8.

Astroscope was developed for fun and does not collect data.
Currently in early development.
Developed by Sidak

