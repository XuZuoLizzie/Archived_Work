# Software-Development

1- Install python 3 and pip on Windows, Follow the link below:
https://www.youtube.com/watch?v=gFNApsyhpKk

2- Install Ruby and RubyGems on Windows
https://forwardhq.com/help/installing-ruby-windows

2-  Follow the link bellow to install MongoDB:
https://www.guru99.com/installation-configuration-mongodb.html

3- As you install MongoDB, MongoDG Compass will be automatically installed.

4- In CMD:
gem install mongo

5- Install django:

pip install django

pip install djongo

6- Install modules

pip install djangorestframework

pip install requests

pip install beautifulsoup4

pip install numpy


7- Open new window in your command prompt and type and keep the window open:
"C:\Program Files\MongoDB\Server\4.0\bin\mongo.exe"

8- Open MongoDB Compass connect then, Create Database:

Database Name: findu-db

Collection Name: results_uni

9- In differnt window of your command prompt, cd to the project directory\DataInserts folder and then run:

python insert-unis.py

python insert-cities.py

python insert-city_data.py


10- Refresh MongoDB Compass

11- cd ..

12- 
python manage.py makemigrations

python manage.py migrate

python manage.py runserver

13- Copy the followig link and paste it in your browser:
http://127.0.0.1:8000/
