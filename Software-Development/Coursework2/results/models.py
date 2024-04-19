from django.db import models

# Create your models here.

from django.db import models
  
# Create your models here.

class Uni(models.Model):
    name = models.CharField("Name", max_length=100)
    rank = models.IntegerField("Rank")
    scores_overall = models.FloatField('Overall Score')
    scores_teaching = models.FloatField("Teaching Score")
    scores_research = models.FloatField("Research Score")
    scores_citations = models.FloatField("Citations Score")
    scores_industry_income = models.FloatField("Industrial Income Score")
    scores_international_outlook = models.FloatField("International Outlook Score")
    location = models.CharField("Country", max_length=100)
    stats_number_students = models.IntegerField("Number of Students")
    stats_student_staff_ratio = models.IntegerField("Ratio of students to staff")
    stats_pc_intl_students = models.IntegerField("Percentage of International Students")
    stats_female_male_ratio = models.FloatField("Ratio of male to female Students")
    s_rank = models.IntegerField("Shanghai Ranking", null=True)
    s_rank_country = models.IntegerField(null=True)
    new_rank = models.IntegerField("Your Ranking", default=0, null=True)
    city = models.CharField("Nearest City", max_length=100, null=True)
    quality_of_life_index = models.FloatField("Quality of Life Index")
    safety_index = models.FloatField("Saftey Index")
    climate_index =models.FloatField("Climate Index")
    cost_of_living_index = models.FloatField("Cost of Living Index")
 
    def __str__(self):
        return self.name
    class Meta:
        verbose_name = 'Universities'

class Uni_cs(models.Model):
    name = models.CharField("Name", max_length=100)
    rank = models.IntegerField("Rank")
    scores_overall = models.FloatField('Overall Score')
    scores_teaching = models.FloatField("Teaching Score")
    scores_research = models.FloatField("Research Score")
    scores_citations = models.FloatField("Citations Score")
    scores_industry_income = models.FloatField("Industrial Income Score")
    scores_international_outlook = models.FloatField("International Outlook Score")
    location = models.CharField("Country", max_length=100)
    stats_number_students = models.IntegerField("Number of Students")
    stats_student_staff_ratio = models.IntegerField("Ratio of students to staff")
    stats_pc_intl_students = models.IntegerField("Percentage of International Students")
    stats_female_male_ratio = models.FloatField("Ratio of male to female Students")
    s_rank = models.IntegerField("Shanghai Ranking", null=True)
    s_rank_country = models.IntegerField(null=True)
    new_rank = models.IntegerField("Your Ranking", default=0, null=True)
    city = models.CharField("Nearest City", max_length=100, null=True)
    quality_of_life_index = models.FloatField("Quality of Life Index")
    safety_index = models.FloatField("Saftey Index")
    climate_index =models.FloatField("Climate Index")
    cost_of_living_index = models.FloatField("Cost of Living Index")


    def __str__(self):
        return self.name

class Uni_eng(models.Model):
    name = models.CharField("Name", max_length=100)
    rank = models.IntegerField("Rank")
    scores_overall = models.FloatField('Overall Score')
    scores_teaching = models.FloatField("Teaching Score")
    scores_research = models.FloatField("Research Score")
    scores_citations = models.FloatField("Citations Score")
    scores_industry_income = models.FloatField("Industrial Income Score")
    scores_international_outlook = models.FloatField("International Outlook Score")
    location = models.CharField("Country", max_length=100)
    stats_number_students = models.IntegerField("Number of Students")
    stats_student_staff_ratio = models.IntegerField("Ratio of students to staff")
    stats_pc_intl_students = models.IntegerField("Percentage of International Students")
    stats_female_male_ratio = models.FloatField("Ratio of male to female Students")
    s_rank = models.IntegerField("Shanghai Ranking", null=True)
    s_rank_country = models.IntegerField(null=True)
    new_rank = models.IntegerField("Your Ranking", default=0, null=True)
    city = models.CharField("Nearest City", max_length=100, null=True)
    quality_of_life_index = models.FloatField("Quality of Life Index")
    safety_index = models.FloatField("Saftey Index")
    climate_index =models.FloatField("Climate Index")
    cost_of_living_index = models.FloatField("Cost of Living Index")


    def __str__(self):
        return self.name


