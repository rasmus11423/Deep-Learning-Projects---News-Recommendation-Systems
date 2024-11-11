# Deep-Learning-Projects---News-Recommendation-Systems
 The aim of this project is to develop effective and responsible news recommender systems. Students will predict which news articles a user is most likely to click on during a specific session, using data such as the user's click history, session details (e.g., time and device used), personal metadata (e.g., gender and age), and a list of candidate news articles. By building models that rank articles based on individual preferences, students will tackle challenges such as modeling user interests from implicit behavior, considering how the current news agenda influences these interests, and managing the rapid decay of news items. Additionally, the project encourages exploring the ethical challenges of recommender systems, such as their impact on news diversity and alignment with editorial values. The ultimate goal is to achieve the best possible performance on a hidden test set in the RecSys Challenge 2024. Furthermore, students can take inspiration from the teams who participated in the RecSys ’24 Challenge (https://recsys.eb.dk/workshop). The dataset is available at https://recsys.eb.dk. The project will be co-supervised by Johannes Kruse (jkru@dtu.dk) and Jes Frellsen (jefr@dtu.dk).


# Initiation of Enviroment

Before hand you must have installed poetry and have python3.11 or newer. To generate the virtual enviroment, run the following (within the project directory). 

```
python3.11 -m venv .venv
cd code
source .venv/bin/activate
poetry update
```

You can now add packages using poetry ```poetry add [package name]```. Poetry will download all the packages into the enviroment, and when the enviroment is deleted, the packages are removed. Poetry creates a file, pyproject.toml, which is part of the .gitignore file, it should be there but should not be uploaded to git.