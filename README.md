### Application description:
The aim of the project was to create an application that would enable the use of a clustering model to match a user to the appropriate group from a loaded data set (data comes from an anonymized survey) - based on data provided by the user.

### Main functionalities:
* the user filters basic data, such as: age, education, gender, favorite animals or favorite places - corresponding to their preferences,
* then the previously trained clustering model creates the appropriate number of clusters for the survey data and matches the user's preferences to the matching group,
* finally, using LLM, adequate cluster descriptions are generated.

### Dependencies:
* streamlit,
* pycaret,
* plotly,
* pandas.

### Result:
The application is publicly deployed at the link: https://find-friends-app.streamlit.app/
