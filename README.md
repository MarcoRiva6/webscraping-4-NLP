The `sentiment_analysis` notebook contains the machine learning part to perform sentiment analysis on the reviews.
This is the part 3 of the project.

The `data_collection` notebook contains the code to download the data from Yelp and OpenTable, and generate the two clean csv files.
This is the part 2 of the project.


Within the `sources` folder, the csv files containing the cleaned reviews can be found.

The final processed reviews, with sentiment analysis and language annotation, can be found in the top level of the folder,
with their names' ending in `_with_sentiment.csv`.

To showcase the this explonatory notebook(s), a **Streamlit** app was created. The code for the app can be found in the `streamlit_app` folder.
In Order to run the app, you need to install the required packages in the `requirements.txt` file, and run the following command in the terminal:
```shell
streamlit run streamlit-app.py
```

The other files present in the top level of the folder are intermediate results produces in the first steps of the project.


Please refer to the project's report for detailed explanations of the code and analysis of the results.