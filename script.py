import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def convert_csv_to_dataframe(filename):
    # Read csv data
    df = pd.read_csv("StudentStudyData.csv")
    if 'Unnamed:0' in df.columns:
        df = df.drop(columns=['Unnamed:0'])
    return df

def load_data(df):

    # Train and test  the linear Regression Model
    X  = df[['HoursStudied']]
    y = df['ExamScore']
    return X,y

def split_data_for_training_testing(X, y, test_size, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def create_train_model(X_train, y_train, optimization_algo):
    if optimization_algo == "Linear":
        model = LinearRegression()
        used_optimization_algo = "LinearRegression"
    else: 
        used_optimization_algo = "Unknown"
        raise ValueError("Unsupported optimization algo provided by the user.")
    
    model.fit(X_train, y_train)
    return model, used_optimization_algo

def make_predictions(model, user_input):
    predicted_output_value = model.predict(user_input)
    return predicted_output_value

def model_performance(y_test, y_pred, optimization_algo):
    print(f"R2 Score using {optimization_algo} algorithm: {r2_score(y_test, y_pred):.2f}")
    print(f"MSE value using {optimization_algo} algorithm: {mean_squared_error(y_test, y_pred):.2f}")

def plot_actual_against_predicted_output(X_test, y_test, y_pred):
    # Visual the Regression line:
    plt.scatter(X_test, y_test, color='blue', label='Actual')
    plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted')
    plt.title("Hours Studied vs Exam Score")
    plt.xlabel("Hours Studied")
    plt.ylabel("Exam Score")
    plt.legend()
    plt.savefig("RegressionLine.png")
    plt.close()

def create_csv_file(df, filename):
    # Create a new csv file, and write data to it partially.
    df.head(10).to_csv(filename, index=False)
  

def add_new_csv_field(df, field):
    # Reading ExamScore data, and dividing it into categories as Low, Medium, High, UltraHigh
    df[field] =  pd.cut(x=df['ExamScore'], bins=[0,50,75,100, 200], labels=['Low', 'Medium', 'High', 'UltraHigh'])
    return df

def generate_pairplot(df,hue):
    """To generate a pairplot for the provided dataframe, and by categorizing various scores based on the value of hue parameter."""
    g = sns.pairplot(data=df,hue=hue, kind='scatter')
    g.fig.suptitle("Pairplot of different Score Group")
    plt.savefig("ScorePairplot.png")
    plt.close()

def main():
    df = convert_csv_to_dataframe(filename = "StudentStudyData.csv")
    X,y = load_data(df=df)
    X_train, X_test, y_train, y_test = split_data_for_training_testing(X, y, test_size=0.2, random_state= 42)
    model, optimization_algo_used = create_train_model(X_train=X_train, y_train=y_train, optimization_algo="Linear")
    y_pred = make_predictions(model=model, user_input=X_test)
    plot_actual_against_predicted_output(X_test=X_test, y_test=y_test, y_pred=y_pred)
    model_performance(y_test=y_test, y_pred=y_pred, optimization_algo=optimization_algo_used)

    create_csv_file(df=df, filename="PartialData.csv")
    updated_df = add_new_csv_field(df=df, field="ScoreGroup")
    generate_pairplot(df=updated_df, hue='ScoreGroup')



if __name__ == "__main__":
    main()